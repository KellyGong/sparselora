from typing import Any, Optional

import torch
from torch import nn

from .base import SparseModule, reft_forward
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

__all__ = ["SparseLlamaMLP"]


silu_mul = LigerSiLUMulFunction.apply

class SparseLlamaMLP(SparseModule):
    inherited_attributes = [
        "gate_proj",
        "up_proj",
        "down_proj",
        "act_fn",
    ]

    def __init__(self, base: nn.Module, *, name: str, idx: int, sparsity: float, cfg, **kwargs) -> None:
        super().__init__(base)
        self.sparsity = sparsity
        self.layer_name = name
        self.layer_idx = idx
        if self.sparsity > 0:
            self.load_predictor(base,cfg)
            self.enable_static = kwargs.get("enable_static", False)
            if self.enable_static:
                channel_act = kwargs.get("channel_act", None)
                if channel_act is None:
                    self.channel_act = torch.zeros(base.up_proj.out_features)
                else:
                    self.channel_act = channel_act
                    # sorted the channel_act values and get the top-k indices with largest values
                    _, self.static_indices = torch.topk(torch.tensor(self.channel_act), int(base.up_proj.out_features * (1 - self.sparsity)), sorted=False)
                    self.static_indices = self.static_indices.to('cuda')
            
            self.dense_stream, self.sparse_stream = torch.cuda.Stream(), torch.cuda.Stream()
        
        if cfg.peft == "reft" and cfg.reft_module_out:
            self.reft = True
            self.reft_lora = nn.Sequential(
                                nn.Linear(base.down_proj.out_features, cfg.rank, bias=True),
                                nn.Dropout(p=0.1),
                                nn.Linear(cfg.rank, base.down_proj.out_features, bias=False)
                             )
            nn.init.zeros_(self.reft_lora[0].weight)
            nn.init.zeros_(self.reft_lora[2].weight)

    def kernel_forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, static_indices = None) -> torch.Tensor:

        if masks is None:  # No Split
            dynamic_indices = self.pred_mlp(x)
            sparse_indices = dynamic_indices
            if static_indices is not None:
                sparse_indices = torch.unique(torch.cat((dynamic_indices, static_indices), dim=0))
            return self._forward_block(x, sparse_indices=sparse_indices)
        
        else: #* Split
            sparse_x, dense_x = self.token_splits(x, masks)

            with torch.cuda.stream(self.sparse_stream):
                dynamic_indices = self.pred_mlp(x)
                if static_indices is not None:
                    sparse_indices = torch.unique(torch.cat((dynamic_indices, static_indices), dim=0))
                else:
                    sparse_indices = dynamic_indices
                sparse_x = self._forward_block(sparse_x, sparse_indices=sparse_indices)
            
            with torch.cuda.stream(self.dense_stream):
                dense_x = self._forward_block(dense_x)
            
            # torch.cuda.synchronize()
            
            #* Token Order: [Sparse | Dense] --> [In | Out]
            out = self.token_join(sparse=sparse_x, dense=dense_x, masks=masks)
            return out
         
        
    def mask_forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, sparse_indices = None) -> torch.Tensor:
        
        if masks is None: #* No Split
            binary_mask  = torch.zeros_like(x)
            binary_mask[:, :, sparse_indices] = 1
            x = x * binary_mask
            return self.down_proj(x)

        else: #* Split
            sparse_x, dense_x = self.token_splits(x, masks)
                
            binary_mask  = torch.zeros_like(sparse_x)
            binary_mask[:, :, sparse_indices] = 1
            sparse_x = sparse_x * binary_mask
            
            #* Let's log the sparsity:
            self.stats["sparsity/ffn"] = torch.count_nonzero(sparse_x) / sparse_x.numel()
            
            out = self.token_join(sparse=sparse_x, dense=dense_x, masks=masks)
            out = self.down_proj(out)
            
            return out
            
        
    def pred_mlp(self, x: torch.Tensor, x_intermediate: Optional[torch.Tensor] = None, sparsity= None) -> torch.Tensor:
        with torch.no_grad():
            if self.mode == "svd":
                if sparsity is not None:
                    return self.pred(x, sparsity)
                else:
                    return self.pred(x, self.sparsity)
            
            elif "oracle" in self.mode:     
                if "svd" in self.mode:
                    return self.pred(x, self.sparsity)     
                elif "wanda" in self.mode and "ffn" in self.mode:
                    scaler_row = torch.norm(x_intermediate.reshape((-1, x_intermediate.shape[-1])).t(), p=2, dim=1) ** 2
                    W_metric = torch.abs(self.down_proj.weight) * torch.sqrt(scaler_row.reshape((1, -1)))
                    score = (W_metric / W_metric.sum(dim=-1).unsqueeze(1)).sum(dim=0).squeeze()
                elif "random" in self.mode and "ffn" in self.mode:
                    score = torch.rand(x_intermediate.shape[-1], device=x_intermediate.device)
                elif "norm" in self.mode and "ffn" in self.mode:     
                    score = x_intermediate.flatten(0, 1).norm(dim=0)
                else:
                    raise ValueError("Invalid oracle mode {}".format(self.mode))
                
                return torch.topk(score, int(x_intermediate.shape[-1] * (1 - self.sparsity)), dim=-1).indices.flatten()

            else: raise ValueError("Not implemented")
        

    def _intermediate_forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        
        g = self.gate_proj(x, **kwargs)
        u = self.up_proj(x, **kwargs)
        x = silu_mul(g, u)
        
        return x
    
    def _forward_block(self, x: torch.tensor, **kwargs: Any) -> torch.Tensor:
        x = self._intermediate_forward(x, **kwargs)
        
        if kwargs.get("sparse_indices", None) is not None:
            #* We assume scattered back!
            self.stats["sparsity/ffn"] = 1-(x.shape[-1] / self.down_proj.weight.shape[-1])
        
        x = self.down_proj(x, **kwargs)
        return x
    
    def forward(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:      
        
        if self.enabled and self.sparsity > 0 and x.shape[1] > 1: #* Prefill Phase
            
            if self.mode == "svd":

                if not hasattr(self, "static_indices"):
                    x = self.kernel_forward(x, masks)

                    if self.enable_static:
                        # aggregate static channel activations based on some examples
                        one_channel_act = torch.zeros_like(self.channel_act)
                        one_channel_act[indices.to('cpu')] = 1.0
                        self.channel_act += one_channel_act
                
                else:
                    # calculate the static channel activations and combine with the dynamic channel activations (use set difference)
                    x = self.kernel_forward(x, masks, static_indices=self.static_indices)
                
                
            elif "oracle" in self.mode: 
                x_intermediate = self._intermediate_forward(x)
                with torch.no_grad():
                    indices = self.pred_mlp(x, x_intermediate)
                x = self.mask_forward(x_intermediate, masks, indices)
                
            
            else: 
                raise ValueError("Not implemented")
            
            return x
        else:
            x_out = self._forward_block(x)
        
            if self.reft and x.shape[1] > 1:
                _, _, reft_index = masks
                
                x_out = reft_forward(x_out, reft_index, self.reft_lora)

            return x_out
