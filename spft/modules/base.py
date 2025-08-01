from typing import List
import torch.nn as nn
import torch
from typing import Optional
from .svd import create_mlp_svd_pred, create_attn_svd_pred
__all__ = ["SparseModule", "reft_forward", "indice_gen"]


class SparseModule(nn.Module):
    enabled: bool = True
    inherited_attributes: List[str] = []

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.stats = {}
        for name in self.inherited_attributes:
            setattr(self, name, getattr(base, name))

    def load_predictor(self, base, cfg):
        
        if "svd" in cfg.mode:
            self.mode = "svd"
            rank = int(cfg.mode.split("_")[-1])
            if "mlp" in self.layer_name:
                self.pred = create_mlp_svd_pred(base, rank, self.layer_name, cfg)
            elif "attn" in self.layer_name:
                self.pred = create_attn_svd_pred(base, rank, self.layer_name, cfg)
            else:
                raise ValueError("Not implemented")
        
        #* Can be SVD + Oracle == Masked SVD    
        if "oracle" in cfg.mode:
            self.mode = cfg.mode
        
        return None
    
    def split_idx(self, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(masks, tuple):
            return masks[1]
        else:
            raise ValueError("Should use slice not mask")
        
    def token_splits(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        #?@ Fix here!
        if isinstance(masks, tuple):
            masks, id_split, _, _ = masks
            sparse_x = x[:, :id_split, :]
            dense_x = x[:, id_split:, :]
        else:
            dense_x = x[masks].view(x.shape[0], -1, x.shape[-1])
            sparse_x = x[~masks].view(x.shape[0], -1, x.shape[-1])
        
        self.stats["token_split/sparse"] = sparse_x.shape[1]
        self.stats["token_split/dense"] = dense_x.shape[1]
        
        return sparse_x.contiguous(), dense_x.contiguous()
    
    def token_join(self, sparse, dense, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        if masks is not None and not isinstance(masks, tuple):
            res = torch.zeros((sparse.shape[0], sparse.shape[1] + dense.shape[1], sparse.shape[-1]), device=sparse.device, dtype=sparse.dtype)
            res[masks] = dense.view(-1, dense.shape[-1])
            res[~masks] = sparse.view(-1, sparse.shape[-1])
        else:
            #* Token Order: [Sparse | Dense] --> [In | Out]
            res = torch.cat([sparse, dense], dim=1)
        return res.contiguous()


def indice_gen(begin_s: torch.Tensor, prefix: int, direction: bool = True) -> torch.Tensor:
    # begin_s include the first bos token position in each sequence
    offset = torch.arange(prefix, device=begin_s.device)

    if not direction:
        offset = offset - prefix
    
    return begin_s.unsqueeze(1) + offset

def reft_forward(x: torch.Tensor, begin_s: torch.Tensor, end_s: torch.Tensor, reft_model: nn.Sequential) -> torch.Tensor:
    """
    Reft forward pass.
    """

    begin_s_indice = begin_s.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    
    end_s_indice = end_s.unsqueeze(-1).expand(-1, -1, x.shape[-1])

    prefix_input = torch.gather(x, 1, begin_s_indice)
    suffix_output = torch.gather(x, 1, end_s_indice)

    x = torch.scatter_add(x, 1, begin_s_indice, reft_model(prefix_input))
    x = torch.scatter_add(x, 1, end_s_indice, reft_model(suffix_output))
    # x = x.scatter_add_(1, begin_s_indice, reft_model(prefix_input))
    # x = x.scatter_add_(1, end_s_indice, reft_model(suffix_output))

    return x
