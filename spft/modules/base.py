from typing import List
import torch.nn as nn
import torch
import random
from typing import Optional
from .svd import create_mlp_svd_pred, create_attn_svd_pred
from transformers import AutoTokenizer
__all__ = ["SparseModule", "reft_forward", "indice_gen", "get_punc_index", "compose_reft_index"]


PUNC = set([".", ",", "?", "!", ";", ":"])
# PUNC = [".", ",", "?", "!", ";", ":", "\n", "\"", "'", "(", ")", "[", "]", "{", "}"]


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
            masks, id_split, _ = masks
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


def get_punc_index(tokenizer: AutoTokenizer, input_ids: torch.Tensor, EOS: Optional[int] = None):
    """Get the indices of punctuation tokens in the input text."""
    # raw_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu()[0])
    punc_inds = []
    for i, input_id in enumerate(input_ids.cpu()):
        tokens = tokenizer.convert_ids_to_tokens(input_id)
        punc_ind = []
        for j in range(min(len(tokens), EOS[i]) if EOS is not None else len(tokens)):
            if tokens[j].replace('Ġ', '') in PUNC:
                punc_ind.append(j)
        punc_inds.append(punc_ind)
    return punc_inds


def compose_reft_index(bos: torch.Tensor, eos:torch.Tensor, prefix_num: int, suffix_num: int, punc_ids: Optional[list[list]] = None) -> torch.Tensor:
    """Compose the reft index from the beginning and end indices."""
    
    device = bos.device

    if punc_ids is None:
        reft_index = torch.cat([indice_gen(bos, prefix_num, True),
                                indice_gen(eos, suffix_num, False)], dim=-1)
    
    # Minimum suffix and prefix number is 2, the maximum depend on the punc ids

    else:
        reft_index = []
        all_num = prefix_num + suffix_num
        bos, eos = bos.tolist(), eos.tolist()
        for i in range(len(bos)):
            bos_i, eos_i, punc_id = bos[i], eos[i], punc_ids[i]
            reft_index_i = []
            if len(punc_id) > all_num - 4:
                random_punc = random.sample(punc_id, all_num - 4)
            else:
                random_punc = punc_id

            bos_index, eos_index = [], []
            while True:
                if bos_i not in random_punc:
                    bos_index.append(bos_i)
                bos_i += 1
                if len(bos_index) == (all_num - len(random_punc) + 1) // 2:
                    break
            
            while True:
                if eos_i not in random_punc:
                    eos_index.append(eos_i)
                eos_i -= 1
                if len(eos_index) == all_num - len(random_punc) - len(bos_index):
                    break

            reft_index_i = bos_index + random_punc + eos_index
            reft_index.append(reft_index_i)

    return torch.tensor(reft_index).to(device)


def indice_gen(begin_s: torch.Tensor, prefix: int, direction: bool = True) -> torch.Tensor:
    # begin_s include the first bos token position in each sequence
    offset = torch.arange(prefix, device=begin_s.device)

    if not direction:
        offset = offset - prefix
    
    return begin_s.unsqueeze(1) + offset

def reft_forward(x: torch.Tensor, index: torch.Tensor, reft_model: nn.Sequential) -> torch.Tensor:
    """
    Reft forward pass.
    """

    index_s_indice = index.unsqueeze(-1).expand(-1, -1, x.shape[-1])

    index_input = torch.gather(x, 1, index_s_indice)

    reft_out = reft_model(index_input)

    x = torch.scatter_add(x, 1, index_s_indice, reft_out)

    return x
