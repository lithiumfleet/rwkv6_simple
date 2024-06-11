from torch import Tensor
from dataclasses import dataclass, field
import torch
from torch import nn
import os
from typing import Optional
from torch.nn import functional as F


class DictLike:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

@dataclass
class RWKV_Layernorm(DictLike):
    weight:Tensor = field(default_factory=Tensor)
    bias:Tensor = field(default_factory=Tensor)

@dataclass
class RWKV_Att(DictLike):
    time_maa_x:Tensor = field(default_factory=Tensor)
    time_maa_w:Tensor = field(default_factory=Tensor)
    time_maa_k:Tensor = field(default_factory=Tensor)
    time_maa_v:Tensor = field(default_factory=Tensor)
    time_maa_r:Tensor = field(default_factory=Tensor)
    time_maa_g:Tensor = field(default_factory=Tensor)
    time_maa_w1:Tensor = field(default_factory=Tensor)
    time_maa_w2:Tensor = field(default_factory=Tensor)
    time_decay:Tensor = field(default_factory=Tensor)
    time_decay_w1:Tensor = field(default_factory=Tensor)
    time_decay_w2:Tensor = field(default_factory=Tensor)
    time_faaaa:Tensor = field(default_factory=Tensor)
    receptance_weight:Tensor = field(default_factory=Tensor)
    key_weight:Tensor = field(default_factory=Tensor)
    value_weight:Tensor = field(default_factory=Tensor)
    output_weight:Tensor = field(default_factory=Tensor)
    gate_weight:Tensor = field(default_factory=Tensor)
    ln_x:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    stacked_weights:Optional[Tensor] = field(default=None)

@dataclass
class RWKV_Ffn(DictLike):
    time_maa_k:Tensor = field(default_factory=Tensor)
    time_maa_r:Tensor = field(default_factory=Tensor)
    key_weight:Tensor = field(default_factory=Tensor)
    receptance_weight:Tensor = field(default_factory=Tensor)
    value_weight:Tensor = field(default_factory=Tensor)

@dataclass
class RWKV_Block_Weights(DictLike):
    ln0:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    ln1:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    ln2:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    att:RWKV_Att = field(default_factory=RWKV_Att)
    ffn:RWKV_Ffn = field(default_factory=RWKV_Ffn)

@dataclass
class RWKV_Weights(DictLike):
    blocks:list[RWKV_Block_Weights] = field(default_factory=list)
    emb_weight:Tensor = field(default_factory=Tensor)
    ln_out:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    head_weight:Tensor = field(default_factory=Tensor)

class RWKV_Block(nn.Module):
    def __init__(self, weights:RWKV_Block_Weights) -> None:
        super().__init__()
        self.weights:RWKV_Block_Weights = weights
        self._n_layer = 24
        self._n_embd = 2048
        self._n_head = self.weights.att.time_faaaa.shape[0] # TODO: test this: each block is equal?
        self._head_size = self.weights.ln1.weight.shape[0] // self.n_head
        self._vocab_size = 65536

    def _layer_norm(self, x:Tensor, weight:Tensor, bias:Tensor, eps:float=1e-5) -> Tensor:
        return F.layer_norm(x, (self._n_embd,), weight=weight, bias=bias, eps=eps)

    def _time_mixing(self, x:Tensor, state:Tensor, i:int) -> Tensor:
        """
        args:
            x (Tensor): in shape [batch_size, seq_len, hidden_size], where hidden_size equals to _n_embd.
            state (Tensor): in shape [batch_size, state_size, hidden_size]
            i (int): time index
        returns:
            Tensor: in the same shape with state. [batch_size, state_size, hidden_size]
        """
        batch_size, seq_len, num_heads, head_size = x.shape[0], x.shape[1], self._n_head, self._head_size
        i1 = (2 + head_size) * i + 1
        sx_lerp = torch.empty_like(x) # sx_lerp.shape = [batch_size, seq_len, hidden_size(=2048)]
        sx_lerp[:, 0] = state[:, i1] - x[:, 0] # sx = state[i1] - x
        sx_lerp[:, 1:] = x[:, :-1] - x[:, 1:] # token shift for sx[:, 1:seq_len]
        state[:, i1] = x[:, -1] # record the last token in each batch: x[: -1]
        xxx = x + sx_lerp * self.weights.att.time_maa_x # xxx in [batch_size, seq_len, hidden_size]
        xxx = torch.tanh(xxx @ self.weights.att.time_maa_w1).view(batch_size, seq_len, 5, 1, 32) # time_maa_w1.shape = [2048, 160]
        xxx = torch.matmul(xxx, self.weights.att.time_maa_w2).view(batch_size, seq_len, 5, 2048)
        if self.weights.att.stacked_weights is None:
            self.weights.att.stacked_weights = (
            torch.stack(
                [
                    self.weights.att.time_maa_k,
                    self.weights.att.time_maa_w,
                    self.weights.att.time_maa_v,
                    self.weights.att.time_maa_r,
                    self.weights.att.time_maa_g,
                ],
                dim=0,
            )
            .unsqueeze(0)
        ) # init on use. shape [1, 1, 5, hidden_size]
        # expand x/sx_lerp in dim2 (seq_len dim)
        x_kwvrg = x.unsqueeze(2) + sx_lerp.unsqueeze(2) * (self.weights.att.stacked_weights + xxx) # x_kwvrg in shape [batch_size, 5, hidden_size]
        raise NotImplementedError


    def _channel_mixing(self, x:Tensor, state:Tensor, i:int) -> Tensor:
        raise NotImplementedError
    
    def forward(self, x:Tensor, state:Tensor, i: int) -> torch.Tensor:
        raise NotImplementedError




class MY_RWKV_RNN(nn.Module):
    def __init__(self, model_path:str = None) -> None:
        super().__init__()
        if model_path is not None and os.path.exists(model_path):
            self.load_weights(model_path)
        
    def load_weights(self, model_path:str):
        print(f"Start Loading weights from:{model_path}")

        self.weights = RWKV_Weights()
        self.weights.blocks = [RWKV_Block_Weights() for _ in range(24)]
        w = torch.load(model_path, map_location='cpu')
        for k in w.keys():
            w[k] = w[k].float()
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

            if   k == 'emb.weight': self.weights.emb_weight = w[k]
            elif k == 'ln_out.weight': self.weights.ln_out.weight = w[k]
            elif k == 'ln_out.bias': self.weights.ln_out.bias = w[k]
            elif k == 'head.weight': self.weights.head_weight = w[k]
            else:
                assert k.startswith('blocks')
                if k == 'blocks.0.ln0.weight': self.weights.blocks[0].ln0.weight = w[k]
                elif k == 'blocks.0.ln0.bias': self.weights.blocks[0].ln0.bias = w[k]
                else:
                    ks = k.split('.') # ['block','layer_index','sub1','sub2','optional_sub3']
                    if len(ks) == 4:
                        self.weights.blocks[int(ks[1])][ks[2]][ks[3]] = w[k]
                    if len(ks) == 5:
                        self.weights.blocks[int(ks[1])][ks[2]][ks[3]+'_'+ks[4]] = w[k]
        print("load weights finish!")

    def forward(self, input_ids:Tensor, state:Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError



if __name__ == "__main__":
    model = MY_RWKV_RNN()
    model.load_weights('RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth')