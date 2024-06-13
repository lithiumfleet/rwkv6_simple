from torch import Tensor
from dataclasses import dataclass, field
import torch
from torch import nn
from torch.nn import LayerNorm, Embedding, GroupNorm
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
    ln_x:RWKV_Layernorm= field(default_factory=RWKV_Layernorm)
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
    ln1:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    ln2:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    att:RWKV_Att = field(default_factory=RWKV_Att)
    ffn:RWKV_Ffn = field(default_factory=RWKV_Ffn)

@dataclass
class RWKV_Weights(DictLike):
    ln0:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    blocks:list[RWKV_Block_Weights] = field(default_factory=list)
    emb_weight:Tensor = field(default_factory=Tensor)
    ln_out:RWKV_Layernorm = field(default_factory=RWKV_Layernorm)
    head_weight:Tensor = field(default_factory=Tensor)

class RWKV_Block(nn.Module):
    def __init__(self, weights:RWKV_Block_Weights) -> None:
        super().__init__()
        self._n_head = weights.att.time_faaaa.shape[0] # TODO: test this: each block is equal?
        self._head_size = weights.ln1.weight.shape[0] // self._n_head
        self._n_layer = 24
        self._n_embd = 2048
        self._vocab_size = 65536
        self.att, self.ln1, self.ffn, self.ln2 = self._load_from_weights(weights)
    
    def _load_from_weights(self, weights:RWKV_Block_Weights):
        att = weights.att
        att.time_maa_x = nn.Parameter(att.time_maa_x)
        att.time_maa_w1 = nn.Parameter(att.time_maa_w1)
        att.time_maa_w2 = nn.Parameter(att.time_maa_w2)
        att.time_decay = nn.Parameter(att.time_decay)
        att.time_decay_w1 = nn.Parameter(att.time_decay_w1)
        att.time_decay_w2 = nn.Parameter(att.time_decay_w2)
        att.time_faaaa = nn.Parameter(att.time_faaaa)

        receptance = nn.Linear(self._n_embd, self._n_embd, bias=False)
        receptance.weight = nn.Parameter(att.receptance_weight)
        att.receptance_weight = receptance

        key = nn.Linear(self._n_embd, self._n_embd, bias=False)
        key.weight = nn.Parameter(att.key_weight)
        att.key_weight = key

        value = nn.Linear(self._n_embd, self._n_embd, bias=False)
        value.weight = nn.Parameter(att.value_weight)
        att.value_weight = value

        output = nn.Linear(self._n_embd, self._n_embd, bias=False)
        output.weight = nn.Parameter(att.output_weight)
        att.output_weight = output

        gate = nn.Linear(self._n_embd, self._n_embd, bias=False)
        gate.weight = nn.Parameter(att.gate_weight)
        att.gate_weight = gate

        ln_x = GroupNorm(num_groups=self._n_head, num_channels=self._n_embd, eps=1e-5, affine=True)
        ln_x.weight = nn.Parameter(att.ln_x.weight)
        ln_x.bias = nn.Parameter(att.ln_x.bias)
        att.ln_x = ln_x

        ffn = weights.ffn
        ffn.time_maa_k = nn.Parameter(ffn.time_maa_k)
        ffn.time_maa_r = nn.Parameter(ffn.time_maa_r)

        key = nn.Linear(self._n_embd, self._n_embd, bias=False)
        key.weight = nn.Parameter(ffn.key_weight)
        ffn.key_weight = key

        receptance = nn.Linear(self._n_embd, self._n_embd, bias=False)
        receptance.weight = nn.Parameter(ffn.receptance_weight)
        ffn.receptance_weight = receptance

        value = nn.Linear(self._n_embd, self._n_embd, bias=False)
        value.weight = nn.Parameter(ffn.value_weight)
        ffn.value_weight = value

        ln1 = nn.LayerNorm(self._n_embd)
        ln1.weight = nn.Parameter(weights.ln1.weight)
        ln1.bias = nn.Parameter(weights.ln1.bias)

        ln2 = nn.LayerNorm(self._n_embd)
        ln2.weight = nn.Parameter(weights.ln2.weight)
        ln2.bias = nn.Parameter(weights.ln2.bias)
        return att, ln1, ffn, ln2

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
        xxx = x + sx_lerp * self.att.time_maa_x # xxx in [batch_size, seq_len, hidden_size]
        xxx = torch.tanh(xxx @ self.att.time_maa_w1).view(batch_size, seq_len, 5, 1, 32) # time_maa_w1.shape = [2048, 160]
        xxx = torch.matmul(xxx, self.att.time_maa_w2).view(batch_size, seq_len, 5, 2048)
        if self.att.stacked_weights is None:
            self.att.stacked_weights = (
            torch.stack(
                [
                    self.att.time_maa_k,
                    self.att.time_maa_w,
                    self.att.time_maa_v,
                    self.att.time_maa_r,
                    self.att.time_maa_g,
                ],
                dim=0,
            )
            .unsqueeze(0)
        ) # init on use. shape [1, 1, 5, hidden_size]
        # expand x/sx_lerp in dim2 (seq_len dim)
        x_kwvrg = x.unsqueeze(2) + sx_lerp.unsqueeze(2) * (self.att.stacked_weights + xxx) # x_kwvrg in shape [batch_size, 5, hidden_size]
        # get k, w, r, v, g
        k = self.att.key_weight(x_kwvrg[:, :, 0]).view(batch_size, seq_len, num_heads, head_size, 1)
        w = torch.exp(-torch.exp((self.att.time_decay + (torch.tanh(x_kwvrg[:, :, 1] @ self.att.time_decay_w1) @ self.att.time_decay_w2)).view(batch_size, seq_len, num_heads, head_size, 1)))
        r = self.att.receptance_weight(x_kwvrg[:, :, 3]).view(batch_size, seq_len, num_heads, 1, head_size)
        v = self.att.value_weight(x_kwvrg[:, :, 2]).view(batch_size, seq_len, num_heads, 1, head_size)
        g = F.silu(self.att.gate_weight(x_kwvrg[:, :, 4]), inplace=False) # [batch_size, seq_len, 2048]

        prev_state = state[:, i1+1:i1+1+head_size, :].view(batch_size, num_heads, head_size, head_size)
        kv = k @ v

        state_s = torch.zeros(batch_size, seq_len, num_heads, head_size, head_size, dtype=x.dtype, device=x.device)
        state_s[:, 0] = prev_state
        for l in range(seq_len-1):
            prev_state = kv[:, l] + w[:, l] * prev_state.clone()
            state_s[:, l+1] = prev_state
        prev_state = (kv[:, -1] + w[:, -1] * prev_state)
        # update state
        state[:, i1+1:i1+1+head_size] = prev_state.view(batch_size, head_size, -1)

        wkv = self.att.time_faaaa * kv + state_s
        rwkv = r @ wkv
        normed_rwkv = self.att.ln_x(rwkv.flatten(start_dim=2).view(batch_size*seq_len, -1)).view(batch_size, seq_len, -1)
        output = self.att.output_weight(normed_rwkv * g)

        return output

    def _channel_mixing(self, x:Tensor, state:Tensor, i:int) -> Tensor:
        """
        args:
            x (Tensor): in shape [batch_size, seq_len, hidden_size], where hidden_size equals to _n_embd.
            state (Tensor): in shape [batch_size, state_size, hidden_size]
            i (int): time index
        returns:
            Tensor: in the same shape with state. [batch_size, state_size, hidden_size]
        """
        i0 = (2 + self._head_size) * i 
        sx_lerp = torch.empty_like(x)
        sx_lerp[:, 0] = state[:, i0] - x[:, 0]
        state[:, i0] = x[:, -1]
        xk = x + sx_lerp * self.ffn.time_maa_k
        xr = x + sx_lerp * self.ffn.time_maa_r
        r = torch.sigmoid(self.ffn.receptance_weight(xr))
        k = torch.relu(self.ffn.key_weight(xk)).pow(2)
        output = r * self.ffn.value_weight(k)
        return output

    def forward(self, x:Tensor, state:Tensor, i: int) -> torch.Tensor:
        return self._time_mixing(self.ln1(x), state, i) # break here
        x = x + self._time_mixing(self.ln1(x), state, i)
        x = x + self._channel_mixing(self.ln2(x), state, i)
        return x


class MY_RWKV_RNN(nn.Module):
    def __init__(self, model_path:str = None) -> None:
        super().__init__()
        self._weights:RWKV_Weights = RWKV_Weights()
        self.blocks:list[RWKV_Block] = []
        self.embedding:Embedding = None
        self.layer_norm_0:LayerNorm = None
        self.layer_norm_out:LayerNorm = None
        self.n_head = 0 # init later
        self.head_size = 0
        self.n_layer = 0
        self.n_embd = 0
        self.vocab_size = 0
        if model_path is not None and os.path.exists(model_path):
            self.load_weights(model_path)
        
    def load_weights(self, model_path:str):
        print(f"Start Loading weights from:{model_path}")

        self._weights.blocks = [RWKV_Block_Weights() for _ in range(24)]
        w = torch.load(model_path, map_location='cpu')
        for k in w.keys():
            w[k] = w[k].float()
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

            # These not belongs to any block, temperaryly saving here
            if   k == 'emb.weight': self._weights.emb_weight = w[k]
            elif k == 'ln_out.weight': self._weights.ln_out.weight = w[k]
            elif k == 'ln_out.bias': self._weights.ln_out.bias = w[k]
            elif k == 'head.weight': self._weights.head_weight = w[k]
            elif k == 'blocks.0.ln0.weight': self._weights.ln0.weight = w[k]
            elif k == 'blocks.0.ln0.bias': self._weights.ln0.bias = w[k]
            else:
                assert k.startswith('blocks')
                ks = k.split('.') # ['block','layer_index','sub1','sub2','optional_sub3']
                if len(ks) == 4:
                    self._weights.blocks[int(ks[1])][ks[2]][ks[3]] = w[k]
                if len(ks) == 5:
                    if ks[3] == 'ln_x':
                        self._weights.blocks[int(ks[1])][ks[2]][ks[3]][ks[4]] = w[k]
                    else:
                        self._weights.blocks[int(ks[1])][ks[2]][ks[3]+'_'+ks[4]] = w[k]

        self.embedding = nn.Embedding.from_pretrained(self._weights.emb_weight, freeze=True)

        self.blocks = [RWKV_Block(block_weight) for block_weight in self._weights.blocks]

        self.layer_norm_0 = nn.LayerNorm(self.blocks[0]._n_embd)
        self.layer_norm_0.weight = nn.Parameter(self._weights.ln0.weight)
        self.layer_norm_0.bias = nn.Parameter(self._weights.ln0.bias)

        self.layer_norm_out = nn.LayerNorm(self.blocks[0]._n_embd)
        self.layer_norm_out.weight = nn.Parameter(self._weights.ln_out.weight)
        self.layer_norm_out.bias = nn.Parameter(self._weights.ln_out.bias)

        self.lm_head = nn.Linear(self.blocks[0]._n_embd, self.blocks[0]._vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self._weights.head_weight)

        self.n_head = self.blocks[0]._n_head
        self.head_size = self.blocks[0]._head_size
        self.n_layer = self.blocks[0]._n_layer
        self.n_embd = self.blocks[0]._n_embd
        self.vocab_size = self.blocks[0]._vocab_size

        print("load weights finish!")

    def forward(self, input_ids:Tensor, state:Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedding(input_ids)
        x = self.layer_norm_0(x)
        for i, block in enumerate(self.blocks):
            x = block.forward(x, state, i)
        x = self.layer_norm_out(x)
        x = self.lm_head(x)
        return x, state

    def new_zero_state(self, batch_size:int = 1):
        state = torch.zeros(batch_size, self.n_layer * (2 + self.head_size), self.n_embd)
        return state


if __name__ == "__main__":
    model = MY_RWKV_RNN()
    model.load_weights('RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth')