import torch
from torch import nn
from torch.nn import functional as F
import types




class MY_RWKV_RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode
        
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')

        for k in w.keys():
            w[k] = w[k].float() # convert to f32 type
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head
        
        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])
        print("load finished!")

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    def time_mixing(self, x, i, att, state):
        r, k, v, g, w = self._get_rkvgw(x, i, att, state)

        prev_state = state[(2+self.head_size)*i+2 : (2+self.head_size)*(i+1), :].reshape(self.n_head, self.head_size, self.head_size)
        kv = k @ v
        wkv = prev_state + att.time_faaaa * kv
        rwkv = (r @ wkv).flatten()
        normed_rwkv = F.group_norm(rwkv.unsqueeze(0), num_groups=self.n_head, weight=att.ln_x.weight, bias=att.ln_x.bias, eps = 64e-5).squeeze(0)
        output = att.output.weight @ (normed_rwkv * F.silu(g))
        # update state
        state[(2+self.head_size)*i+2 : (2+self.head_size)*(i+1), :] = (prev_state * w + kv).reshape(self.head_size, -1)
        return output

    def _get_rkvgw(self, x, i, att, state):
        x = self.layer_norm(x, self.w.blocks[i].ln1)
        sx = state[(2+self.head_size)*i+1] - x
        state[(2+self.head_size)*i+1] = x
        xxx = x + sx * att.time_maa_x
        xxx = torch.tanh(xxx @ att.time_maa_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, att.time_maa_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xw = x + sx * (att.time_maa_w + mw)
        xk = x + sx * (att.time_maa_k + mk)
        xv = x + sx * (att.time_maa_v + mv)
        xr = x + sx * (att.time_maa_r + mr)
        xg = x + sx * (att.time_maa_g + mg)
        r = (att.receptance.weight @ xr).view(self.n_head, 1, self.head_size)
        k = (att.key.weight @ xk).view(self.n_head, self.head_size, 1)
        v = (att.value.weight @ xv).view(self.n_head, 1, self.head_size)
        g =  att.gate.weight @ xg
        w = self._get_decay_factor_w(att, xw)
        return r,k,v,g,w

    def _get_decay_factor_w(self, att, xw):
        w = (att.time_decay + (torch.tanh(xw @ att.time_decay_w1) @ att.time_decay_w2).float()).view(self.n_head, self.head_size, 1)
        w = torch.exp(-torch.exp(w.float()))
        return w

    def channel_mixing(self, x, i, ffn, state):
        """same as rwkv5"""
        x = self.layer_norm(x, self.w.blocks[i].ln2)
        sx = state[(2+self.head_size)*i] - x
        xk = x + sx * ffn.time_maa_k
        xr = x + sx * ffn.time_maa_r
        state[(2+self.head_size)*i] = x
        r = torch.sigmoid(ffn.receptance.weight @ xr)
        k = torch.square(torch.relu(ffn.key.weight @ xk)) # square relu, primer paper
        return r * (ffn.value.weight @ k)

    def forward(self, token, state):
        with torch.no_grad():
            if state is None:
                state = torch.zeros(self.args.n_layer * (2+self.head_size), self.args.n_embd)
            
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(len(self.w.blocks)):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(x, i, att, state)
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(x, i, ffn, state)
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

    def __repr__(self):
        return f"""
######### RWKV x060 Intro ##########
Model args:
    num heads:{self.n_head}")
    head size:{self.head_size}")
    num layers:{len(self.w.blocks)}")

Weight has three parts:
    1. emb: denotes for embedding, a big id-to-tensor dictory
    2. ln_out, head: denotes for lm-head, the last output layer of rwkv
    3. blocks: a list contains all weights.

What's in a block?
    for example: blocks[2], the third layer.
    blocks[2] has four parts: ['ln1', 'ln2', 'att', 'ffn']
    when model forwarding through this layer:
    x ==ln1&att==> time_mixed_x ==ln2&ffn==> channel_mixed_x

What's in "att"?
    ['time_maa_x', 'time_maa_w', 'time_maa_k', 'time_maa_v', 'time_maa_r', 'time_maa_g', 'time_maa_w1', 'time_maa_w2', 'time_decay', 'time_decay_w1', 'time_decay_w2', 'time_faaaa', 'receptance', 'key', 'value', 'output', 'gate', 'ln_x']
    don't worry, I'll explain these later :)

How "att" works?
    0. Token shift
        > Mix current token with previous token.
        mixed_token = current_token + (previous_token - current_token) * time_maa_x
        wkvrg = tanh(mixed_token @ time_maa_w1). time_maa_w1 can be seen as a block matrix like stack([(2048,32,)]*5)
        reshape wkvrg from (2048,) to (5,1,32,)
        m_wkvrg = bmm(wkvrg, time_maa_w2). bmm is batch matrix multiplication.
        reshape m_wkvrg from (5,1,2048) to (5,2048). Just like squeeze oprator does.
        > So m_wrkvg is [mw, mk, mv, mr, mg]
        return m_wkvrg

    1. prepare for Mixed args.
        > Mixed args are xw,xk,xv,xr,xg. They have the same steps to get. For example I want to get xw.
        xw = current_token + (previous_token - current_token) * (time_decay_w2 + mw)
        > others are the same.

    2. prepare decay factor w.
        w = (time_decay + (tanh(xw @ time_maa_w1) @ time_maa_w2))
        reshape w to (32, 64, 1)
        w = exp(-exp(w))

    3. calculate rkvg, wkv/skv
        r,k,v,g = rw@xr,kw@xk,vw@xv,silu(gw@xg)
        > I think wkv is not a proper name... skv maybe better.
        skv = (time_faaaa * k@v + previous_state);
        xo = r @ skv
        output = ow @ group_norm(xo)
    
    4. update state
        current_state = k@v + w * previous_state
    
"""


    # def layer_norm(self, x, w):
    #     return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # # @MyFunction
    # def channel_mixing(self, x, state, i:int, time_maa_k, time_maa_r, kw, vw, rw):
    #     """same as rwkv5"""
    #     i0 = (2+self.head_size)*i+0
    #     sx = state[i0] - x
    #     xk = x + sx * time_maa_k
    #     xr = x + sx * time_maa_r
    #     state[i0] = x
    #     r = torch.sigmoid(rw @ xr)
    #     k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
    #     return r * (vw @ k)

    # # @MyFunction
    # def time_mixing(self, x, state, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
    #     H = self.n_head
    #     S = self.head_size

    #     ############### prepare for mixed args ###############
    #     i1 = (2+S)*i+1
    #     sx = state[i1] - x
    #     state[i1] = x
    #     xxx = x + sx * x_maa
    #     xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
    #     xxx = torch.bmm(xxx, tm_w2).view(5, -1)
    #     mw, mk, mv, mr, mg = xxx.unbind(dim=0)

    #     xw = x + sx * (w_maa + mw)
    #     xk = x + sx * (k_maa + mk)
    #     xv = x + sx * (v_maa + mv)
    #     xr = x + sx * (r_maa + mr)
    #     xg = x + sx * (g_maa + mg)
    #     ######################################################

    #     ###############  previous version: w = exp(-exp(decay))  ################
    #     w = (time_decay + (torch.tanh(xw @ td_w1) @ td_w2).float()).view(H, S, 1)
    #     w = torch.exp(-torch.exp(w.float()))
    #     #########################################################################

    #     r = (rw @ xr).view(H, 1, S)
    #     k = (kw @ xk).view(H, S, 1)
    #     v = (vw @ xv).view(H, 1, S)
    #     g = F.silu(gw @ xg)

    #     s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S)

    #     x = torch.zeros(H, S)
    #     a = k @ v
    #     x = r @ (time_first * a + s) # wkv = (time_first * k@v + s); x = r @ wkv
    #     ################ update status ###################
    #     s = a + w * s # s *= w; s += k@v
    #     state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1)
    #     ##################################################

    #     x = x.flatten()

    #     x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze(0) * g # same as gn(x/8, eps=1e-5)
    #     return ow @ x

    # def forward(self, token, state):

if __name__ == "__main__":
    args = types.SimpleNamespace()
    args.MODEL_NAME = 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
    args.n_layer = 24
    args.n_embd = 2048
    args.vocab_size = 65536
    my_rwkv_rnn = MY_RWKV_RNN(args)
    print(my_rwkv_rnn)
