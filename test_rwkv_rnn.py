import unittest
import torch
import types
from unittest import TestCase
from reference_code.rwkv6_simple import RWKV_RNN
from MyRWKV import MY_RWKV_RNN


class TestRWKVRNN(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        args = types.SimpleNamespace()
        args.MODEL_NAME = 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
        args.n_layer = 24
        args.n_embd = 2048
        args.vocab_size = 65536
        self.args = args
        self.rwkv = None
        self.my_rwkv = None

    def _rwkv_get_rkvg(self, x, i, att, state):
        x = self.rwkv.layer_norm(x, self.rwkv.w.blocks[i].ln1)
        state = state
        i = i
        x_maa = att.time_maa_x
        w_maa = att.time_maa_w
        k_maa = att.time_maa_k
        v_maa = att.time_maa_v
        r_maa = att.time_maa_r
        g_maa = att.time_maa_g
        tm_w1 = att.time_maa_w1
        tm_w2 = att.time_maa_w2
        td_w1 = att.time_decay_w1
        td_w2 = att.time_decay_w2
        time_first = att.time_faaaa
        time_decay = att.time_decay
        kw = att.key.weight
        vw = att.value.weight
        rw = att.receptance.weight
        gw = att.gate.weight
        ow = att.output.weight
        ln_w = att.ln_x.weight
        ln_b = att.ln_x.bias
        H = self.rwkv.n_head
        S = self.rwkv.head_size

        ############### prepare for mixed args ###############
        i1 = (2+S)*i+1  # previous token
        sx = state[i1] - x  # sx: 2-gram
        state[i1] = x   # replace previous token with x
        xxx = x + sx * x_maa    # use time_x_maa transfer sx, then add x
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)    # non-linear function active xxx@time_maa_w1
        xxx = torch.bmm(xxx, tm_w2).view(5, -1) # batch matrix matmult
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xw = x + sx * (w_maa + mw)
        xk = x + sx * (k_maa + mk)
        xv = x + sx * (v_maa + mv)
        xr = x + sx * (r_maa + mr)
        xg = x + sx * (g_maa + mg)
        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = gw @ xg
        return r,k,v,g

    @unittest.skip
    def test_init(self):
        self.assertEqual(self.rwkv.n_head, self.my_rwkv.n_head)
        self.assertEqual(self.rwkv.head_size, self.my_rwkv.head_size)

    @unittest.skip
    def test_get_rkvg(self):
        with torch.no_grad():
            i = 3
            att = self.rwkv.w.blocks[i].att
            state1 = torch.zeros(self.rwkv.args.n_layer * (2+self.rwkv.head_size), 2048)
            state2 = torch.zeros(self.rwkv.args.n_layer * (2+self.rwkv.head_size), 2048)
            x = self.rwkv.w.emb.weight[200]
            x = self.rwkv.layer_norm(x, self.rwkv.w.blocks[0].ln0)
            r,k,v,g,_ = self.my_rwkv._get_rkvgw(x, i, att, state1)
            o_r, o_k, o_v, o_g = self._rwkv_get_rkvg(x, i, att, state2)
            self.assertTrue(torch.equal(r,o_r))
            self.assertTrue(torch.equal(k,o_k))
            self.assertTrue(torch.equal(v,o_v))
            self.assertTrue(torch.equal(g,o_g))

    @unittest.skip
    def test_time_mixing(self):
        with torch.no_grad():
            i = 5
            att = self.rwkv.w.blocks[i].att
            state1 = torch.ones(self.rwkv.args.n_layer * (2+self.rwkv.head_size), 2048)
            state2 = torch.ones(self.rwkv.args.n_layer * (2+self.rwkv.head_size), 2048)
            x = self.rwkv.w.emb.weight[203]
            x = self.rwkv.layer_norm(x, self.rwkv.w.blocks[0].ln0)
            tout1 = self.rwkv.time_mixing(self.rwkv.layer_norm(x, self.rwkv.w.blocks[i].ln1), state1, i, att.time_maa_x, att.time_maa_w, att.time_maa_k, att.time_maa_v, att.time_maa_r, att.time_maa_g, att.time_maa_w1, att.time_maa_w2, att.time_decay_w1, att.time_decay_w2, att.time_faaaa, att.time_decay, att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight, att.ln_x.weight, att.ln_x.bias)
            self.rwkv = None
            self.my_rwkv = MY_RWKV_RNN(self.args)
            tout2 = self.my_rwkv.time_mixing(x, i,att, state2)
            self.assertTrue(torch.equal(tout1, tout2))


    def test_forward(self):
        with torch.no_grad():
            self.rwkv = RWKV_RNN(self.args)
            state1 = torch.ones(self.rwkv.args.n_layer * (2+self.rwkv.head_size), 2048)
            state2 = torch.ones(self.rwkv.args.n_layer * (2+self.rwkv.head_size), 2048)
            token = 123

            out1, state1 = self.rwkv.forward(token, state1)
            self.rwkv = None
            self.my_rwkv = MY_RWKV_RNN(self.args)
            out2, state2 = self.my_rwkv.forward(token, state2)
            self.my_rwkv = None

            self.assertTrue(torch.equal(out1, out2))
            self.assertTrue(torch.equal(state1, state2))




if __name__ == "__main__":
    unittest.main()
