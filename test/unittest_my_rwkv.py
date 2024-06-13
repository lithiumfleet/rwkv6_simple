import unittest
from unittest import TestCase, skip
from src.MyRWKV import MY_RWKV_RNN
import types
from torch import equal

class TestMyRWKV(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        args = types.SimpleNamespace()
        args.MODEL_NAME = 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
        args.n_layer = 24
        args.n_embd = 2048
        args.vocab_size = 65536
        self.args = args

    def test_init(self):
        model = MY_RWKV_RNN(self.args)
        samples1 = [model.w.emb.weight[211], model.w.blocks[15].ffn.time_maa_r, model.w.blocks[4].att.time_decay_w1, model.w.blocks[7].att.gate.weight]
        model.w = None
        model.load_weights('RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth')
        samples2 = [model.weights.emb_weight[211], model.weights.blocks[15].ffn.time_maa_r, model.weights.blocks[4].att.time_decay_w1, model.weights.blocks[7].att.gate_weight]
        for s1, s2 in zip(samples1, samples2):
            self.assertTrue(equal(s1, s2))

if __name__ == "__main__":
    unittest.main()