# this test is for parallel forward
from MyRWKV_v2 import MY_RWKV_RNN as MyRWKV
from reference_code.model import RWKV_RNN as RWKV
import torch
import pytest
import types



def test_forward():
    args = {}
    args['MODEL_NAME'] = 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
    args['n_layer'] = 24
    args['n_embd'] = 2048
    args['vocab_size'] = 65536
    args['device'] = 'cpu'
    args['onnx_opset'] = 18

    input_ids = torch.as_tensor([[1,2,1,9,3,4,5,2,3,5,4,6]])

    my_model = MyRWKV('RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth')
    state1 = my_model.new_zero_state(1)
    output1, _state1 = my_model.forward(input_ids, state1)

    my_model = RWKV(args)
    state2 = my_model.init_state(1)
    output2, _state2 = my_model.forward_parallel(input_ids, state2)
    
    assert torch.equal(output1, output2)
    assert torch.equal(_state1, _state2)