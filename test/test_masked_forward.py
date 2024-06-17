import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 

from src.MyRWKV_v2 import MY_RWKV_RNN as MyRWKV
import torch
import pytest


def test_forward():

    input_ids = torch.as_tensor([[1,2,1,9,3,4,5,0,0,0,0,0],[1,2,1,9,3,4,5,2,3,5,4,6]])
    attn_mask = torch.as_tensor([[1,1,1,1,0,0,1,1,1,0,0,0],[1,0,0,0,0,1,1,1,1,1,1,0]])

    my_model = MyRWKV('./model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth')
    state1 = my_model.new_zero_state(2)
    output1, _state1 = my_model.forward(input_ids, state1, attn_mask)

    assert input_ids.shape == output1.shape[:2]
