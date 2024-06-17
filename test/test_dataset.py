import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 

from src import MY_TOKENIZER, MY_RWKV_RNN
from train import sftDataset    
import pytest

@pytest.mark.skip("print needed")
def test_dataset_getitem(idx):
    ds = sftDataset("./data/atri_with_history_2_9k_vicuna.json", "./model/rwkv_vocab_v20230424.txt")
    return ds[idx]

if __name__ == "__main__":
    tokenizer = MY_TOKENIZER("./model/rwkv_vocab_v20230424.txt")
    i,j,k = test_dataset_getitem(9)
    assert i.shape == j.shape == k.shape
    print("######### input_ids #########")
    print(repr(tokenizer.decode(i.tolist())))
    print("######### attn_mask #########")
    print(j.tolist())
    print("######## target_ids #########")
    print(repr(tokenizer.decode(k.tolist())))