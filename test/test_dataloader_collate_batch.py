import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 

from train import _collate_batch, _group_index
import torch
from torch import Tensor    
import pytest
from random import randint


def test_collate_fn():
    """
    mock max_model_len=20, batch_size = 8
    """
    def generate_one() -> tuple[Tensor]:
        a = randint(1,20)
        return (torch.randint(10,size=(a,)),torch.randint(10,size=(a,)),torch.randint(10,size=(a,)))
    dummy_batch = [generate_one() for _ in range(8)]

    print("========== original batch ===========")
    for batch in dummy_batch:
        print([i.shape[0] for i in batch])

    print("========== grouped batch ============")
    max_model_len = 20
    each_len = [sample[0].shape[0] for sample in dummy_batch]
    for index, batch in zip(_group_index(each_len, max_model_len), _collate_batch(dummy_batch, max_model_len)):
        print("index:", end="")
        print(index, end="  ")
        print("sum_len:", end="")
        print("+".join([str(dummy_batch[i][0].shape[0]) for i in index]), end="=>")
        print(batch[0].shape[0])
        assert batch[0].shape[0] <= max_model_len


if __name__ == "__main__":
    test_collate_fn()
