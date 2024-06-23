import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 

from train import _collate_batch, _group_index, sftDataset
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import pytest

class TrainingArgs:
    ckpt_dir = "./ckpts"
    data_dir = "./data/atri_with_history_2_9k_vicuna.json"
    model_path = "./model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"
    tokenizer_path = "./model/rwkv_vocab_v20230424.txt"
    learning_rate = 1e-3
    epoches = 4
    accumulation_steps = 2
    max_model_len = 2048
