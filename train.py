import torch
from torch.utils.data import Dataset
from reference_code.rwkv6_simple import RWKV_TOKENIZER
import json
from dataclasses import dataclass
import os
from src.MyRWKV_v2 import MY_RWKV_RNN
from tqdm import trange
from torch import Tensor


########## Get data ##########
class TextDataset(Dataset):
    def __init__(self, model_path:str, tokenizer_path:str) -> None:
        self.tokenizer = RWKV_TOKENIZER(tokenizer_path)
        self.data = []
        with open(model_path, "r", encoding="u8") as f:
            for line in f.readlines():
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError



########## hyper paramters ##########
class TrainingArgs:
    ckpt_dir = "./ckpts"
    data_dir = "./data/toytext.json"
    model_path = "./model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"
    tokenizer_path = "./model/rwkv_vocab_v20230424.txt"
    learning_rate = 1e-3
    epoches = 4
    accumulation_steps = 2




########## training loop ##########

# load dataset
dataset = TextDataset(TrainingArgs.data_dir)
model = MY_RWKV_RNN(TrainingArgs.model_path)
optimizer = torch.optim.Adam(model.parameters(), TrainingArgs.learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


for epoch in trange(TrainingArgs.epoches, desc="epoch"):
    accumulated_loss = 0.0
    optimizer.zero_grad()
    for step, (input_ids,attn_mask,target_ids) in enumerate(dataset, start=1):
        state = model.new_zero_state(batch_size=1)
        logits, _ = model.forward(input_ids, state)
        accumulated_loss += loss_fn(target_ids, logits) / sum(attn_mask)
        if step % TrainingArgs.accumulation_steps == 0 or step == len(dataset):
            optimizer.step()
            optimizer.zero_grad()

raise NotImplementedError