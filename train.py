import torch
from torch.utils.data import Dataset
from src.MyTokenizer import MY_TOKENIZER as RWKV_TOKENIZER
import json
from dataclasses import dataclass
import os
from src.MyRWKV_v2 import MY_RWKV_RNN
from tqdm import trange
from torch import Tensor


########## Get data ##########
class sftDataset(Dataset):
    def __init__(self, model_path:str, tokenizer_path:str) -> None:
        self.tokenizer = RWKV_TOKENIZER(tokenizer_path)
        with open(model_path, "r", encoding="u8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def _get_attn_mask(self, input_ids:Tensor) -> Tensor:
        role_flag = 1 # 1 denotes for user and 0 denotes for assistant
        attn_mask = []
        for token in input_ids:
            if token == self.tokenizer.encode("<put user id here>"):
                role_flag = 1
                attn_mask.append(role_flag)
            elif token == self.tokenizer.encode("<put assistant id here>"):
                attn_mask.append(role_flag)
                role_flag = 0
            elif role_flag == 0 and token == self.tokenizer.encode("<put eos id here>"):
                attn_mask.append(role_flag)
                role_flag = 1
            else:
                attn_mask.append(role_flag)
        raise NotImplementedError
        return torch.as_tensor(attn_mask)

    def _apply_chat_template(self, sample:list) -> Tensor:
        return self.tokenizer.apply_chat_template(sample, need_tokenize=True)
    
    def _get_target_ids(self, input_ids:Tensor, attn_mask:Tensor):
        target_ids = input_ids.clone().detach()
        for index, mask in enumerate(attn_mask):
            if mask == 1:
                target_ids[index] = -100 # the default IGNORE_INDEX for CrossentropyLoss
        raise NotImplementedError # TODO: not test yet
        return target_ids
        

    def __getitem__(self, index) -> tuple[Tensor]:
        # FIXME: tokenizer need chat_templete method
        input_ids = self._apply_chat_template(self.data[index])
        attn_mask = self._get_attn_mask(self.data[index])
        target_ids = self._get_target_ids(input_ids, attn_mask)
        raise NotImplementedError # TODO: not test yet
        return (input_ids, attn_mask, target_ids)



########## hyper paramters ##########
class TrainingArgs:
    ckpt_dir = "./ckpts"
    data_dir = "./data/atri_with_history_2_9k_vicuna.json.json"
    model_path = "./model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"
    tokenizer_path = "./model/rwkv_vocab_v20230424.txt"
    learning_rate = 1e-3
    epoches = 4
    accumulation_steps = 2


########## tools ###########
def is_zero_grad(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                if not torch.all(param.grad == 0):
                    return False
    return True


########## training loop ##########

# load dataset
dataset = sftDataset(TrainingArgs.data_dir)
model = MY_RWKV_RNN(TrainingArgs.model_path)
optimizer = torch.optim.Adam(model.parameters(), TrainingArgs.learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


for epoch in trange(TrainingArgs.epoches, desc="epoch"):
    assert is_zero_grad(optimizer), "Between two epoches the optimizer is not set to zero state."
    for step, (input_ids,attn_mask,target_ids) in enumerate(dataset, start=1):
        state = model.new_zero_state(batch_size=1)
        logits, _ = model.forward(input_ids, state)  # FIXME: forward with attn mask.
        loss = loss_fn(target_ids, logits) / attn_mask.sum()
        loss.backward()
        if step % TrainingArgs.accumulation_steps == 0 or step == len(dataset) - 1:
            optimizer.step()
            optimizer.zero_grad()

raise NotImplementedError