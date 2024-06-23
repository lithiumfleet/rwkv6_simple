import torch
from torch.utils.data import Dataset, DataLoader
from src.MyTokenizer import MY_TOKENIZER as RWKV_TOKENIZER
import json
from dataclasses import dataclass
import os
from src.MyRWKV_v2 import MY_RWKV_RNN
from tqdm import trange, tqdm
from torch import Tensor
from functools import partial


########## Get data ##########
class sftDataset(Dataset):
    def __init__(self, ds_file_path:str, tokenizer_path:str) -> None:
        self.tokenizer = RWKV_TOKENIZER(tokenizer_path)
        with open(ds_file_path, "r", encoding="u8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def _get_attn_mask(self, input_ids:Tensor) -> Tensor:
        role_flag = 1 # 1 denotes for user and 0 denotes for assistant
        attn_mask = []
        for token in input_ids:
            if token == self.tokenizer.encode(self.tokenizer.sp_user)[0]:
                role_flag = 1
                attn_mask.append(role_flag)
            elif token == self.tokenizer.encode(self.tokenizer.sp_assistant)[0]:
                attn_mask.append(role_flag)
                role_flag = 0
            elif role_flag == 0 and token == self.tokenizer.encode(self.tokenizer.sp_eos)[0]:
                attn_mask.append(role_flag)
                role_flag = 1
            else:
                attn_mask.append(role_flag)
        return torch.as_tensor(attn_mask).long()

    def _apply_chat_template(self, sample:list, need_tokenize=False) -> Tensor:
        return self.tokenizer.apply_chat_template(sample, need_tokenize)
    
    def _get_target_ids(self, input_ids:Tensor, attn_mask:Tensor) -> Tensor:
        target_ids = input_ids.clone().detach()
        for index, mask in enumerate(attn_mask):
            if mask == 1:
                target_ids[index] = -100 # the default IGNORE_INDEX for CrossentropyLoss
        return target_ids.long()
        

    def __getitem__(self, index) -> tuple[Tensor]:
        input_ids = self._apply_chat_template(self.data[index], need_tokenize=True)
        attn_mask = self._get_attn_mask(input_ids)
        target_ids = self._get_target_ids(input_ids, attn_mask)
        assert input_ids.shape[0]!=0 and attn_mask.shape[0]!=0 and target_ids.shape[0]!=0, "input_ids, attn_mask, target_ids can't be zero length."
        return (input_ids, attn_mask, target_ids)



########## hyper paramters ##########
class TrainingArgs:
    ckpt_dir = "./ckpts"
    data_dir = "./data/atri_with_history_2_9k_vicuna.json"
    model_path = "./model/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"
    tokenizer_path = "./model/rwkv_vocab_v20230424.txt"
    learning_rate = 1e-3
    epoches = 4
    accumulation_steps = 2
    max_model_len = 2048
    sp_pad_tokenid = 0 # FIXME: currently sp_pad_tokenid need to pass to TraningArgs manually, otherwise cannot init the dataloader.

########## tool func ###########
def is_zero_grad(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                if not torch.all(param.grad == 0):
                    return False
    return True

def _group_index(each_len:list[int], max_model_len:int) -> list[list[int]]:
    """
    return grouped index of each sample in raw_batch.
    ensure index of samples in the same list can concat and (nearly) reach to max_model_len.
    """
    result = []
    added_index = set()
    while len(added_index) < len(each_len):
        current_res = []
        current_len = 0
        for index, length in enumerate(each_len):
            if index not in added_index and max_model_len-current_len >= length:
                current_res.append(index)
                added_index.add(index)
                current_len += length
        result.append(current_res)
    return result

def _concat_and_pad(sample:list[Tensor], max_model_len:int, pad_token:int) -> Tensor:
    concated_sample = torch.concat(sample)
    assert concated_sample.shape[0] <= max_model_len
    pad_seq = torch.as_tensor([pad_token for _ in range(max_model_len-concated_sample.shape[0])])
    padded_sample = torch.concat([concated_sample, pad_seq])
    assert padded_sample.shape[0] <= max_model_len
    return padded_sample

def _collate_batch(raw_batch:list[tuple[Tensor]], max_model_len:int) -> list[tuple[Tensor]]:
    each_len = [sample[0].shape[0] for sample in raw_batch]
    grouped_index = _group_index(each_len, max_model_len)

    batch = []
    for indexs in grouped_index:
        input_ids = _concat_and_pad([raw_batch[i][0] for i in indexs], max_model_len, TrainingArgs.sp_pad_tokenid)
        attn_mask = _concat_and_pad([raw_batch[i][1] for i in indexs], max_model_len, TrainingArgs.sp_pad_tokenid)
        target_ids= _concat_and_pad([raw_batch[i][2] for i in indexs], max_model_len, TrainingArgs.sp_pad_tokenid)
        batch.append(tuple([input_ids, attn_mask, target_ids]))
    return batch


########## training loop ##########
if __name__ == "__main__":

    # load dataset
    dataloader = DataLoader(
        sftDataset(TrainingArgs.data_dir, TrainingArgs.tokenizer_path), 
        batch_size=32,
        shuffle=True, 
        num_workers=1, 
        collate_fn=partial(_collate_batch, max_model_len=TrainingArgs.max_model_len)
    )

    model = MY_RWKV_RNN(TrainingArgs.model_path)
    optimizer = torch.optim.Adam(model.parameters(), TrainingArgs.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()


    for epoch in range(TrainingArgs.epoches):
        print(f"############ epoch {epoch} ############")
        # assert is_zero_grad(optimizer), "Between two epoches the optimizer is not set to zero state."
        for step, (input_ids,attn_mask,target_ids) in enumerate(tqdm(dataloader), start=1):
            state = model.new_zero_state(batch_size=1)
            logits, _ = model.forward(input_ids, state, attn_mask)
            loss = loss_fn(target_ids, logits) / attn_mask.sum()
            loss.backward()
            if step % TrainingArgs.accumulation_steps == 0 or step == len(dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()

    raise NotImplementedError("need to save model.")