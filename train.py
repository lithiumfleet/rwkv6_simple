import torch
from torch.utils.data import Dataset
from reference_code.rwkv6_simple import RWKV_TOKENIZER
import json


########## Get data ##########
class TextDataset(Dataset):
    def __init__(self, file_path:str) -> None:
        self.tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")
        self.data = []
        with open(file_path, "r", encoding="u8") as f:
            for line in f.readlines():
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError







########## hyper paramters ##########






########## training loop ##########







import torch
import torch.nn as nn

# Assuming vocab_size is the size of the vocabulary
vocab_size = 10  # Example size, replace with actual vocab size

# Model output probabilities (logits) for each position (batch_size, seq_length, vocab_size)
model_output = torch.tensor([
    [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prob_dist_1
    [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prob_dist_2
    [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prob_dist_3
    [0.7, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # prob_dist_4
    [0.1, 0.1, 0.1, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],  # prob_dist_5
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0],  # prob_dist_6
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # prob_dist_7
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]   # prob_dist_8
]).unsqueeze(0)  # Add batch dimension

# Target tokens (shifted input_ids)
target_ids = torch.tensor([1, 2, 3, 4, 5, 0, 0, 0]).unsqueeze(0)  # Add batch dimension

# Loss mask
loss_mask = torch.tensor([-100, -100, -100, 1, 1, -100, -100, -100]).unsqueeze(0)  # Add batch dimension

# Flatten the tensors
model_output = model_output.view(-1, vocab_size)  # (batch_size * seq_length, vocab_size)
target_ids = target_ids.view(-1)  # (batch_size * seq_length)
loss_mask = loss_mask.view(-1)  # (batch_size * seq_length)

# Apply the loss mask to the target_ids
target_ids = target_ids * (loss_mask != -100).long()

# Cross-entropy loss
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding token
loss = criterion(model_output, target_ids)

print(f"Cross-Entropy Loss: {loss.item()}")

print(f"shape of model_output:{model_output.size()}")
print(f"shape of target_ids:{target_ids.size()}")


import types
from MyRWKV import MY_RWKV_RNN
# from reference_code.rwkv6_simple import RWKV_RNN
args = types.SimpleNamespace()
args.MODEL_NAME = 'RWKV-x060-World-1B6-v2.1-20240328-ctx4096'
args.n_layer = 24
args.n_embd = 2048
args.vocab_size = 65536
my_rwkv = MY_RWKV_RNN(args)
token = 200
state = torch.zeros(args.n_layer * (2+my_rwkv.head_size), 2048)
out2, state2 = my_rwkv.forward(token, state)
print()
