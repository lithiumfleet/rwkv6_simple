import json
from tqdm import tqdm

file_path = "atri_with_history_2_9k.json"
save_as = "atri_with_history_2_9k.jsonl"

with open(file_path, "r", encoding="u8") as f:
    raw_dataset = json.load(f)


# a piece of sample
# {
#     "instruction": "是因为我的高性能么…我真是厉害呢。",
#     "input": "",
#     "output": "自信满满啊。",
#     "history": [
#         [
#             "你可真够受欢迎的啊。",
#             "因为夏生先生深受大家的爱戴。"
#         ]
#     ]
# },



def convert_one_sample(fp, sample):
    assert fp.writable(), "Error: fp is closed."

    res = {"input":"", "output":""}
    res["output"] = sample["output"]


with open(save_as, "w", encoding="u8") as fp:
    for sample in tqdm(raw_dataset, desc="converting"):
        convert_one_sample(fp, sample)
    
print("all finished!")