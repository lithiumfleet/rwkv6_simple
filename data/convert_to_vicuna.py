import json
from tqdm import tqdm
from dataclasses import dataclass

file_path = "./data/atri_with_history_2_9k.json"
save_as = "./data/atri_with_history_2_9k_vicuna.json"

@dataclass
class SharegptFormat:
    instruction:str
    input:str
    output:str
    history:str

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



def convert_one_sample(fp, sample, suffix=""):
    assert fp.writable(), "Error: fp is closed."

    new_sample = []
    old_sample = SharegptFormat(**sample)

    for hist in old_sample.history:
        new_sample.append({"role":"user", "content":hist[0]})
        new_sample.append({"role":"assistant", "content":hist[1]})

    new_sample.append({"role":"user", "content":old_sample.instruction+old_sample.input})
    new_sample.append({"role":"assistant", "content":old_sample.output})

    json.dump(new_sample, fp, ensure_ascii=False, indent=4)
    fp.write(suffix)


########## load ds ##########
with open(file_path, "r", encoding="u8") as ds_fp:
    raw_dataset = json.load(ds_fp)

######### start convert ###########
with open(save_as, "w", encoding="u8") as fp:
    # adjust the format
    fp.write("[\n")

    index = 0 # so i can't find tenumerate....
    for sample in tqdm(raw_dataset, desc="converting"):
        if index != len(raw_dataset)-1:
            convert_one_sample(fp, sample, ",\n")
        else:
            convert_one_sample(fp, sample, "\n")
        index += 1

    # adjust the format
    fp.write("]")
    
print("all finished!")