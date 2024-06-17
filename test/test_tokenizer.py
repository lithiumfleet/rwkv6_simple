import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0,parentdir) 
from src.MyTokenizer import MY_TOKENIZER
import pytest

@pytest.skip("you can see the conslone to check.")
def test_template():
    tokenizer = MY_TOKENIZER("D:\\rwkv_simple\\model\\rwkv_vocab_v20230424.txt")
    conversation = [
        {"role":"user", "content":"hello!"},
        {"role":"assistant", "content":"hi"},
        {"role":"user", "content":"what are you \ndoing?"},
        {"role":"assistant", "content":"sleeping."}
    ]
    res = tokenizer.apply_chat_template(conversation)
    print(res)
