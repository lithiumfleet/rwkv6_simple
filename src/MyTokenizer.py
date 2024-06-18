from reference_code.rwkv6_simple import RWKV_TOKENIZER
import torch

class MY_TOKENIZER(RWKV_TOKENIZER):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.spEos = "<|eos|>"
        self.spPad = "<|eos|>"
        self.spUser = "<|user|>"
        self.spAssistant = "<|mira|>"


    def apply_chat_template(self, conversation:list, need_tokenize:bool=False) -> str:
        """
        chat template looks like this, (no returns here, i add it for better format XD )
        <|user|>
        Which is bigger, the moon or the sun?<|eos|>
        <|mira|>
        The sun.<|eos|>
        """
        result = ""
        for line in conversation:
            role_tag = self._get_role_tag(line['role'])
            role_content = line['content']
            result += role_tag+role_content+"<|eos|>"
        if need_tokenize:
            return torch.as_tensor(self.encode(result), dtype=int).long()
        else:
            return result

    @staticmethod 
    def _get_role_tag(role:str) -> str:
        if role == "user":
            return "<|user|>"
        if role == "assistant":
            return "<|mira|>"
    
    @property
    def sp_eos(self):
        return self.spEos
    @property
    def sp_pad(self): 
        return self.spPad
    @property
    def sp_user(self): 
        return self.spUser
    @property
    def sp_assistant(self): 
        return self.spAssistant