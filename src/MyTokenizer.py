from reference_code.rwkv6_simple import RWKV_TOKENIZER

class MY_TOKENIZER(RWKV_TOKENIZER):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.spEos = "</s>"
        self.spPad = "</s>"
        self.spUser = "<|user|>"
        self.spAssistant = "<|mira|"


    def apply_chat_template(self, conversation:list, need_tokenize:bool=False) -> str:
        """
        chat template looks like this:
        <|user|>
        Which is bigger, the moon or the sun?</s>
        <|mira|>
        The sun.</s>
        """
        result = ""
        for line in conversation:
            role_tag = self._get_role_tag(line['role'])
            role_content = line['content']
            result += role_tag+role_content+"</s>\n"
        if need_tokenize:
            return self.encode(result)
        else:
            return result

    @staticmethod 
    def _get_role_tag(role:str) -> str:
        if role == "user":
            return "<|user|>\n"
        if role == "assistant":
            return "<|mira|>\n"
    
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