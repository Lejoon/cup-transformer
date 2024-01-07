# Class for tokenizing strings into a list of integers and vice versa
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.idx_to_s = {i:ch for i, ch in enumerate(vocab)}
        self.s_to_idx = {ch:i for i, ch in enumerate(vocab)}
        
    def encode_tokens(self, s: str) -> list[int]:
        ids = []
        i = 0
        while i < len(s):
            max_len = -1
            max_token = None
            for token in self.s_to_idx.keys():
                token_len = len(token)
                if s[i:i+token_len] == token:
                    if token_len > max_len:
                        max_len = token_len
                        max_token = token
            if max_token:
                ids.append(self.s_to_idx[max_token])
                i += max_len
            else:
                print(f"Unrecognized sequence at index {i}, {s[i:i+1]}")
                
                break
        
        return ids
    
    def decode_tokens(self, ids: list[int]) -> str:
        return "".join([self.idx_to_s[i] for i in ids])
    
    def __len__(self):
        return len(self.vocab)
    
    def __repr__(self):
        return f"Tokenizer({self.vocab})"
    
    def __str__(self):
        return f"Tokenizer({self.vocab})"
    
    def __getitem__(self, key):
        return self.vocab[key]
    
    def __contains__(self, item):
        return item in self.vocab