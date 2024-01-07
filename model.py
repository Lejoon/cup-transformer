import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class positional_encoding(nn.Module):
    """ 
    Positional encoding according to [VSP17] paper "Attention is all you need" based on sine and cosine functions.
    
    B = batch size
    T = sequence length
    d_model = embedding dimension
    
    Input: x a sequence of tokens of shape (B, T, d_model)
    Output: p, where p is the positional encoding, of shape (B, T, d_model)
    """
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        l_max = config.l_max
        dtype = config.dtype
        
        self.p = torch.zeros((1, l_max, d_model)) #TODO: Is this really necessary? Should this be l_max, d_model instead?
        num = torch.arange(l_max, dtype=dtype).reshape(-1, 1) # Creates X = [[0], [1], ..., [l_max - 1]]
        denum = torch.pow(10000, torch.arange(0, d_model, 2, dtype=dtype) / d_model) # Creates Y = [10000^0/d_model, 10000^2/d_model, ..., 10000^(d_model - 1)/d_model]
        self.p[:, :, 0::2] = torch.sin(num / denum)
        self.p[:, :, 1::2] = torch.cos(num / denum)
        
    def forward(self, x):
        return self.p[:, :x.shape[1], :].to(x.device)

class AttentionHead(nn.Module):
    def __init__(self, d_q, d_v, d_attn, d_out, bias=False, mask=None):
        super().__init__()

        # Linear layers for Q, K, V of dimensions dq, dv, dv respectively
        # TODO: More efficient to do the linear transformation together and then split the result?
        # TODO: Allow a toggle for whether to use bias or not
        self.linear_q = nn.Linear(d_attn, d_q, bias=bias)
        self.linear_k = nn.Linear(d_attn, d_v, bias=bias)
        self.linear_v = nn.Linear(d_out, d_v, bias=bias)
        
    def forward(self, x, mask=None):
        _, _, D = x.shape # x is of shape (B, T, d_model) TODO: Maybe should make this dynamic through reshaping -1?
        
        q = self.linear_q(x) # (B, T, d_model)
        k = self.linear_k(x) # (B, T, d_model)
        v = self.linear_v(x) # (B, T, d_model)
        
        S = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(D) # Calculate the attention weights (B, T, d_model) * (B, d_model, T) = (B, T, T)
        
        if mask == None:
            weights = S
        else:
            weights = S.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(weights, dim=-1) # Apply the softmax on the last dimension, meaning the last dimension sums to 1
        v_bar = torch.bmm(weights, v) # Apply the attention weights to the values (B, T, T) * (B, T, d_model) = (B, T, d_model)
        # Y_t = att(X_t W_h^Q, X_t W_h^K, X_t W_h^V) = softmax((X_t W_h^Q)(X_t W_h^K)^t / sqrt(d_model)) (X_t W_h^V)

        return v_bar
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask=None):
        super().__init__()
        self.n_heads = n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        head_size = d_model // n_heads
        
        self.heads = nn.ModuleList(
            [AttentionHead(d_attn = head_size, d_out = head_size, d_q=d_model, d_v=d_model, mask=mask) for _ in range(n_heads)]
        )
        
        self.linear_o = nn.Linear(d_model * n_heads, d_model)
        
    def forward(self, x):
        H = self.n_heads

        B, T, _ = x.shape # x is of shape (B, T, d_model)
        
        x = x.view(B, T, H, -1) # Reshape x to (B, T, n_heads, d_model/n_heads)
        x = x.transpose(1, 2) # Transpose to get shape (B, n_heads, T, d_model/n_heads)
        
        v = [head(x[:, i, :, :]) for i, head in enumerate(self.heads)] # Apply attention heads to shape (B, 1, T, d_model/n_heads)

        v = torch.stack(v, dim=1) # Stack heads to get shape (B, n_heads, T, d_model/n_heads)
        v = v.transpose(1, 2).contiguous().view(B, T, -1) # Reshape to (B, T, d_model)
        #TODO: Understand what this transformation does
        
        v_bar = self.linear_o(v) # Apply the linear layer (B, T, d_model) -> (B, T, d_model)
        
        return v_bar
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, mask=None):
        super().__init__()

        self.ln_mha = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, mask=mask)
        
        self.fcn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        # x is of shape (B, T, d_model)
        
        x = self.ln_mha(x) # (B, T, d_model) -> (B, T, d_model)
        x = self.mha(x) + x # (B, T, d_model) -> (B, T, d_model)
        x = self.fcn(x) + x # (B, T, d_model) -> (B, T, d_model)
        
        return x

class cup_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size  # Define vocab_size here
        self.n_layers = config.n_layers
        self.l_max = config.l_max
        self.causal = config.causal
        
        d_model = config.d_model
        n_heads = config.n_heads
        d_model = config.d_model
        vocab_size = config.vocab_size
        n_layers = config.n_layers
        l_max = config.l_max
        causal = config.causal
        
        if causal: #Causal mask
            mask = torch.triu(torch.ones((l_max, l_max)), diagonal=1)
        else:
            mask = None

        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = positional_encoding(config)
        
        self.mha_layers = nn.ModuleList(
            [MultiHeadAttentionLayer(d_model=d_model, n_heads=n_heads, mask=mask) for _ in range(n_layers)]
        )
        
        self.ln = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, targets=None):
        B, T = x.shape # x is of shape (B, T)

        x = self.embed(x) # Embed the tokens (B, T) -> (B, T, d_model)

        x = x + self.pos_enc(x) # Add the positional encoding (B, T, d_model) -> (B, T, d_model)

        for mha_layer in self.mha_layers: 
            x = mha_layer(x) # Apply the MHA layers (B, T, d_model) -> (B, T, d_model)
        
        x = self.ln(x) # Apply the layer norm (B, T, d_model) -> (B, T, d_model)
        unnorm_logits = self.unembed(x) # Apply the unembedding layer (B, T, d_model) -> (B, T, vocab_size), unnormalized logits
        
        if targets is not None:
            print("Need to implement this")           
        else:
            loss = None
            
        return unnorm_logits, loss
