import torch
from dataclasses import dataclass, field
from dataset import vocab
# Create a config file for the cup shuffling task
@dataclass
class MASTER_CONFIG:
    # Training
    seed: int = 1337
    batch_size: int = 32
    training_split: float = 0.8
    
    epochs = 1
    batch_eval_internal = 10
    learning_rate = 2e-5
    eval_iters = 50
    
    # Model parameters
    dtype: torch.dtype = torch.float32
    d_model: int = 128 # This is the size of the embedding
    l_max: int = 128 # Max sequence length
    n_heads: int = 8 # Number of heads in the multi-head attention
    n_layers: int = 8 # Number of layers in the transformer of MHA blocks
    dropout: float = 0.1 # Dropout rate
    causal: bool = True # Whether to use a causal mask in the attention layer
    
    # Data parameters
    n_cups: int = 3
    n_moves: int = 4
    n_samples: int = 1000

    # Tokenizer
    vocab: list[str] = field(default_factory = lambda: [])
    vocab_size: int = 0
    
    # Use CUDA or MPS if available else CPU
    if (torch.cuda.is_available()):
        device = torch.device("cuda")
        print("Using CUDA")
    elif (torch.backends.mps.is_available()):
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
setattr(MASTER_CONFIG, "vocab", vocab)
setattr(MASTER_CONFIG, "vocab_size", len(vocab))
