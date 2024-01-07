import random
from tokenizer import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# Dataset for the cup shuffling problem

vocab = ['<PAD>', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '\n', 'Ball', 'There are', ' is in', 'Switch', ' switches', ' of', ' and', ' cup', ' cups']

tokenizer = Tokenizer(vocab)

def initial_ball_position(n=3):
    return random.randint(1, n)

def generate_shuffle_moves(n=3, num_moves=3):
    moves = []
    
    for _ in range(num_moves):
        # Randomly pick two different cups
        cup1, cup2 = random.sample(range(1, n + 1), 2)
        moves.append((cup1, cup2))
    
    return moves

def final_ball_position(initial_position, shuffle_moves):
    position = initial_position
    for move in shuffle_moves:
        # If the ball's current position matches one of the cups in the move, swap it.
        if position == move[0]:
            position = move[1]
        elif position == move[1]:
            position = move[0]
    
    return position
    
# Generate batches of data
def generate_batch_cup_data(n_cups = 3, num_examples=1000, num_moves=3, verbose=False):
    inputs_idx = []
    targets_idx = []
    pad_token = tokenizer.encode_tokens('<PAD>')
    pad_token_tensor = torch.tensor(pad_token[0])
    
    for _ in range(num_examples):
        n_moves = random.choice(range(1,num_moves))

        input, target = generate_masked_cup_shuffling_scenario(n_cups=n_cups, n_moves=n_moves)
        inputs_idx.append(tokenizer.encode_tokens(input))

        # Finds the last cup number in the string
        targets_idx.append(tokenizer.encode_tokens(target))
    
    max_inputs_len = max(len(input) for input in inputs_idx)
    
    padded_input_ids = [input + pad_token * (max_inputs_len - len(input)) for input in inputs_idx]
    target_ids = [targets for targets in targets_idx]
        
    input_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    target_tensor = torch.tensor(target_ids, dtype=torch.long)

    masked_tensor = (input_tensor == pad_token_tensor).long().argmax(dim=1) 
    
    return input_tensor, target_tensor, masked_tensor
    
def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

def dims(a):
    for i in range(len(a)):
        print(f"Len of dimension {i}: {len(a[i])}")

def generate_cup_shuffling_scenario(n=3, num_moves=3):
    # Generate initial ball position and shuffle movesx
    initial_position = initial_ball_position(n)
    shuffle_moves = generate_shuffle_moves(n, num_moves)
    
    # Calculate the final ball position
    final_position = final_ball_position(initial_position, shuffle_moves)
    
    # Construct the input and output strings
    string = f"There are {n} cups and {num_moves} switches of cups\n"
    string += f"Ball is in cup {initial_position}\n"
    string += "\n".join([f"Switch cup {move[0]} and cup {move[1]}" for move in shuffle_moves])
    string += f"\nBall is in cup {final_position}"
    
    return string

def generate_masked_cup_shuffling_scenario(n_cups=3, n_moves=3):
    # Generate initial ball position and shuffle moves
    initial_position = initial_ball_position(n_cups)
    shuffle_moves = generate_shuffle_moves(n_cups, n_moves)
    
    # Calculate the final ball position for tokenizing
    final_position = " " + str(final_ball_position(initial_position, shuffle_moves))
    
    # Construct the input and output strings
    input = f"There are {n_cups} cups and {n_moves} switches of cups\n"
    input += f"Ball is in cup {initial_position}\n"
    input += "\n".join([f"Switch cup {move[0]} and cup {move[1]}" for move in shuffle_moves])
    input += f"\nBall is in cup<PAD>"
    
    return input, final_position

class CupDataset(Dataset):
    def __init__(self, split='train', config=None):
        self.inputs, self.targets, self.masked_positions = self._prepare_data(split, config)

    def _prepare_data(self, split, config):
        # Assuming generate_batch_cup_data is your data generation function
        if split not in ["train", "val"]:
            raise ValueError("split must be either 'train' or 'val'")

        # Generate data
        generated_data = generate_batch_cup_data(n_cups=config.n_cups, 
                                                 num_moves=config.n_moves, 
                                                 num_examples=config.n_samples)
       

        # Split the data
        split_index = int(config.training_split * len(generated_data[0]))
        if split == 'train':
            split_data = [d[:split_index] for d in generated_data]
            print(f"Splitting data into {len(split_data[0])} training examples")
            return split_data
        else:  # split == 'val'
            split_data = [d[split_index:] for d in generated_data]
            print(f"Splitting data into {len(split_data[0])} validation examples")
            return split_data
        
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "targets": self.targets[idx],
            "masked_positions": self.masked_positions[idx]
        }