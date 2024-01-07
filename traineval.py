import torch
import time
import torch.nn.functional as F
from matplotlib import pyplot as plt

# Method for evaluating the loss of the PyTorch model on the validation set without defining the model.
@torch.no_grad()
def evaluate_model(model, train_loader, val_loader):
    out = {}
    model.eval()

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        total_tokens = 0

        for batch in loader:
            # Get the inputs and targets
            inputs = batch["inputs"]
            targets = batch["targets"]
            mask_tensor = batch["masked_positions"]

            # Get the model outputs
            logits, _ = model(inputs)

            target_logits = logits[torch.arange(logits.size(0)), mask_tensor]

            loss = F.cross_entropy(target_logits, targets.squeeze(1))

            # Update the total loss and tokens
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

        # Calculate the average loss
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

        # Store the average loss
        out[split] = avg_loss

    model.train()
    return out

# Train the model with evaluate_model
def train_model(model, optimizer, config, train_loader, val_loader):
    # Set the model to train mode
    model.train()
    
    # Initialize the losses
    losses = []
    
    # Initialize the timer
    start = time.time()
    
    # Loop over the training data
    for i, batch in enumerate(train_loader):
        # Get the inputs and targets
        inputs = batch["inputs"]
        targets = batch["targets"]
        mask_tensor = batch["masked_positions"]
        
        #Inputs should be the tokens before the masked token, target is in targets
        
        #Find position of masked token

        # Get the model outputs
        logits, _ = model(inputs)

        target_logits = logits[torch.arange(logits.size(0)), mask_tensor]

        loss = F.cross_entropy(target_logits,targets.squeeze(1))
        # Backpropagate the loss
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Append the loss
        losses.append(loss.item())
        
        # Print the loss every 100 iterations
        if i % config.batch_size * 10 == 0:
            print(f"Training batch {i}, loss = {loss.item()}")
            
        # Evaluate the model every 1000 iterations
        if i % config.batch_size * 100 == 0:
            print(f"Evaluating model at batch {i}")
            eval_out = evaluate_model(model, train_loader, val_loader)
            print(f"Train loss = {eval_out['train']}, val loss = {eval_out['val']}")
            
    # Evaluate the model at the end of training
    print("Evaluating model...")
    eval_out = evaluate_model(model, train_loader, val_loader)
    print(f"Train loss = {eval_out['train']}, val loss = {eval_out['val']}")
    
    # Print the total time
    print(f"Total time: {time.time() - start} seconds")
    
    # Plot the losses
    plt.plot(losses)
    plt.show()
    
    return model