import torch
from torch.utils.data import DataLoader

from model_config import MASTER_CONFIG
from model import cup_GPT
from traineval import evaluate_model, train_model
from dataset import CupDataset

if __name__ == "__main__":
    config = MASTER_CONFIG
    
    train_dataset = CupDataset(split='train', config=config)
    val_dataset = CupDataset(split='val', config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = cup_GPT(config=config)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_model(model, optimizer, config=config, train_loader=train_loader, val_loader=val_loader)
    
    evaluate_model(model, train_loader, val_loader)