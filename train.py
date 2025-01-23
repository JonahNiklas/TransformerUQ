import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import validate


def train(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, max_updates: int, validate_every: int = 1000):
    model.train()
    for update in tqdm(range(max_updates), desc=f"Step {update}/{max_updates}"):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            src_tokens, src_lengths, tgt_tokens = batch['net_input']['src_tokens'], batch['net_input']['src_lengths'], batch['target']
            device = model.device
            src_tokens, src_lengths, tgt_tokens = src_tokens.to(device), src_lengths.to(device), tgt_tokens.to(device)

            optimizer.zero_grad()
            output = model(src_tokens, src_lengths, tgt_tokens)
            loss = criterion(output, tgt_tokens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % validate_every == 0:
                validate(model, test_data)

    avg_loss = total_loss / max_updates
    print(f"Average Loss: {avg_loss}")

