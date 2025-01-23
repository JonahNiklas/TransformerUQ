import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from validate import validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    model: nn.Module,
    training_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    max_updates: int,
    validate_every: int = 1000,
):
    model.train()
    update = 0
    for update in tqdm(range(max_updates), desc=f"Step {update}/{max_updates}"):
        total_loss = 0
        for i, batch in enumerate(training_loader):
            src_tokens, tgt_tokens = batch
            src_tokens, tgt_tokens = src_tokens.to(device), tgt_tokens.to(device)

            optimizer.zero_grad()
            output = model(src_tokens, tgt_tokens)
            output = output.transpose(1, 2)
            loss = criterion(output, tgt_tokens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % validate_every == 0:
                validate(model, test_loader, criterion)

    avg_loss = total_loss / max_updates
    print(f"Average Loss: {avg_loss}")
