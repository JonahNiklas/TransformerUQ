import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from validate import validate
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: nn.Module,
    training_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    max_steps: int,
    validate_every: int = 5000,
) -> None:
    logger.info(
        f"Training model for {max_steps} steps or {max_steps / len(training_loader):2f} epochs"
    )
    model.train()
    step_num = 0
    with tqdm(total=max_steps, desc="Training Progress") as pbar:
        while step_num < max_steps:
            for batch in training_loader:
                step_num += 1

                src_tokens, tgt_tokens = batch
                src_tokens, tgt_tokens = src_tokens.to(device), tgt_tokens.to(device)

                optimizer.zero_grad()
                decoder_input = tgt_tokens[:, :-1]
                labels = tgt_tokens[:, 1:]
                logits = model(src_tokens, decoder_input)
                logits = logits.transpose(1, 2)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})

                wandb.log({"loss": loss.item()}, step=step_num)

                if step_num % validate_every == 0:
                    bleu = validate(model, test_loader, criterion)
                    # wandb.log({"val_loss": val_loss, "bleu": bleu}, step=step_num)
                    wandb.log({"bleu": bleu}, step=step_num)
                    os.makedirs("local/checkpoints", exist_ok=True)
                    save_checkpoint(
                        model, optimizer, f"local/checkpoints/checkpoint-{step_num}.pth"
                    )
                    wandb.save(f"local/checkpoints/checkpoint-{step_num}.pth")
                    model.train()


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
