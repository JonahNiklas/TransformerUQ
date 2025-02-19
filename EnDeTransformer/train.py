import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from EnDeTransformer.hyperparameters import hyperparameters

from EnDeTransformer.utils.checkpoints import save_checkpoint
from EnDeTransformer.validate import validate
import logging

logger = logging.getLogger(__name__)


def train(
    model: nn.Module,
    training_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,  # Add scheduler parameter
    criterion: nn.Module,
    max_steps: int,
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
                src_tokens, tgt_tokens = src_tokens.to(
                    hyperparameters.device
                ), tgt_tokens.to(hyperparameters.device)

                batch_size = src_tokens.shape[0]
                tgt_len = tgt_tokens.shape[1]
                vocab_size = model.vocab_size
                assert vocab_size > 10_000

                optimizer.zero_grad()
                decoder_input = tgt_tokens[:, :-1]
                labels = tgt_tokens[:, 1:]
                logits = model(src_tokens, decoder_input)
                logits = logits.transpose(1, 2)
                assert logits.shape == (batch_size, vocab_size, tgt_len - 1)
                assert labels.shape == (batch_size, tgt_len - 1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate

                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})

                wandb.log(
                    {"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]},
                    step=step_num,
                )

                if step_num % hyperparameters.training.validate_every == 0:
                    bleu = validate(model, test_loader)
                    # wandb.log({"val_loss": val_loss, "bleu": bleu}, step=step_num)
                    wandb.log({"bleu": bleu}, step=step_num)
                    model.train()
                if step_num % hyperparameters.training.save_every == 0:
                    os.makedirs("EnDeTransformer/local/checkpoints", exist_ok=True)
                    save_checkpoint(
                        model, optimizer, f"EnDeTransformer/local/checkpoints/checkpoint-{step_num}.pth"
                    )
                    wandb.save(f"EnDeTransformer/local/checkpoints/checkpoint-{step_num}.pth")
