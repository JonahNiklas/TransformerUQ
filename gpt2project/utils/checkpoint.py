import os
from typing import Tuple
import torch
from gpt2project.hyperparameters import GPT2ModelConfig, TrainingConfig, hyperparameters
from gpt2project.gpt2model import GPT
from gpt2project.ddp import device_type


def save_checkpoint(
    model: GPT,
    step: int,
    val_loss: float,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "model_config": model.config.model_dump(),
        "training_config": hyperparameters.training.model_dump(),
        "step": step,
        "val_loss": val_loss,
        "optimizer": optimizer.state_dict(),
    }
    # you might also want to add rng seeds etc., if you wanted to more exactly resume training
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Tuple[GPT, int, torch.optim.Optimizer]:
    checkpoint = torch.load(checkpoint_path)
    model_config = GPT2ModelConfig.model_validate(checkpoint["model_config"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    training_config: TrainingConfig = TrainingConfig.model_validate(
        checkpoint["training_config"]
    )
    optimizer = model.configure_optimizers(
        weight_decay=training_config.weight_decay,
        learning_rate=training_config.max_lr,
        device_type=device_type,
    )

    return model, checkpoint["step"], optimizer
