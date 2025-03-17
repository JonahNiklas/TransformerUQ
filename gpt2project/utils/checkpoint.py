from __future__ import annotations

import wandb
import os
from typing import Tuple
import torch
from gpt2project.hyperparameters import GPT2ModelConfig, TrainingConfig, hyperparameters
from gpt2project.gpt2model import GPT
from gpt2project.bayesformer_gpt import BayesformerGPT
from gpt2project.ddp import device_type


def save_checkpoint(
    model: GPT | BayesformerGPT,
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
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(
    checkpoint_path: str,
    remove_orig_prefix: bool,
) -> Tuple[GPT | BayesformerGPT, int, torch.optim.Optimizer]:
    checkpoint = torch.load(checkpoint_path)
    model_config = GPT2ModelConfig.model_validate(checkpoint["model_config"])
    if model_config.transformer_impl == "bayesformer":
        model = BayesformerGPT(model_config)
    else:
        model = GPT(model_config)

    if remove_orig_prefix:
        checkpoint["model"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model"].items()
        }
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

def get_model_from_wandb_checkpoint(wandb_artifact_path: str,checkpoint_name:str, remove_orig_prefix:bool) -> None:
    """Loads a model from a wandb checkpoint"""
    os.makedirs("local/gpt_checkpoints", exist_ok=True)
    artifact_dir = "local/gpt_checkpoints"

    if not os.path.exists(os.path.join(artifact_dir, checkpoint_name)):
        api = wandb.Api()
        artifact = api.artifact(wandb_artifact_path)
        artifact.download(artifact_dir)
    from gpt2project.utils.checkpoint import load_checkpoint
    model, step, optimizer = load_checkpoint(artifact_dir+"/"+checkpoint_name, remove_orig_prefix)
    return model