import torch
from torch import nn, optim
from data_processing.vocab import Vocabulary
from hyperparameters import hyperparameters


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str) -> None:
    checkpoint = torch.load(
        path, map_location=hyperparameters.device, weights_only=True
    )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError as error:
        checkpoint["model_state_dict"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
