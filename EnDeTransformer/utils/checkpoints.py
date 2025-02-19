import torch
from torch import nn, optim
from EnDeTransformer.data_processing.vocab import Vocabulary
from EnDeTransformer.hyperparameters import hyperparameters


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, path: str, remove_orig_prefix: bool
) -> None:
    checkpoint = torch.load(path, map_location=hyperparameters.device, weights_only=True)

    if remove_orig_prefix:
        checkpoint["model_state_dict"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
