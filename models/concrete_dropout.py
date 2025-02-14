from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from torch import nn


class ConcreteDropout(nn.Module):
    def __init__(
        self,
        weight_regularizer: float = 1e-6,
        dropout_regularizer: float = 1e-5,
        init_min: float = 0.1,
        init_max: float = 0.1,
    ) -> None:
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x: torch.Tensor, layer: nn.Module | None) -> Tuple[torch.Tensor, torch.Tensor]:
        p = torch.sigmoid(self.p_logit)

        out = self._concrete_dropout(x, p)
        if layer is not None:
            out = layer(out)

        sum_of_square: torch.Tensor = 0 # type: ignore
        if layer is not None:
            for param in layer.parameters():
                sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1.0 - p) * torch.log(1.0 - p)

        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (
            torch.log(p + eps)
            - torch.log(1 - p + eps)
            + torch.log(unif_noise + eps)
            - torch.log(1 - unif_noise + eps)
        )

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x