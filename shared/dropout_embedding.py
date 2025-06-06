from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dropout: float,
        padding_idx: Optional[int] = None,
    ) -> None:
        """
        Applies dropout to entire rows of the embedding matrix.

        Args:
            num_embeddings (int): number of embeddings (vocabulary size).
            embedding_dim (int): dimension of each embedding vector.
            dropout (float): probability of dropping an entire embedding row.
            padding_idx (int): index of the padding token (never dropped).
        """
        super().__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # When training, apply dropout to the embedding weights.
        if self.training and self.dropout > 0:
            weight = self.embedding.weight  # shape: [num_embeddings, embedding_dim]
            # Create a dropout mask for rows: shape: [num_embeddings, 1]
            mask = weight.new_empty((weight.size(0), 1)).bernoulli_(1 - self.dropout)
            # Scale the surviving rows to maintain expected values
            mask = mask / (1 - self.dropout)
            # Make sure that the padding index is always kept.
            if self.embedding.padding_idx is not None:
                mask[self.embedding.padding_idx] = 1
            # Apply the mask to zero out (drop) entire rows.
            dropped_weight = weight * mask
            # Use the masked weights for the embedding lookup.
            return F.embedding(
                input,
                dropped_weight,
                self.embedding.padding_idx,
                self.embedding.max_norm,
                self.embedding.norm_type,
                self.embedding.scale_grad_by_freq,
                self.embedding.sparse,
            )
        else:
            # In evaluation mode (or if dropout == 0), use the regular embedding.
            out = self.embedding(input)
            assert isinstance(out, torch.Tensor)
            return out
