from __future__ import annotations

import math

import torch
import torch.nn as nn
from sympy import hyper
from torch.nn import functional as F

from hyperparameters import hyperparameters
from models.bayesformer import BayesTransformer
from models.transformer import Transformer as TransformerOwn


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        positional_dropout = (
            hyperparameters.transformer.dropout
            if hyperparameters.transformer.transformer_implementation == "bayesformer"
            else 0
        )
        self.embedding = DropoutEmbedding(
            vocab_size,
            d_model,
            padding_idx=0,
            dropout=hyperparameters.transformer.dropout,
        )
        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len=max_len)
        self.transformer: torch.nn.Module
        if hyperparameters.transformer.transformer_implementation == "pytorch":
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
        elif hyperparameters.transformer.transformer_implementation == "own":
            self.transformer = TransformerOwn(
                d_model=d_model,
                nhead=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
            )
        elif hyperparameters.transformer.transformer_implementation == "bayesformer":
            self.transformer = BayesTransformer(
                d_model=d_model,
                nhead=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
            )
        else:
            raise ValueError("Invalid transformer implementation")
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.out.weight
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0
    ) -> torch.Tensor:

        src_key_padding_mask = src == pad_idx
        tgt_key_padding_mask = tgt == pad_idx
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )

        # # # Apply dropout to the rows of the embedding matrix
        # if hyperparameters.transformer.transformer_implementation == "bayesformer":
        #     src = token_dropout(src, dropout_prob=self.dropout.p, pad_idx=pad_idx)
        #     tgt = token_dropout(tgt, dropout_prob=self.dropout.p, pad_idx=pad_idx)

        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

        # if hyperparameters.transformer.transformer_implementation == "bayesformer":
        #     src = self.dropout(src)
        #     tgt = self.dropout(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        if hyperparameters.transformer.transformer_implementation in ["pytorch", "own"]:
            src = self.dropout(src)
            tgt = self.dropout(tgt)

        out = self.transformer(
            src,
            tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        result: torch.Tensor = self.out(out)
        return result


class DropoutEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dropout: float,
        padding_idx: int | None,
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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.pos_embedding = DropoutEmbedding(
            max_len,
            d_model,
            dropout=hyperparameters.transformer.dropout,
            padding_idx=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        positions = (
            torch.arange(0, seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        pos_embeddings = self.pos_embedding(positions)
        x = x + pos_embeddings
        return x
