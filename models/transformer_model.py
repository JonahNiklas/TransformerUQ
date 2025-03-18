import math
import torch
import torch.nn as nn

from torch.nn import functional as F
from hyperparameters import hyperparameters
from models.bayesformer import BayesTransformer
from models.transformer import Transformer as TransformerOwn
from shared.dropout_embedding import DropoutEmbedding


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
            hyperparameters.transformer.dropout_pre_embedding
            if hyperparameters.transformer.transformer_implementation == "bayesformer"
            else 0
        )
        self.embedding = DropoutEmbedding(
            vocab_size, d_model, padding_idx=0, dropout=positional_dropout
        )
        self.pos_encoder = LearnedPositionalEncoding(
            d_model, max_len=max_len, dropout=positional_dropout
        )
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

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0
    ) -> torch.Tensor:

        src_key_padding_mask = src == pad_idx
        tgt_key_padding_mask = tgt == pad_idx
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )

        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)

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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.pos_embedding = DropoutEmbedding(
            max_len,
            d_model,
            dropout=dropout,
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
