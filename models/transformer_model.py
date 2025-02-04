import math
import torch
import torch.nn as nn

from torch.nn import functional as F
from hyperparameters import hyperparameters
from models.bayesformer import BayesTransformer
from models.transformer import Transformer as TransformerOwn


class TransformerPyTorch(nn.Module):
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
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
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

        # Apply dropout to the rows of the embedding matrix
        if hyperparameters.transformer.transformer_implementation == "bayesformer":
            src = self.dropout(src)
            tgt = self.dropout(tgt)

        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        positional_dropout = (
            hyperparameters.transformer.dropout
            if hyperparameters.transformer.transformer_implementation == "bayesformer"
            else 0
        )
        src = self.pos_encoder(src, dropout=positional_dropout)
        tgt = self.pos_encoder(tgt, dropout=positional_dropout)

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


class PositionalEncoding(nn.Module):
    """
    If a dropout rate is provided, this module will apply row dropout (i.e., drop entire position vectors)
    independently for each sample. This simulates dropping rows from the positional encoding matrix before
    it is added to the token embeddings.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        pe = (
            self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        )  # (batch_size, seq_len, d_model)
        if self.training and self.dropout_rate > 0:
            mask = (
                torch.rand(batch_size, seq_len, 1, device=x.device) > self.dropout_rate
            ).float()  # (batch_size, seq_len, 1)
            pe = pe * mask / (1 - self.dropout_rate)
        x = x + pe
        return x
