import math
import torch
import torch.nn as nn

from hyperparameters import hyperparameters
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
        self.pos_encoder = PositionalEncoding(d_model, max_len)
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
        else:
            raise ValueError("Invalid transformer implementation")
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.out.weight

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0
    ) -> torch.Tensor:
        src = self.dropout(src)
        tgt = self.dropout(tgt)
        
        src_key_padding_mask = src == pad_idx
        tgt_key_padding_mask = tgt == pad_idx
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
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
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :].unsqueeze(0)
        return x
