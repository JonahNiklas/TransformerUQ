from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from hyperparameters import hyperparameters
from models.masks import create_transformer_masks


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, p_dropout: float) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.p_dropout = p_dropout

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor
    ) -> Tensor:
        batch_size, seq_length, d_k = query.shape

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into (batch_size, num_heads, seq_length, d_k)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # 3) Apply scaled dot-product attention
        #    Q, K, V shape: (batch_size, num_heads, seq_length, d_k)
        # attention_output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        attention_output = nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=(
                hyperparameters.transformer.dropout if self.training else 0.0
            ),  # green dropout
        )

        # Concatenate heads
        # (batch_size, num_heads, seq_length, d_k) -> (batch_size, seq_length, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.out(attention_output)
        assert isinstance(output, torch.Tensor)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        output = self.linear2(self.dropout(self.relu(self.linear1(x))))
        assert isinstance(output, torch.Tensor)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, p_dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_skip_connection = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Self-attention sub-layer
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout_skip_connection(attn_output)  # orange dropout
        x = self.norm1(x)

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout_skip_connection(ff_output)  # blue dropout
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        src: (batch_size, src_seq_length)
        src_mask: (batch_size, 1, 1, src_seq_length) or (batch_size, 1, src_seq_length, src_seq_length)
        """
        for layer in self.layers:
            src = layer(src, src_mask)

        assert isinstance(src, torch.Tensor)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, p_dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, p_dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_skip_connection = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, enc_output: Tensor, tgt_mask: Tensor, memory_mask: Tensor
    ) -> Tensor:
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout_skip_connection(attn_output)
        x = self.norm1(x)

        # Cross-attention sub-layer
        attn_output = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = x + self.dropout_skip_connection(attn_output)
        x = self.norm2(x)

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout_skip_connection(ff_output)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float
    ) -> None:
        super(Decoder, self).__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        enc_output: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        tgt: (batch_size, tgt_seq_length)
        enc_output: (batch_size, src_seq_length, d_model)
        """
        for layer in self.layers:
            tgt = layer(tgt, enc_output, tgt_mask, memory_mask)

        assert isinstance(tgt, torch.Tensor)
        return tgt


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=nhead,
            d_ff=dim_feedforward,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=nhead,
            d_ff=dim_feedforward,
            num_layers=num_decoder_layers,
            dropout=dropout,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        src_key_padding_mask: Tensor,
        tgt_key_padding_mask: Tensor,
    ) -> Tensor:
        """
        src: (batch_size, src_seq_length)
        tgt: (batch_size, tgt_seq_length)
        """

        src = self.dropout(src)  # red dropout
        tgt = self.dropout(tgt)  # red dropout

        enc_src_mask, tgt_mask, memory_mask = create_transformer_masks(
            src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask, self.nhead
        )

        enc_output = self.encoder(src, enc_src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)

        assert isinstance(dec_output, torch.Tensor)
        return dec_output
