from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from hyperparameters import hyperparameters
from models.masks import create_transformer_masks


class BayesMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, p_dropout: float) -> None:
        super(BayesMultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)

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

        # Apply an *independent* dropout mask to q, k, v for each head
        # The easiest way is to sample a mask of shape (batch_size, num_heads, 1, d_k)
        # (or (B, num_heads, T, head_dim) if you want *per-token* dropout).
        # Here we do per-head/feature dropout for demonstration.

        if self.training and self.p_dropout > 0:
            q_dropout = (
                torch.rand(batch_size, self.num_heads, 1, self.d_k, device=query.device)
                > self.p_dropout
            ).float()
            k_dropout = (
                torch.rand(batch_size, self.num_heads, 1, self.d_k, device=query.device)
                > self.p_dropout
            ).float()
            v_dropout = (
                torch.rand(batch_size, self.num_heads, 1, self.d_k, device=query.device)
                > self.p_dropout
            ).float()

            Q = Q * q_dropout
            K = K * k_dropout
            V = V * v_dropout

        attention_output = nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
        )

        # Concatenate heads
        # (batch_size, num_heads, seq_length, d_k) -> (batch_size, seq_length, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.out(attention_output)
        assert isinstance(output, torch.Tensor)
        return output


class BayesFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(BayesFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        output = self.linear2(self.dropout(self.relu(self.linear1(x))))
        assert isinstance(output, torch.Tensor)
        return output


class BayesEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super(BayesEncoderLayer, self).__init__()
        self.self_attn = BayesMultiheadAttention(d_model, num_heads, p_dropout=dropout)
        self.feed_forward = BayesFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(
            d_model
        )  # Authors use softmax instead of layernorm here
        self.dropout_skip_connection = nn.Dropout(dropout)

        # Dropout on the input to the feed-forward block
        self.dropout_mlp_input = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Self-attention + skip connection
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout_skip_connection(attn_output)  # orange dropout
        x = self.norm1(x)

        # Dropout on the input to the feed-forward block
        x = self.dropout_mlp_input(x)  # pink dropout

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout_skip_connection(ff_output)  # brown dropout
        x = self.norm2(x)

        return x


class BayesEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super(BayesEncoder, self).__init__()

        self.d_model = d_model
        self.p_dropout = dropout

        # The stack of BayesFormer encoder layers
        self.layers = nn.ModuleList(
            [
                BayesEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
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


class BayesDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super(BayesDecoderLayer, self).__init__()
        self.self_attn = BayesMultiheadAttention(d_model, num_heads, p_dropout=dropout)
        self.cross_attn = BayesMultiheadAttention(d_model, num_heads, p_dropout=dropout)
        self.feed_forward = BayesFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_skip_connection = nn.Dropout(dropout)

        # Dropout on the input to the feed-forward block
        self.dropout_mlp_input = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, enc_output: Tensor, tgt_mask: Tensor, memory_mask: Tensor
    ) -> Tensor:
        # Masked self-attention + skip connection
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout_skip_connection(attn_output)
        x = self.norm1(x)

        # Cross-attention sub-layer + skip connection
        attn_output = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = x + self.dropout_skip_connection(attn_output)
        x = self.norm2(x)

        # Dropout on the input to the feed-forward block
        x = self.dropout_mlp_input(x)

        # Feed-forward sub-layer + skip connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout_skip_connection(ff_output)
        x = self.norm3(x)

        return x


class BayesDecoder(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float
    ) -> None:
        super(BayesDecoder, self).__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList(
            [
                BayesDecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
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


class BayesTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super(BayesTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.encoder = BayesEncoder(
            d_model=d_model,
            num_heads=nhead,
            d_ff=dim_feedforward,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.decoder = BayesDecoder(
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
        enc_src_mask, tgt_mask, memory_mask = create_transformer_masks(
            src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask, self.nhead
        )

        enc_output = self.encoder(src, enc_src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)

        assert isinstance(dec_output, torch.Tensor)
        return dec_output
