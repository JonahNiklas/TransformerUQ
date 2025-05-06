from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from hyperparameters import hyperparameters
from models.concrete_dropout import ConcreteDropout
from models.masks import create_transformer_masks


class BayesMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, p_dropout: float) -> None:
        super(BayesMultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.dropout = ConcreteDropout()

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size, q_seq_length, _ = query.shape
        k_seq_length = key.shape[1]

        Q, regularization1 = self.dropout(query, self.W_q) # red dropout
        K, regularization2 = self.dropout(key, self.W_k) # green dropout
        V, regularization3 = self.dropout(value, self.W_v) # blue dropout

        # Split into (batch_size, num_heads, seq_length, d_k)
        Q = Q.view(batch_size, q_seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, k_seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, k_seq_length, self.num_heads, self.d_k).transpose(1, 2)

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
        return output, regularization1 + regularization2 + regularization3


class BayesFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(BayesFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = ConcreteDropout()
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.relu(self.linear1(x))
        output, regularization = self.dropout(output, self.linear2)
        assert isinstance(output, torch.Tensor)
        return output, regularization


class BayesEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super(BayesEncoderLayer, self).__init__()
        self.self_attn = BayesMultiheadAttention(d_model, num_heads, p_dropout=dropout)
        self.feed_forward = BayesFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_skip_connection = ConcreteDropout()

        # Dropout on the input to the feed-forward block
        self.dropout_mlp_input = ConcreteDropout()

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Self-attention sub-layer
        attn_output, regularization1 = self.self_attn(x, x, x, mask)
        skip_connection, regularization2 = self.dropout_skip_connection(attn_output, None)
        x = x + skip_connection  # orange dropout
        x = self.norm1(x)

        # Dropout on the input to the feed-forward block
        x, regularization3 = self.dropout_mlp_input(x, None)  # pink dropout

        # Feed-forward sub-layer
        ff_output, regularization4 = self.feed_forward(x)
        skip_connection, regularization5 = self.dropout_skip_connection(ff_output, None)
        x = x + skip_connection  # brown dropout
        x = self.norm2(x)

        return x, regularization1 + regularization2 + regularization3 + regularization4 + regularization5


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
        # The stack of BayesFormer encoder layers
        self.layers = nn.ModuleList(
            [
                BayesEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor, src_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        src: (batch_size, src_seq_length)
        src_mask: (batch_size, 1, 1, src_seq_length) or (batch_size, 1, src_seq_length, src_seq_length)
        """
        regularization: Tensor = 0 # type: ignore
        for layer in self.layers:
            src, regularization_layer = layer(src, src_mask)
            regularization += regularization_layer

        assert isinstance(src, torch.Tensor)
        return src, regularization


class BayesDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super(BayesDecoderLayer, self).__init__()
        self.self_attn = BayesMultiheadAttention(d_model, num_heads, p_dropout=dropout)
        self.cross_attn = BayesMultiheadAttention(d_model, num_heads, p_dropout=dropout)
        self.feed_forward = BayesFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_skip_connection = ConcreteDropout()

        # Dropout on the input to the feed-forward block
        self.dropout_mlp_input = ConcreteDropout()

    def forward(
        self, x: Tensor, enc_output: Tensor, tgt_mask: Tensor, memory_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Masked self-attention
        attn_output, regularization1 = self.self_attn(x, x, x, tgt_mask)
        skip_connection, regularization2 = self.dropout_skip_connection(attn_output, None)
        x = x + skip_connection  # orange dropout
        x = self.norm1(x)

        # Cross-attention sub-layer
        attn_output, regularization3 = self.cross_attn(x, enc_output, enc_output, memory_mask)
        skip_connection, regularization4 = self.dropout_skip_connection(attn_output, None)
        x = x + skip_connection  # green dropout
        x = self.norm2(x)

        # Dropout on the input to the feed-forward block
        x, regularization5 = self.dropout_mlp_input(x, None)

        # Feed-forward sub-layer
        ff_output, regularization6 = self.feed_forward(x)
        skip_connection, regularization7 = self.dropout_skip_connection(ff_output, None)
        x = x + skip_connection  # brown dropout
        x = self.norm3(x)

        return x, regularization1 + regularization2 + regularization3 + regularization4 + regularization5 + regularization6 + regularization7


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
    ) -> Tuple[Tensor, Tensor]:
        """
        tgt: (batch_size, tgt_seq_length)
        enc_output: (batch_size, src_seq_length, d_model)
        """
        regularization: Tensor = 0 # type: ignore
        for layer in self.layers:
            tgt, regularization_layer = layer(tgt, enc_output, tgt_mask, memory_mask)
            regularization += regularization_layer

        assert isinstance(tgt, torch.Tensor)
        return tgt, regularization


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
    ) -> Tuple[Tensor, Tensor]:
        """
        src: (batch_size, src_seq_length)
        tgt: (batch_size, tgt_seq_length)
        """
        enc_src_mask, tgt_mask, memory_mask = create_transformer_masks(
            src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask, self.nhead
        )

        enc_output, regularization1 = self.encoder(src, enc_src_mask)
        dec_output, regularization2 = self.decoder(tgt, enc_output, tgt_mask, memory_mask)

        assert isinstance(dec_output, torch.Tensor)
        return dec_output, regularization1 + regularization2
