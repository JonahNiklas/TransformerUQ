import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional, Tuple


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    query, key, value shape: (batch_size, num_heads, seq_length, d_k)
    mask: same shape broadcastable to (batch_size, 1, seq_length, seq_length)
    Returns:
      - attention_output: (batch_size, num_heads, seq_length, d_k)
      - attention_weights: (batch_size, num_heads, seq_length, seq_length)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask is typically applied by setting scores to a large negative value
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = torch.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value)
    return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learnable linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Final linear layer after concat of all heads
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = query.size(0)

        # 1) Linear projections: (batch_size, seq_length, d_model) -> (batch_size, seq_length, num_heads * d_k)
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 2) Split into heads: (batch_size, seq_length, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3) Apply scaled dot-product attention
        #    Q, K, V shape: (batch_size, num_heads, seq_length, d_k)
        attention_output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 4) Concatenate heads
        # (batch_size, num_heads, seq_length, d_k) -> (batch_size, seq_length, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # 5) Final linear layer
        output = self.out(attention_output)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()

        # Create a long enough PEmatrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer so it is not a parameter but is saved in the state_dict
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, seq_length, d_model)
        """
        seq_len = x.size(1)
        # Add the positional encoding to embeddings
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention sub-layer
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        max_len: int,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model

        # Token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        src: (batch_size, src_seq_length)
        src_mask: (batch_size, 1, 1, src_seq_length) or (batch_size, 1, src_seq_length, src_seq_length)
        """
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, enc_output: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        # 1) Masked self-attention
        _x = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = _x + self.dropout(x)
        x = self.norm1(x)

        # 2) Cross-attention
        _x = x
        x = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = _x + self.dropout(x)
        x = self.norm2(x)

        # 3) Feed-forward
        _x = x
        x = self.feed_forward(x)
        x = _x + self.dropout(x)
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        max_len: int,
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model

        # Token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, enc_output: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        tgt: (batch_size, tgt_seq_length)
        enc_output: (batch_size, src_seq_length, d_model)
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return x
    


def make_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a mask for padding tokens in `seq`.
    seq: (batch_size, seq_len)
    Returns a binary mask of shape (batch_size, 1, 1, seq_len),
    where '1' indicates "allowed to attend" and '0' indicates "PAD/masked out".
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # shape -> (B, 1, 1, S)

def make_subsequent_mask(size: int) -> torch.Tensor:
    """
    Creates a causal (look-ahead) mask of shape (size, size).
    Lower triangular is 1 (allow), upper triangular is 0 (block).
    """
    return torch.tril(torch.ones(size, size)).bool()  # (S, S)


def create_src_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a source mask that masks out source <pad> tokens.
    src: (batch_size, src_seq_len)
    """
    return make_padding_mask(src, pad_idx)  # (B, 1, 1, src_len)

def create_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a target mask that (1) masks out <pad> tokens
    and (2) prevents the model from 'looking ahead' (causal mask).
    tgt: (batch_size, tgt_seq_len)
    """
    # 1) Padding mask
    tgt_pad_mask = make_padding_mask(tgt, pad_idx)  # (B, 1, 1, tgt_len)

    # 2) Subsequent (causal) mask
    seq_len = tgt.size(1)
    causal_mask = make_subsequent_mask(seq_len).to(tgt.device)  # (tgt_len, tgt_len)
    causal_mask = causal_mask.unsqueeze(0)  # (1, tgt_len, tgt_len), broadcast over batch

    # Combine them by logical AND: 0's in either => block
    # shape after broadcast: (B, 1, tgt_len, tgt_len)
    combined_mask = tgt_pad_mask & causal_mask
    return combined_mask



class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        max_len: int,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_encoder_layers,
            dropout,
            max_len,
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_decoder_layers,
            dropout,
            max_len,
        )

        # Final linear layer to project decoder outputs to vocab
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        src: (batch_size, src_seq_length)
        tgt: (batch_size, tgt_seq_length)
        """

        src_mask = create_src_mask(src, pad_idx=0)
        tgt_mask = create_tgt_mask(tgt, pad_idx=0)
        memory_mask = src_mask

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        out = self.output_projection(dec_output)
        return out


if __name__ == "__main__":

    class Hyperparameter:
        def __init__(self):
            self.encoder_embed_dim: int = 512
            self.encoder_ffn_embed_dim: int = 1024
            self.encoder_attention_heads: int = 4
            self.encoder_layers: int = 6
            self.src_vocab_size: int = 6632
            self.trg_vocab_size: int = 8848


    hyperparameters = Hyperparameter()

    src = torch.randint(0, hyperparameters.src_vocab_size, (2, 10))
    model = Transformer(
        src_vocab_size=hyperparameters.src_vocab_size,
        tgt_vocab_size=hyperparameters.trg_vocab_size,
        d_model=hyperparameters.encoder_embed_dim,
        num_heads=hyperparameters.encoder_attention_heads,
        d_ff=hyperparameters.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.encoder_layers,
        num_decoder_layers=hyperparameters.encoder_layers,
        dropout=0.1,
        max_len=512,
    )
    number_of_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {number_of_params/1e6:.2f}M")
    output = model(src, src)
    print(output.shape)
