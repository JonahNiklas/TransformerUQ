import torch
import torch.nn as nn
import torch.nn.functional as F


class Hyperparameter():
    def __init__(self):
        self.encoder_embed_dim = 512
        self.encoder_ffn_embed_dim = 1024
        self.encoder_attention_heads = 4
        self.encoder_layers = 6
        # self.decoder_embed_dim = 512
        # self.decoder_ffn_embed_dim = 1024
        # self.decoder_attention_heads = 4
        # self.decoder_layers = 6

hyperparameters = Hyperparameter()


class BayesMultiheadAttention(nn.Module):
    """
    Multi-head self-attention with per-head dropout masks on Q, K, and V.
    """
    def __init__(self, d_model, num_heads, p_dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # self.head_dim = d_model
        
        # Linear projections for Q, K, V (one per head) 
        # For simplicity, we just use a single large projection
        # and reshape into heads. But we will apply dropout masks
        # *after* we split them for each head.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout probability
        self.p_dropout = p_dropout
        
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: optional attention mask
        """
        B, T, D = x.shape
        
        # 1. Compute Q, K, V 
        #    shape => (batch_size, seq_len, d_model)
        q = self.W_q(x)#; assert q.shape == (B, T, self.d_model)
        k = self.W_k(x)#; assert k.shape == (B, T, self.d_model)
        v = self.W_v(x)#; assert v.shape == (B, T, self.d_model)
        
        # 2. Reshape into (batch_size, num_heads, seq_len, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2); assert k.shape == (B, self.num_heads, T, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Apply an *independent* dropout mask to q, k, v for each head
        #    The easiest way is to sample a mask of shape (B, num_heads, 1, head_dim)
        #    (or (B, num_heads, T, head_dim) if you want *per-token* dropout).
        #    Here we do per-head/feature dropout for demonstration.
        
        if self.training and self.p_dropout > 0:
            # Create masks of shape (B, num_heads, 1, head_dim)
            q_mask = (torch.rand(B, self.num_heads, 1, self.head_dim, device=x.device) 
                      > self.p_dropout).float()
            k_mask = (torch.rand(B, self.num_heads, 1, self.head_dim, device=x.device)
                      > self.p_dropout).float()
            v_mask = (torch.rand(B, self.num_heads, 1, self.head_dim, device=x.device)
                      > self.p_dropout).float()
            
            q = q * q_mask
            k = k * k_mask
            v = v * v_mask
        
        # 4. Scaled dot-product attention
        #    q, k: (B, num_heads, T, head_dim)
        #    v: (B, num_heads, T, head_dim)
        
        att_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nH, T, T)
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, float('-inf'))
            
        att_weights = F.softmax(att_scores, dim=-1)  # (B, nH, T, T)
        
        # 5. Multiply by V
        out = torch.matmul(att_weights, v); assert out.shape == (B, self.num_heads, T, self.head_dim)
        
        # 6. Recombine heads by Hadamard product
        assert out.shape == (B, self.num_heads, T, self.head_dim)
        out = torch.prod(out, dim=1) # B, T, d_mode
        assert out.shape == (B, T, self.d_model)
        
        # 7. Final output projection (usually followed by dropout,
        #    but in BayesFormer the main dropout is in the Q,K,V projections).
        out = self.out_proj(out)
        
        return out

class BayesFeedForward(nn.Module):
    """A simple 2-layer MLP block used inside the Transformer encoder."""
    def __init__(self, d_model, dim_feedforward, p_dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.ReLU()   # or GELU, etc.
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class BayesTransformerEncoderLayer(nn.Module):
    """
    A single layer of the "BayesFormer" style Transformer encoder:
      - Independent dropout on Q, K, V per head
      - Dropout on input to the feed-forward block
      - Omit the usual post-attention dropout
    """
    def __init__(self, d_model, nhead, dim_feedforward, p_dropout):
        super().__init__()
        self.self_attn = BayesMultiheadAttention(d_model, nhead, p_dropout=p_dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.ff = BayesFeedForward(d_model, dim_feedforward, p_dropout=p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) # Authors use softmax instead of layernorm here
        self.dropout_skip_connection = nn.Dropout(p_dropout)
        
        # Dropout on the input to the feed-forward block
        self.dropout_mlp_input = nn.Dropout(p_dropout)
        
    def forward(self, src, src_mask=None):
        # Self-attention + skip connection
        attn_out = self.self_attn(src, src_mask)
        skip_connection = self.dropout_skip_connection(src) # arrow
        
        src = self.norm1(skip_connection + attn_out)
        
        mlp_in = self.dropout_mlp_input(src) # orange arrow
        # Dropout on input to MLP, then feed-forward
        ff_out = self.ff(mlp_in) # brown arrow within ff block
        # dropout brown arrow
        out = self.norm2(src + ff_out)

        return out

class BayesTransformerEncoder(nn.Module):
    """
    Full 'BayesFormer' encoder stack. Also demonstrates how to apply dropout
    to input embeddings *before* embedding them, as recommended.
    """
    def __init__(
        self, 
        vocab_size,
        d_model, 
        nhead, 
        num_layers, 
        dim_feedforward, 
        p_dropout
    ):
        super().__init__()
        
        self.d_model = d_model

        # For demonstration, define separate input-embedding & positional-embedding.
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)  # max length of 512 for example
        
        # We'll apply dropout masks to the "rows" of the token IDs or pos IDs
        # before calling the embedding.  One way: 
        self.p_dropout = p_dropout
        
        # The stack of BayesFormer encoder layers
        self.layers = nn.ModuleList([
            BayesTransformerEncoderLayer(
                d_model, nhead, dim_feedforward, p_dropout
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, src_tokens):
        """
        src_tokens: (batch_size, seq_len) of token IDs
        """
        B, T = src_tokens.shape
        
        # 1. Sample dropout masks for token IDs and positional IDs "rows".
        #    In practice, you might do something more sophisticated, 
        #    e.g. zero out entire rows or etc.  For simplicity, we just 
        #    show a single Bernoulli mask for each token.
        
        device = src_tokens.device
        if self.training and self.p_dropout > 0:
            # With probability p_dropout, replace token with a special [PAD] or [MASK] etc.
            mask_tokens = (torch.rand(B, T, device=device) < self.p_dropout)
            # Example: set masked positions to 0 (supposing 0 is <PAD>)
            src_tokens = src_tokens.masked_fill(mask_tokens, 0)
        
        # 2. Similarly for position IDs
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        if self.training and self.p_dropout > 0:
            mask_pos = (torch.rand(B, T, device=device) < self.p_dropout)
            pos_ids = pos_ids.masked_fill(mask_pos, 0)
        
        # 3. Convert to embeddings
        token_emb = self.token_embed(src_tokens)  # (B, T, d_model)
        pos_emb = self.pos_embed(pos_ids)         # (B, T, d_model)
        
        # 4. Use hadard multiplication to combine token and pos
        x = token_emb * pos_emb; assert x.shape == (B, T, self.d_model)
        
        # 5. Pass through the stack of BayesFormer encoder layers
        mask = None  # if you want an attention mask for padding, etc.
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        
        # x is now (B, T, d_model)
        return x

# ----------------------------
# Example usage:
if __name__ == "__main__":
    model = BayesTransformerEncoder(
        vocab_size=2_000, 
        d_model=hyperparameters.encoder_embed_dim, 
        nhead=hyperparameters.encoder_attention_heads,
        num_layers=hyperparameters.encoder_layers,
        dim_feedforward=hyperparameters.encoder_ffn_embed_dim,
        p_dropout=0.1)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params / 1e6:.1f}M parameters. The baseline has 34.5M parameters.")
    src_tokens = torch.randint(1, 10000, (8, 20))  # (batch_size=8, seq_len=20)
    # out = model(src_tokens)  # (8, 20, 128)
    # print(out.shape)


