class Hyperparameter:
    def __init__(self):
        self.encoder_embed_dim: int = 512
        self.encoder_ffn_embed_dim: int = 1024
        self.encoder_attention_heads: int = 4
        self.encoder_layers: int = 6
        self.dropout: float = 0.1
        self.max_len: int = 512

hyperparameters = Hyperparameter()