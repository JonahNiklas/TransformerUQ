class TransformerHyperparameters:
    def __init__(self) -> None:
        self.encoder_embed_dim: int = 512
        self.encoder_ffn_embed_dim: int = 1024
        self.encoder_attention_heads: int = 4
        self.encoder_layers: int = 6
        self.dropout: float = 0.1
        self.max_len: int = 512


class TrainingHyperparameters:
    def __init__(self) -> None:
        self.max_steps: int = 500000
        self.validate_every: int = 5000
        self.label_smoothing: float = 0.1
        self.batch_size: int = 64
        self.shuffle: bool = True
        self.learning_rate: float = 0.0001

class VocabHyperparameters:
    def __init__(self) -> None:
        self.token_min_freq: int = 2000

class Hyperparameter:
    def __init__(self) -> None:
        self.transformer = TransformerHyperparameters()
        self.training = TrainingHyperparameters()
        self.vocab = VocabHyperparameters()


hyperparameters = Hyperparameter()
