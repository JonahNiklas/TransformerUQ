from typing import List, Tuple
from pydantic import BaseModel

# class FairseqTransformerHyperparameters(BaseModel):
#     # def transformer_vaswani_wmt_en_de_big(args):
#     # args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
#     # args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
#     # args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
#     # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     # args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
#     # args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
#     # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
#     # args.dropout = getattr(args, "dropout", 0.3)
#     # base_architecture(args)
#     hidden_size: int = 1024
#     max_len = 256
#     encoder_ffn_embed_dim: int = 4096
#     num_heads: int = 16
#     num_hidden_layers: int = 6
#     dropout: float = 0.3


# Taken from tensor2tensor: https://github.com/tensorflow/tensor2tensor/blob/28adf2690c551ef0f570d41bef2019d9c502ec7e/tensor2tensor/models/transformer.py#L1627
class TransformerHyperparameters(BaseModel):
    hidden_size: int = 512  # found in t2t
    max_len: int = 256  # found in t2t
    encoder_ffn_embed_dim: int = 2048  # found in t2t, known as filter_size in t2t
    num_heads: int = 8  # found in t2t
    num_hidden_layers: int = 6  # found in t2t
    dropout: float = 0.2  # found in t2t, transformer_base_v1()


class TrainingHyperparameters(BaseModel):
    max_steps: int = 350_000  # known from wat zei je
    validate_every: int = 5000
    label_smoothing: float = 0.1  # found in t2t
    batch_size: int = (
        64  # ALTERED found in t2t, batch size of 4096 means number of examples per batch i.e. 4096/256 = 16
    )
    shuffle: bool = True
    learning_rate_decay_scheme: str = "warmup_cosine_decay" # found in nanoGPT
    learning_rate: float = 6e-4  # found in nanogpt
    learning_rate_warm_up_steps: int = 4000  # found in t2t
    adam_betas: Tuple[float, float] = (0.9, 0.98)  # found in t2t


class VocabHyperparameters(BaseModel):
    token_min_freq: int = 2000


class Hyperparameter(BaseModel):
    transformer: TransformerHyperparameters = TransformerHyperparameters()
    training: TrainingHyperparameters = TrainingHyperparameters()
    vocab: VocabHyperparameters = VocabHyperparameters()


hyperparameters = Hyperparameter()


if __name__ == "__main__":
    print(hyperparameters.model_dump())
