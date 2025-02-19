from typing import List, Literal, Tuple
from pydantic import BaseModel
import torch

# 30.01.2025:
# next run I would like to increase batch size
# from 64 to 128 for more stable training
# also use dropout of 0.2 instead of 0.1


# Taken from tensor2tensor: https://github.com/tensorflow/tensor2tensor/blob/28adf2690c551ef0f570d41bef2019d9c502ec7e/tensor2tensor/models/transformer.py#L1627
class TransformerHyperparameters(BaseModel):
    hidden_size: int = 512  # found in transformer_iwslt_de_en
    max_len: int = 128  # ALTERED - 256 found in t2t
    encoder_ffn_embed_dim: int = 1024  # found in transformer_iwslt_de_en
    num_heads: int = 4  # found in transformer_iwslt_de_en
    num_hidden_layers: int = 6  # found in transformer_iwslt_de_en
    dropout: float = 0.1  # 0.1 in attention  # 0.2found in t2t, transformer_base_v1(), 0.1 used by bayesformer (fairseq)
    dropout_mlp_input: float = 0.05  # found in bayesformer
    transformer_implementation: Literal["pytorch", "own", "bayesformer"] = "bayesformer"

class TrainingHyperparameters(BaseModel):
    max_steps: int = 500_000  # found in bayesformer
    validate_every: int = 5000
    save_every: int = 50_000
    label_smoothing: float = 0.1  # found in t2t
    batch_size: int = (
        128  # ALTERED found in t2t, batch size of 4096 means number of examples per batch i.e. 4096/256 = 16
    )
    shuffle: bool = True
    # learning_rate_decay_scheme: str = "warmup_cosine_decay" # found in nanoGPT
    # learning_rate: float = 6e-4  # found in nanogpt
    learning_rate_warm_up_steps: int = 4000  # found in t2t
    adam_betas: Tuple[float, float] = (0.9, 0.98)  # found in t2t
    adam_eps: float = 1e-9  # found in attention


class VocabHyperparameters(BaseModel):
    token_min_freq: int = 1
    bpe_num_symbols: int = 10000



class UncertaintyQuantificationHyperparameters(BaseModel):
    num_inferences: int = 5


class BeamSearchHyperparameters(BaseModel):
    beam_size: int = 4


class Hyperparameter(BaseModel):
    transformer: TransformerHyperparameters = TransformerHyperparameters()
    training: TrainingHyperparameters = TrainingHyperparameters()
    vocab: VocabHyperparameters = VocabHyperparameters()
    beam_search: BeamSearchHyperparameters = BeamSearchHyperparameters()
    uq: UncertaintyQuantificationHyperparameters = (
        UncertaintyQuantificationHyperparameters()
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


hyperparameters = Hyperparameter()


if __name__ == "__main__":
    print(hyperparameters.model_dump())
