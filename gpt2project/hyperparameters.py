from __future__ import annotations
import logging
from typing import Literal, cast
from pydantic import BaseModel, Field
import os

logger = logging.getLogger(__name__)

transformer_impl_raw = os.getenv("TRANSFORMER_IMPL")
transformer_impl_str = transformer_impl_raw or "transformer"
if transformer_impl_raw is None:
    logger.warning("TRANSFORMER_IMPL is not set, using default value: transformer")
if transformer_impl_str not in ("transformer", "bayesformer"):
    raise ValueError(f"Invalid TRANSFORMER_IMPL value: {transformer_impl_str}, must be one of: transformer, bayesformer")

transformer_impl: Literal["transformer", "bayesformer"] = cast(Literal["transformer", "bayesformer"], transformer_impl_str)

class GPT2ModelConfig(BaseModel):
    """Configuration for the GPT2 model architecture."""

    block_size: int = Field(default=1024, description="Maximum sequence length")
    vocab_size: int = Field(
        default=50304,
        description="Number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token",
    )
    n_layer: int = Field(default=12, description="Number of transformer layers")
    n_head: int = Field(default=12, description="Number of attention heads")
    n_embd: int = Field(default=768, description="Embedding dimension")
    dropout: float = Field(default=0.1, description="Dropout rate")
    dropout_mlp_input: float = Field(
        default=0.05, description="Dropout rate for MLP input"
    )
    dropout_pre_embedding: float = Field(
        default=0.05, description="Dropout rate for pre-embedding"
    )
    transformer_impl: Literal["transformer", "bayesformer"] = Field(
        default=transformer_impl, description="Transformer implementation"
    )


class TrainingConfig(BaseModel):
    """Configuration for the training process."""

    total_batch_size: int = Field(
        default=524288, description="Total batch size in tokens (2**19, ~0.5M)"
    )
    micro_batch_size: int = Field(default=16, description="Micro batch size in samples")
    sequence_length: int = Field(default=1024, description="Sequence length")

    max_steps: int = Field(
        default=19073 * 4,
        description="Maximum number of training steps (19,073 steps is ~1 epoch for 10B tokens and batch size 0.5M tokens)",
    )
    warmup_steps: int = Field(
        default=715, description="Number of warmup steps for learning rate scheduler"
    )

    max_lr: float = Field(default=6e-4, description="Maximum learning rate")
    min_lr: float = Field(default=6e-5, description="Minimum learning rate")
    weight_decay: float = Field(
        default=0.1, description="Weight decay for AdamW optimizer"
    )

    val_loss_steps: int = Field(
        default=20, description="Number of steps for validation loss computation"
    )

    evaluate_every: int = Field(
        default=250, description="Evaluate validation loss and hellaswag every X steps"
    )
    save_every: int = Field(
        default=20000, description="Save model checkpoint every X steps"
    )

    grad_clip: float = Field(default=1.0, description="Gradient clipping norm")


class WandBConfig(BaseModel):
    """Configuration for Weights & Biases."""

    enabled: bool = Field(
        default_factory=lambda: os.getenv("USE_WANDB", "TRUE") != "FALSE",
        description="Whether to use WandB for logging",
    )
    project: str = Field(default="GPT2Project", description="WandB project name")
    entity: str | None = Field(
        default_factory=lambda: os.getenv("WANDB_ENTITY", None),
        description="WandB entity name",
    )
    dir: str = Field(default="local", description="Directory for WandB logs")
    mode: Literal["online", "offline", "disabled"] = Field(
        default_factory=lambda: (
            "online" if os.getenv("USE_WANDB", "TRUE") != "FALSE" else "disabled"
        ),
        description="WandB mode",
    )


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    num_return_sequences: int = Field(
        default=4, description="Number of sequences to generate"
    )
    max_length: int = Field(
        default=32, description="Maximum length of generated sequences"
    )
    seed: int = Field(default=1337, description="Random seed for generation")


class GPT2Hyperparameters(BaseModel):
    """Main hyperparameters class that includes all configuration groups."""

    model: GPT2ModelConfig = Field(default_factory=GPT2ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# Create default hyperparameters instance
hyperparameters = GPT2Hyperparameters()
