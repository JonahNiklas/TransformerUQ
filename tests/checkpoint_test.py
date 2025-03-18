import os
from typing import Any, Dict, OrderedDict

import torch
from gpt2project.gpt2model import GPT
from gpt2project.hyperparameters import TrainingConfig, GPT2ModelConfig
from gpt2project.utils.checkpoint import load_checkpoint, save_checkpoint


def test_save_load_checkpoint() -> None:
    test_config = GPT2ModelConfig(
        block_size=10,
        vocab_size=100,
        n_layer=1,
        n_head=1,
        n_embd=24,
    )
    test_training_config = TrainingConfig(max_lr=0.001, weight_decay=0.0)
    device_type = "cpu"
    test_model = GPT(test_config)
    num_params = sum(p.numel() for p in test_model.parameters())
    test_optimizer = test_model.configure_optimizers(
        weight_decay=test_training_config.weight_decay,
        learning_rate=test_training_config.max_lr,
        device_type=device_type,
    )

    test_step = 10
    test_val_loss = 0.1
    test_checkpoint_path = "local/test_checkpoint.pt"
    os.makedirs("local", exist_ok=True)

    save_checkpoint(
        test_model, test_step, test_val_loss, test_checkpoint_path, test_optimizer
    )

    loaded_model, loaded_step, loaded_optimizer = load_checkpoint(test_checkpoint_path)

    assert loaded_model.config == test_config
    assert loaded_step == test_step
    for param, loaded_param in zip(test_model.parameters(), loaded_model.parameters()):
        assert (param == loaded_param).all()

    os.remove(test_checkpoint_path)
