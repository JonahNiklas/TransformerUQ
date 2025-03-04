from typing import Literal
import pytest

from hyperparameters import hyperparameters
from models.transformer_model import TransformerModel
from uq.generate_with_uq import enable_fast_test_time_dropout
from torch import nn


@pytest.mark.parametrize(
    "transformer_implementation", ["pytorch", "own", "bayesformer"]
)
def test_enable_fast_test_time_dropout(
    transformer_implementation: Literal["pytorch", "own", "bayesformer"]
) -> None:
    # Modify hyperparameters for test
    hyperparameters.transformer.transformer_implementation = transformer_implementation

    vocab_size = 32000

    # Create model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=hyperparameters.transformer.hidden_size,
        num_heads=hyperparameters.transformer.num_heads,
        d_ff=hyperparameters.transformer.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.transformer.num_hidden_layers,
        num_decoder_layers=hyperparameters.transformer.num_hidden_layers,
        dropout=hyperparameters.transformer.dropout,
        max_len=hyperparameters.transformer.max_len,
    )
    # Initially all dropout layers should be in eval mode
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            assert not module.training, "Dropout should be in eval mode initially"

    # Enable fast test time dropout
    enable_fast_test_time_dropout(model)

    # Check that only the final decoder layer dropouts are enabled
    final_decoder_layer = model.transformer.decoder.layers[-1]
    for module in final_decoder_layer.modules():
        if isinstance(module, nn.Dropout):
            assert (
                module.training
            ), f"Dropout in final decoder layer should be enabled for {transformer_implementation}"

    # Check that other dropouts remain disabled
    for layer in model.transformer.decoder.layers[
        :-1
    ]:  # All layers except the last one
        for module in layer.modules():
            if isinstance(module, nn.Dropout):
                assert (
                    not module.training
                ), f"Dropout in non-final layers should remain disabled for {transformer_implementation}"

    # Check encoder dropouts remain disabled
    for module in model.transformer.encoder.modules():
        if isinstance(module, nn.Dropout):
            assert (
                not module.training
            ), f"Encoder dropouts should remain disabled for {transformer_implementation}"
