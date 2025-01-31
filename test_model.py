import torch
from torch import nn

import wandb
from acquisition_func import BeamScore,BLEUVariance
from dataloader import get_data_loader
from hyperparameters import hyperparameters
from models.transformer_pytorch import TransformerPyTorch
from train import load_checkpoint
from validate import validate
from vocab import load_vocab

def main() -> None:
    # Load shared vocabulary
    # wandb.restore("checkpoints/checkpoint-175000.pth", run_path="sondresorbye-magson/TransformerUQ/54inz442")  # type: ignore
    shared_vocab = load_vocab("local/vocab_shared.pkl")
    print(f"Shared vocab size: {len(shared_vocab)}")
    device = hyperparameters.device
    print(f"Device: {device}")

    # Initialize the model with shared vocab size
    model: nn.Module = TransformerPyTorch(
        vocab_size=len(shared_vocab),
        d_model=hyperparameters.transformer.hidden_size,
        num_heads=hyperparameters.transformer.num_heads,
        d_ff=hyperparameters.transformer.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.transformer.num_hidden_layers,
        num_decoder_layers=hyperparameters.transformer.num_hidden_layers,
        dropout=hyperparameters.transformer.dropout,
        max_len=hyperparameters.transformer.max_len,
    ).to(device)

    if torch.cuda.is_available():
        model = torch.compile(model)  # type: ignore
        torch.set_float32_matmul_precision("high")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Load the checkpoint
    load_checkpoint(
        model, 
        optimizer, 
        "checkpoints/checkpoint-175000.pth",
        remove_orig_prefix=not torch.cuda.is_available()
    )

    # Set up the test data loader with the shared vocabulary
    test_loader = get_data_loader(
        src_file="local/data/test/bpe_test.de",
        tgt_file="local/data/test/bpe_test.en",
        vocab=shared_vocab,
        batch_size=hyperparameters.training.batch_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    test_ood_loader = get_data_loader(
        src_file="local/data/test_ood/bpe_test_ood.nl",
        tgt_file="local/data/test_ood/bpe_test_ood.en",
        vocab=shared_vocab,
        batch_size=hyperparameters.training.batch_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )
    # Validate the model and calculate BLEU score
    bleu, avg_uq = validate(model, test_loader, None,aq_func=BLEUVariance())
    print(f"BLEU Score on test_set: {bleu}")
    print(f"Average UQ on test_set: {avg_uq}")
    
    bleu, avg_uq = validate(model, test_ood_loader, None,aq_func=BLEUVariance())
    print(f"BLEU Score on test_ood: {bleu}")
    print(f"Average UQ on test_ood: {avg_uq}")

if __name__ == "__main__":
    main()
