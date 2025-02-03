import os
from typing import Tuple
from sympy import plot
import torch
from torch import nn

import wandb
from uq.acquisition_func import BeamScore, BLEUVariance
from beam_search import beam_search_unbatched, beam_search_batched
from data_processing.dataloader import get_data_loader
from hyperparameters import hyperparameters
from models.transformer_pytorch import TransformerPyTorch
from uq.plot_uq import plot_data_retained_curve, plot_uq_histogram
from utils.checkpoints import load_checkpoint
from uq.validate_uq import validate_uq
from data_processing.vocab import load_vocab, output_to_text
from constants import constants

def main() -> None:
    # Load shared vocabulary
    # wandb.restore("checkpoints/checkpoint-175000.pth", run_path="sondresorbye-magson/TransformerUQ/54inz442")  # type: ignore
    shared_vocab = load_vocab(constants.file_paths.vocab)
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
        batch_size=hyperparameters.training.batch_size // hyperparameters.beam_search.beam_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    test_ood_loader = get_data_loader(
        src_file="local/data/test_ood/bpe_test_ood.nl",
        tgt_file="local/data/test_ood/bpe_test_ood.en",
        vocab=shared_vocab,
        batch_size=hyperparameters.training.batch_size // hyperparameters.beam_search.beam_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )
    # Validate the model and calculate BLEU score
    cache_file = "local/results/validation_cache.pth"
    if os.path.exists(cache_file):
        print("Loading cached validation results...")
        cache = torch.load(cache_file)
        bleu = cache["bleu"]
        avg_uq = cache["avg_uq"]
        hyp_ref_uq_pair = cache["hyp_ref_uq_pair"]
    else:
        bleu, avg_uq, hyp_ref_uq_pair = validate_uq(model, test_loader, aq_func=BLEUVariance(), num_batches_to_validate_on=None)
        cache_validation_results(bleu, avg_uq, hyp_ref_uq_pair, "validation_cache")
    


    cache_file = "local/results/validation_cache_ood.pth"
    if os.path.exists(cache_file):
        print("Loading cached validation ood results...")
        cache = torch.load(cache_file)
        bleu_ood = cache["bleu"]
        avg_uq_ood = cache["avg_uq"]
        hyp_ref_uq_pair_ood = cache["hyp_ref_uq_pair"]
    else:
        bleu, avg_uq, hyp_ref_uq_pair = validate_uq(model, test_loader, aq_func=BLEUVariance(), num_batches_to_validate_on=None)
        cache_validation_results(bleu, avg_uq, hyp_ref_uq_pair, "validation_cache_ood")
    bleu_ood, avg_uq_ood,hyp_ref_uq_pair_ood = validate_uq(model, test_ood_loader, aq_func=BLEUVariance(), num_batches_to_validate_on=None)
    
    print(f"BLEU Score on test_set: {bleu}")
    print(f"Average UQ on test_set: {avg_uq}")
    
    print(f"BLEU Score on test_ood: {bleu_ood}")
    print(f"Average UQ on test_ood: {avg_uq_ood}")

    os.makedirs("local/results", exist_ok=True)

    plot_data_retained_curve(hyp_ref_uq_pair, "local/results/hypotheses_uq_pairs.png")
    plot_data_retained_curve(hyp_ref_uq_pair_ood, "local/results/hypotheses_uq_pairs_ood.png")

    plot_uq_histogram(hyp_ref_uq_pair,hyp_ref_uq_pair_ood, "local/results/uq_histogram.png")


def cache_validation_results(bleu: float, avg_uq: float, hyp_ref_uq_pair: list[Tuple[str,str, torch.Tensor]], filename: str) -> None:
    os.makedirs("local/results", exist_ok=True)
    torch.save({"bleu": bleu, "avg_uq": avg_uq, "hyp_ref_uq_pair": hyp_ref_uq_pair}, f"local/results/{filename}.pth")
if __name__ == "__main__":
    main()
