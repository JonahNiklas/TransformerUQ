import os
from typing import List, Tuple
import torch
from torch import nn

from uq.acquisition_func import AcquisitionFunction, BeamScore, BLEUVariance, VR_mpnet_base_cosine, VR_mpnet_base_matrix_norm, VR_mpnet_dot
from beam_search import beam_search_unbatched, beam_search_batched # | not using beam search yet
from data_processing.dataloader import get_data_loader
from hyperparameters import hyperparameters 
from models.transformer_model import TransformerModel
from uq.plot_uq import plot_data_retained_curve, plot_uq_histogram_and_roc
from utils.checkpoints import load_checkpoint
from uq.validate_uq import validate_uq
from data_processing.vocab import load_vocab, output_to_text
from constants import constants
import wandb


def main() -> None:
    # Load shared vocabulary
    run_id="7sy5cau3"
    run_name="Bayesformer"
    checkpoint = "checkpoints/checkpoint-300000b.pth"
    # wandb.restore(checkpoint, run_path=f"sondresorbye-magson/TransformerUQ/{run_id}")  # type: ignore
    src_vocab = load_vocab(constants.file_paths.src_vocab)
    tgt_vocab = load_vocab(constants.file_paths.tgt_vocab)
    print(f"Shared vocab size: {len(src_vocab)}")
    device = hyperparameters.device
    print(f"Device: {device}")

    # Initialize the model with shared vocab size
    model: TransformerModel = TransformerModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
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
        checkpoint,
        remove_orig_prefix=not torch.cuda.is_available()
    )

    # Set up the test data loader with the shared vocabulary
    test_loader = get_data_loader(
        src_file="local/data/test/bpe_test.de",
        tgt_file="local/data/test/bpe_test.en",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=hyperparameters.training.batch_size,# // hyperparameters.beam_search.beam_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    test_ood_loader = get_data_loader(
        src_file="local/data/test_ood/bpe_test_ood.nl",
        tgt_file="local/data/test_ood/bpe_test_ood.en",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=hyperparameters.training.batch_size,# // hyperparameters.beam_search.beam_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )
    
    bleu, avg_uq, hyp_ref_uq_pair = load_or_validate(
        model=model,
        loader=test_loader,
        aq_func=BLEUVariance(),
        filename="validation_cache",
        run_id=run_id
    )

    bleu_bs, avg_uq_bs, hyp_ref_uq_pair_bs = load_or_validate(
        model=model,
        loader=test_loader,
        aq_func=[BeamScore()],
        filename="validation_cache_bs",
        run_id=run_id
    )

    bleu_vr_mpnet, avg_uq_vr_mpnet, hyp_ref_uq_pair_vr_mpnet = load_or_validate(
        model=model,
        loader=test_loader,
        aq_func=VR_mpnet_base_cosine(),
        filename="validation_cache_vr_mpnet",
        run_id=run_id
    )

    bleu_vr_matrix_norm, avg_uq_vr_matrix_norm, hyp_ref_uq_pair_vr_matrix_norm = load_or_validate(
        model=model,
        loader=test_loader,
        aq_func=VR_mpnet_base_matrix_norm(),
        filename="validation_cache_vr_matrix_norm",
        run_id=run_id
    )

    bleu_vr_dot, avg_uq_vr_dot, hyp_ref_uq_pair_vr_dot = load_or_validate(
        model=model,
        loader=test_loader,
        aq_func=VR_mpnet_dot(),
        filename="validation_cache_vr_dot",
        run_id=run_id
    )

    bleu_ood, avg_uq_ood, hyp_ref_uq_pair_ood = load_or_validate(
        model=model,
        loader=test_ood_loader,
        aq_func=[BLEUVariance(), VR_mpnet_base_cosine(), VR_mpnet_base_matrix_norm(),VR_mpnet_dot()],
        filename="validation_cache_ood",
        run_id=run_id
    )

    bleu_ood_bs, avg_uq_ood_bs, hyp_ref_uq_pair_ood_bs = load_or_validate(
        model=model,
        loader=test_ood_loader,
        aq_func=[BeamScore()],
        filename="validation_cache_ood_bs",
        run_id=run_id
    )
    
    bleu_ood_vr_mpnet, avg_uq_ood_vr_mpnet, hyp_ref_uq_pair_ood_vr_mpnet = load_or_validate(
        model=model,
        loader=test_ood_loader,
        aq_func=VR_mpnet_base_cosine(),
        filename="validation_cache_ood_vr_mpnet",
        run_id=run_id
    )

    bleu_ood_vr_matrix_norm, avg_uq_ood_vr_matrix_norm, hyp_ref_uq_pair_ood_vr_matrix_norm = load_or_validate(
        model=model,
        loader=test_ood_loader,
        aq_func=VR_mpnet_base_matrix_norm(),
        filename="validation_cache_ood_vr_matrix_norm",
        run_id=run_id
    )

    bleu_ood_vr_dot, avg_uq_ood_vr_dot, hyp_ref_uq_pair_ood_vr_dot = load_or_validate(
        model=model,
        loader=test_ood_loader,
        aq_func=VR_mpnet_dot(),
        filename="validation_cache_ood_vr_dot",
        run_id=run_id
    )


    print(f"BLEU Score on test_set: {bleu}")
    print(f"Average UQ on test_set: {avg_uq}")

    print(f"BLEU Score on test_set_bs: {bleu_bs}")
    print(f"Average UQ on test_set_bs: {avg_uq_bs}")
    
    print(f"BLEU Score on test_ood: {bleu_ood}")
    print(f"Average UQ on test_ood: {avg_uq_ood}")

    print(f"BLEU Score on test_ood_bs: {bleu_ood_bs}")
    print(f"Average UQ on test_ood_bs: {avg_uq_ood_bs}")

    os.makedirs("local/results", exist_ok=True)

    plot_data_retained_curve(
        [hyp_ref_uq_pair, hyp_ref_uq_pair_bs, hyp_ref_uq_pair_vr_mpnet, hyp_ref_uq_pair_vr_matrix_norm, hyp_ref_uq_pair_vr_dot],
        methods=["BLUEvar", "BeamScore", "VR_mpnet", "VR_matrix_norm", "VR_dot"],
        save_path=f"local/results/{run_id}/hypotheses_uq_pairs.svg",
        run_name=run_name
    )


    plot_data_retained_curve(
        [hyp_ref_uq_pair_ood, hyp_ref_uq_pair_ood_bs, hyp_ref_uq_pair_ood_vr_mpnet, hyp_ref_uq_pair_ood_vr_matrix_norm, hyp_ref_uq_pair_ood_vr_dot],
        methods=["BLUEvar", "BeamScore", "VR_mpnet", "VR_matrix_norm", "VR_dot"],
        save_path=f"local/results/{run_id}/hypotheses_uq_pairs_ood.svg",
        run_name=run_name
    )

    plot_uq_histogram_and_roc(hyp_ref_uq_pair, hyp_ref_uq_pair_ood, method="BLUEvar", save_path=f"local/results/{run_id}/uq_histogram_bluevar.svg",run_name=run_name)
    plot_uq_histogram_and_roc(hyp_ref_uq_pair_bs, hyp_ref_uq_pair_ood_bs, method="BeamScore", save_path=f"local/results/{run_id}/uq_histogram_bs.svg",run_name=run_name)
    plot_uq_histogram_and_roc(hyp_ref_uq_pair_vr_mpnet, hyp_ref_uq_pair_ood_vr_mpnet, method="VR_mpnet", save_path=f"local/results/{run_id}/uq_histogram_vr_mpnet.svg",run_name=run_name)
    plot_uq_histogram_and_roc(hyp_ref_uq_pair_vr_matrix_norm, hyp_ref_uq_pair_ood_vr_matrix_norm, method="VR_matrix_norm", save_path=f"local/results/{run_id}/uq_histogram_vr_matrix_norm.svg",run_name=run_name)
    plot_uq_histogram_and_roc(hyp_ref_uq_pair_vr_dot, hyp_ref_uq_pair_ood_vr_dot, method="VR_dot", save_path=f"local/results/{run_id}/uq_histogram_vr_dot.svg",run_name=run_name)

# Validate the model and calculate BLEU score
def load_or_validate(
    model: TransformerModel,
    loader: torch.utils.data.DataLoader,
    aq_func: AcquisitionFunction,
    filename: str,
    run_id: str
) -> Tuple[float, float, List[Tuple[str, str, float]]]:
    cache_file = f"local/results/{run_id}/{filename}.pth"
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}...")
        cache = torch.load(cache_file)
        bleu = cache["bleu"]
        avg_uq = cache["avg_uq"]
        hyp_ref_uq_pair = cache["hyp_ref_uq_pair"]
    else:
        bleu, avg_uq, hyp_ref_uq_pair = validate_uq(model, loader, aq_func=aq_func, num_batches_to_validate_on=None)
        os.makedirs(f"local/results/{run_id}", exist_ok=True)
        torch.save({"bleu": bleu, "avg_uq": avg_uq, "hyp_ref_uq_pair": hyp_ref_uq_pair}, f"local/results/{run_id}/{filename}.pth")
        print(f"Cached validation results in local/results/{run_id}/{filename}.pth")
    return bleu, avg_uq, hyp_ref_uq_pair
    
if __name__ == "__main__":
    main()
