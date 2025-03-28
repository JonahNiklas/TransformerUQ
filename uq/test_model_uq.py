from gc import enable
import os
from typing import List, Literal, Tuple
from pdb import run
from tabnanny import check
from typing import List
import torch

from uq.acquisition_func import (
    AcquisitionFunction,
    BeamScore,
    BLEUVar,
    mpnet_cosine,
    mpnet_norm,
    mpnet_dot,
    roberta_cosine,
)
from data_processing.dataloader import get_data_loader
from hyperparameters import hyperparameters
from models.transformer_model import TransformerModel
from uq.plot_uq import (
    calc_ret_curve_plot_data_wmt,
    plot_data_retained_curve,
    plot_uq_histogram_and_roc,
)
from utils.checkpoints import load_checkpoint
from uq.validate_uq import ValidationResult, validate_uq
from data_processing.vocab import load_vocab
from constants import constants
from utils.general_plotter import plot_ret_curve


def main() -> None:
    # Load shared vocabulary
    checkpoint = "local/checkpoints/iwslt/iwslt-transformer-checkpoint-500000.pth"
    transformer_impl: Literal["bayesformer", "pytorch", "own"] = "own"
    hyperparameters.transformer.transformer_implementation = transformer_impl
    run_id = "ot0v1maq"
    run_name = "iwslt_transformer"

    src_vocab = load_vocab(constants.file_paths.src_vocab)
    tgt_vocab = load_vocab(constants.file_paths.tgt_vocab)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
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
    load_checkpoint(model, optimizer, checkpoint)

    # Set up the test data loader with the shared vocabulary
    test_loader = get_data_loader(
        src_file="local/data/test/bpe_test.de",
        tgt_file="local/data/test/bpe_test.en",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=hyperparameters.training.batch_size,  # // hyperparameters.beam_search.beam_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    test_ood_loader = get_data_loader(
        src_file="local/data/test_ood/bpe_test_ood.nl",
        tgt_file="local/data/test_ood/bpe_test_ood.en",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        batch_size=hyperparameters.training.batch_size,  # // hyperparameters.beam_search.beam_size,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    # Define the acquisition functions used for validation
    aq_funcs: List[AcquisitionFunction] = [
        BeamScore(),
        BLEUVar(),
        mpnet_cosine(),
        mpnet_norm(),
    ]

    # The different validation configs to run
    val_spec = [
        {
            "search_method": "greedy",
            "dropout": True,
        },
        {
            "search_method": "beam",
            "dropout": True,
        },
        {
            "search_method": "sample",
            "dropout": True,
        },
        {
            "search_method": "sample",
            "dropout": False,
        },
    ]

    for spec in val_spec:
        search_method: str = str(spec["search_method"])
        dropout: bool = bool(spec["dropout"])
        filename = f"{run_name}_{search_method}_{dropout}"
        os.makedirs(
            f"local/results/{run_id}/{search_method}/dropout{dropout}", exist_ok=True
        )
        print(f"Validating model with {search_method} search, dropout={dropout}")

        # run validate_uq or load cached results for in-distribution data
        validation_results_id = load_or_validate(
            model,
            test_loader,
            search_method,
            aq_funcs,
            dropout,
            filename + "_id",
            run_id,
        )

        # run validate_uq or load cached results for ood data
        validation_results_ood = load_or_validate(
            model,
            test_ood_loader,
            search_method,
            aq_funcs,
            dropout,
            filename + "_ood",
            run_id,
        )

        for validation_result, benchmark_name, save_path in [
            (
                validation_results_id,
                "german_wmt_id",
                f"local/translation-results/{run_name}/german_wmt_id_{hyperparameters.transformer.transformer_implementation}_dropout{dropout}_{search_method}.json",
            ),
            (
                validation_results_ood,
                "dutch_wmt_ood",
                f"local/translation-results/{run_name}/dutch_wmt_ood_{hyperparameters.transformer.transformer_implementation}_dropout{dropout}_{search_method}.json",
            ),
        ]:
            calc_ret_curve_plot_data_wmt(
                validation_result,
                aq_func_names=[aq_func.__class__.__name__ for aq_func in aq_funcs],
                model_name=hyperparameters.transformer.transformer_implementation,
                eval_method="BLEU",
                benchmark_name=benchmark_name,
                save_path=save_path,
            )

        get_run_curves_and_histograms(
            validation_results_id,
            validation_results_ood,
            aq_funcs,
            run_id,
            run_name,
            search_method,
            dropout,
        )


def get_run_curves_and_histograms(
    validation_results_id: List[ValidationResult],
    validation_results_ood: List[ValidationResult],
    aq_funcs: List[AcquisitionFunction],
    run_id: str,
    run_name: str,
    search_method: str,
    dropout: bool,
) -> None:
    plot_data_retained_curve(
        validation_results_id,
        methods=[aq_func.__class__.__name__ for aq_func in aq_funcs],
        save_path=f"local/results/{run_id}/{search_method}/dropout{dropout}/{run_name}_{search_method}_drop{dropout}_retcurve_id.svg",
        run_name=run_name,
    )

    plot_data_retained_curve(
        validation_results_ood,
        methods=[aq_func.__class__.__name__ for aq_func in aq_funcs],
        save_path=f"local/results/{run_id}/{search_method}/dropout{dropout}/{run_name}_{search_method}_drop{dropout}_retcurve_ood.svg",
        run_name=run_name,
    )

    for i, aq_func in enumerate(aq_funcs):
        plot_uq_histogram_and_roc(
            validation_results_id[i],
            validation_results_ood[i],
            aq_func.__class__.__name__,
            f"local/results/{run_id}/{search_method}/dropout{dropout}/{run_name}_{search_method}_drop{dropout}_hist_{aq_func.__class__.__name__}.svg",
            run_name,
        )


def load_or_validate(
    model: TransformerModel,
    loader: torch.utils.data.DataLoader,
    sample_beam_greed: str,
    aq_funcs: List[AcquisitionFunction],
    enable_dropout: bool,
    filename: str,
    run_id: str,
) -> List[ValidationResult]:
    cache_file = f"local/results/{run_id}/{filename}.pth"
    validation_results: List[ValidationResult] = []
    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}...")
        cache = torch.load(cache_file)
        validation_results = cache
    else:
        validation_results = validate_uq(
            model,
            loader,
            sample_beam_greed,
            aq_funcs,
            enable_dropout,
            num_batches_to_validate_on=None,
        )
        os.makedirs(f"local/results/{run_id}", exist_ok=True)
        torch.save(validation_results, cache_file)
        print(f"Cached validation results in local/results/{run_id}/{filename}.pth")
    return validation_results


if __name__ == "__main__":
    main()
