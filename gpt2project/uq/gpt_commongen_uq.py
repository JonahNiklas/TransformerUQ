from __future__ import annotations
import os
from typing import List, Tuple
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.bayesformer_gpt import BayesformerGPT
from gpt2project.data_processing.load_commongen import get_common_gen_dataloader
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import (
    GPT_search_method,
    greedy_search_gpt,
    topk_sampling_gpt,
)
from gpt2project.utils.checkpoint import get_model_from_wandb_checkpoint
from utils.general_plotter import get_gpt_cache_filename, plot_ret_curve
from gpt2project.uq.gpt_aq_funcs import (
    BALD,
    AcquisitionFunctionGPT,
    BLEUVar,
    BeamScore,
    mpnet_cosine,
)
from gpt2project.uq.calc_plot_data import calc_retention_curve_commongen
from gpt2project.utils.benchmark_eval_funcs import (
    BLEU_eval,
    ConceptUsageEval,
    KeywordEval,
    MultipleTargetEval,
)
from hyperparameters import hyperparameters

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_commongen(
    model: GPT | BayesformerGPT,
    benchmark_name: str,
    model_name: str,
    tokenizer: tiktoken.Encoding,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunctionGPT],
    run_name: str,
    enable_mcdo: bool,
    search_method: GPT_search_method,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], torch.Tensor]:
    folder = f"local/gpt-cache/{benchmark_name}/{model_name}/mcdo{enable_mcdo}/{search_method.__name__}"
    os.makedirs(folder, exist_ok=True)
    filename = (
        folder
        + f"/{benchmark_name}_outputs_{model_name}_{run_name}_n{n_batch_to_validate}.pt"
    )

    all_outputs: List[List[str]] = [[] for _ in range(len(aq_funcs))]
    all_concepts: List[List[str]] = []
    all_targets: List[List[str]] = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)

    if os.path.exists(filename):
        all_outputs, all_concepts, all_targets, all_uqs = torch.load(filename)
        logger.info("Loaded inference results from file.")
        return all_outputs, all_concepts, all_targets, all_uqs

    logger.info("Generating inference results...")

    dataloader = get_common_gen_dataloader()
    for i, (input_texts, concepts, targets, encoding_tensors) in tqdm(
        enumerate(dataloader),
        desc="Running commongen validation",
        total=len(dataloader) if n_batch_to_validate == -1 else n_batch_to_validate,
    ):
        output_texts, uq = generate_autoregressivly_gpt2_with_uq(
            model,
            tokenizer,
            encoding_tensors,
            search_method,
            enable_mcdo,
            break_on_newline=True,
            aq_funcs=aq_funcs,
            max_tokens=20,
        )
        for aq in range(len(aq_funcs)):
            all_outputs[aq].extend(output_texts[aq])
        all_concepts.extend(concepts)
        all_targets.extend(targets)
        all_uqs = torch.cat((all_uqs, uq), dim=0)
        if i == n_batch_to_validate:
            break

    torch.save((all_outputs, all_concepts, all_targets, all_uqs), filename)
    logger.info("Saved inference results to file: %s", filename)
    return all_outputs, all_concepts, all_targets, all_uqs


def get_commongen_run(
    model: GPT | BayesformerGPT,
    tokenizer: tiktoken.Encoding,
    run_name: str,
    enable_mcdo: bool,
    search_method: GPT_search_method,
    eval_function: MultipleTargetEval | KeywordEval,
    n_batch_to_validate: int = -1,
) -> None:
    benchmark_name = "commongen"
    model_name = "GPT" if model.config.transformer_impl == "transformer" else "BayesGPT"

    aq_funcs = [BeamScore(), BALD(), BLEUVar()]

    all_outputs, all_concepts, all_targets, all_uqs = eval_commongen(
        model,
        benchmark_name,
        model_name,
        tokenizer,
        n_batch_to_validate,
        aq_funcs,
        run_name=run_name,
        enable_mcdo=enable_mcdo,
        search_method=search_method,
    )
    stepsize = 25

    calc_retention_curve_commongen(
        all_outputs,
        all_concepts,
        all_targets,
        all_uqs,
        eval_function,
        aq_func_names=[aq_func.__class__.__name__ for aq_func in aq_funcs],
        model_name=model_name,
        enable_mcdo=enable_mcdo,
        search_method_type=search_method.__name__,
        benchmark_name=benchmark_name,
        stepsize=stepsize,
        filename=get_gpt_cache_filename(
            run_name,
            eval_function.__class__.__name__,
            n_batch_to_validate,
            stepsize,
        ),
    )


if __name__ == "__main__":
    # Load the  GPT-2 model and tokenizer
    os.makedirs("local/gpt-results/commongen", exist_ok=True)
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = get_model_from_wandb_checkpoint(
        wandb_artifact_path="sondresorbye-magson/GPT2Project/model-checkpoint-76291:v1",
        checkpoint_name="model_transformer_76291.pt",
    )
    model.to(hyperparameters.device)
    model.eval()
    enable_mcdo = True
    search_method = greedy_search_gpt

    run_name = "GPT"

    get_commongen_run(
        model,
        tokenizer,
        run_name="run1",
        enable_mcdo=True,
        search_method=greedy_search_gpt,
        eval_function=ConceptUsageEval(),
    )
