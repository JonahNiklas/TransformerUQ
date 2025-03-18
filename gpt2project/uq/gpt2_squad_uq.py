from __future__ import annotations
import os
from typing import List, Tuple
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.bayesformer_gpt import BayesformerGPT
from gpt2project.data_processing.load_squad import (
    create_squad_prompt_batched,
    get_squad_dataloader,
)
from gpt2project.gpt2model import GPT
from gpt2project.uq.gpt_aq_funcs import (
    AcquisitionFunctionGPT,
    BLEUVar,
    BeamScore,
    mpnet_cosine,
)
from gpt2project.uq.calc_plot_data import calc_retention_curve
from gpt2project.utils.benchmark_eval_funcs import MultipleTargetEval, TargetUsageEval
from gpt2project.utils.checkpoint import get_model_from_wandb_checkpoint
from hyperparameters import hyperparameters
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.search_methods_gpt import (
    GPT_search_method,
    greedy_search_gpt,
    topk_sampling_gpt,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_squad(
    model: GPT | BayesformerGPT,
    benchmark_name: str,
    model_name: str,
    tokenizer: tiktoken.Encoding,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunctionGPT],
    run_name: str,
    enable_mcdo: bool,
    search_method: GPT_search_method,
) -> Tuple[List[List[str]], List[List[str]], torch.Tensor]:
    folder = f"local/gpt-cache/{benchmark_name}/{model_name}/mcdo{enable_mcdo}/{search_method.__name__}"
    os.makedirs(folder, exist_ok=True)
    filename = (
        folder
        + f"/{benchmark_name}_outputs_{model_name}_{run_name}_n{n_batch_to_validate}.pt"
    )
    if os.path.exists(filename):
        all_output_texts, all_targets, all_uqs = torch.load(filename)
        logger.info("Loaded inference results from file.")
        return all_output_texts, all_targets, all_uqs

    logger.info("Generating inference results...")
    all_output_texts = [[] for _ in range(len(aq_funcs))]
    all_targets = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)

    dataloader = get_squad_dataloader()
    for i, (context, question, targets) in tqdm(
        enumerate(dataloader),
        desc="Running squad validation",
        total=len(dataloader) if n_batch_to_validate == -1 else n_batch_to_validate,
    ):
        if i == n_batch_to_validate:
            break
        tokens = tokenizer.encode_batch(create_squad_prompt_batched(context, question))
        encoding_tensors = torch.tensor(tokens).to(hyperparameters.device)
        output_texts, uq = generate_autoregressivly_gpt2_with_uq(
            model,
            tokenizer,
            encoding_tensors,
            search_method,
            enable_mcdo,
            break_on_newline=False,
            aq_funcs=aq_funcs,
        )
        for aq in range(len(aq_funcs)):
            all_output_texts[aq].extend(output_texts[aq])
        all_targets.extend(targets)
        all_uqs = torch.cat((all_uqs, uq), dim=0)

    os.makedirs("local/gpt-results/squad", exist_ok=True)
    torch.save((all_output_texts, all_targets, all_uqs), filename)
    logger.info("Saved inference results to file:", filename)
    return all_output_texts, all_targets, all_uqs


def get_squad_run(
    model: GPT | BayesformerGPT,
    tokenizer: tiktoken.Encoding,
    run_name: str,
    enable_mcdo: bool,
    search_method: GPT_search_method,
    eval_function: MultipleTargetEval,
    n_batch_to_validate: int = 1000,
) -> None:
    benchmark_name = "squad"
    model_name = "GPT" if model.config.transformer_impl == "transformer" else "BayesGPT"

    aq_funcs = [BeamScore(), BLEUVar(), mpnet_cosine()]

    all_outputs, all_targets, all_uqs = eval_squad(
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
    calc_retention_curve(
        all_outputs,
        all_targets,
        all_uqs,
        eval_function=eval_function,
        aq_func_names=[aq_func.__class__.__name__ for aq_func in aq_funcs],
        stepsize=stepsize,
        enable_mcdo=enable_mcdo,
        search_method_type=search_method.__name__,
        benchmark_name=benchmark_name,
        model_name=model_name,
        filename=f"plot_data_{run_name}_{eval_function.__class__.__name__}_n{n_batch_to_validate}_step{stepsize}.pt",
    )


if __name__ == "__main__":
    # Load the GPT-2 moloadel and tokenizer
    model_name = "gpt2"

    tokenizer = tiktoken.get_encoding(model_name)
    model = get_model_from_wandb_checkpoint(
        wandb_artifact_path="sondresorbye-magson/GPT2Project/model-checkpoint-76291:v1",
        checkpoint_name="model_transformer_76291.pt",
        remove_orig_prefix=True,
    )
    model.to(hyperparameters.device)
    model.eval()
    run_name = "GPT"

    get_squad_run(
        model,
        tokenizer,
        run_name,
        enable_mcdo=True,
        search_method=greedy_search_gpt,
        eval_function=TargetUsageEval(),
        n_batch_to_validate=1000,
    )
