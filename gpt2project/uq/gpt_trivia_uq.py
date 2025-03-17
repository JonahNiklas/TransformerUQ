import os
from typing import List, Tuple
from regex import F
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.bayesformer_gpt import BayesformerGPT
from gpt2project.data_processing.load_triviaQA import get_triviaqa_dataloader
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import (
    GPT_search_method,
    greedy_search_gpt,
    topk_sampling_gpt,
)
from gpt2project.uq.gpt_aq_funcs import AcquisitionFunctionGPT, BLEUVar, BeamScore
from gpt2project.uq.calc_plot_data import calc_retention_curve
from gpt2project.utils.benchmark_eval_funcs import TargetUsageEval
from hyperparameters import hyperparameters

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_triviaqa(
    model: GPT | BayesformerGPT,
    benchmark_name: str,
    model_name: str,
    tokenizer: tiktoken.Encoding,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunctionGPT],
    shuffle: bool,
    run_name: str,
    enable_mcdo: bool,
    search_method: GPT_search_method,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], torch.Tensor]:
    folder = f"local/gpt-cache/{benchmark_name}/{model_name}/mcdo{enable_mcdo}"
    os.makedirs(folder, exist_ok=True)
    filename = (
        folder
        + f"/{benchmark_name}_outputs_{model_name}_{run_name}_n{n_batch_to_validate}_shuffle-{shuffle}.pt"
    )
    all_outputs: List[List[str]] = [[] for _ in range(len(aq_funcs))]
    all_questions: List[List[str]] = []
    all_targets: List[List[str]] = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)

    if os.path.exists(filename):
        all_outputs, all_questions, all_targets, all_uqs = torch.load(filename)
        logger.info("Loaded inference results from file.")
        return all_outputs, all_questions, all_targets, all_uqs

    logger.info("Generating inference results...")

    dataloader = get_triviaqa_dataloader(shuffle=shuffle)
    for i, (input_texts, questions, targets, encoding_tensors) in tqdm(
        enumerate(dataloader),
        desc="Running triviaqa validation",
        total=len(dataloader) if n_batch_to_validate == -1 else n_batch_to_validate,
    ):
        output_texts, uq = generate_autoregressivly_gpt2_with_uq(
            model,
            tokenizer,
            encoding_tensors,
            search_method,
            enable_mcdo=enable_mcdo,
            break_on_newline=True,
            aq_funcs=aq_funcs,
            max_tokens=20,
        )
        for aq in range(len(aq_funcs)):
            all_outputs[aq].extend(output_texts[aq])
        all_questions.extend(questions)
        all_targets.extend(targets)
        all_uqs = torch.cat((all_uqs, uq), dim=0)
        if i == n_batch_to_validate:
            break

    torch.save((all_outputs, all_questions, all_targets, all_uqs), filename)
    logger.info("Saved inference results to file: %s", filename)
    return all_outputs, all_questions, all_targets, all_uqs


def get_triviaqa_run(
    model: GPT | BayesformerGPT,
    tokenizer: tiktoken.Encoding,
    run_name: str,
    enable_mcdo: bool,
    search_method: GPT_search_method,
    n_batch_to_validate: int = -1,
):
    benchmark_name = "triviaqa"
    model_name = "GPT" if model.config.transformer_impl == "transformer" else "BayesGPT"

    n_batch_to_validate = -1

    aq_funcs = [BeamScore(), BLEUVar()]
    eval_function_triviaqa = TargetUsageEval()

    all_outputs, all_questions, all_targets, all_uqs = eval_triviaqa(
        model,
        benchmark_name,
        model_name,
        tokenizer,
        n_batch_to_validate,
        aq_funcs,
        shuffle=False,
        run_name=run_name,
        enable_mcdo=enable_mcdo,
        search_method=search_method,
    )

    stepsize = 25
    calc_retention_curve(
        all_outputs,
        all_targets,
        all_uqs,
        eval_function=eval_function_triviaqa,
        aq_func_names=[aq_func.__class__.__name__ for aq_func in aq_funcs],
        stepsize=stepsize,
        benchmark_name=benchmark_name,
        model_name=model_name,
        enable_mcdo=enable_mcdo,
        search_method=search_method.__name__,
        filename=f"plot_data_{run_name}_{eval_function_triviaqa.__class__.__name__}_n{n_batch_to_validate}_step{stepsize}.pt",
    )


if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    os.makedirs("local/gpt-results/triviaqa", exist_ok=True)
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name)
    model.to(hyperparameters.device)
    model.eval()
    enable_mcdo = True
    search_method = greedy_search_gpt

    run_name = "BayesGPT"

    get_triviaqa_run(
        model,
        tokenizer,
        run_name=run_name,
        enable_mcdo=enable_mcdo,
        search_method=search_method,
    )
