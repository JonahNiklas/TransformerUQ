import os
from typing import List, Tuple
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.data_processing.load_lambada import get_lambada_dataloader
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import greedy_search_gpt, topk_sampling_gpt
from gpt2project.uq.general_plotter import plot_ret_curve
from gpt2project.uq.gpt_aq_funcs import BALD, AcquisitionFunctionGPT, BLEUVar, BeamScore
from gpt2project.uq.plot_uq import calc_retention_curve
from gpt2project.utils.benchmark_eval_funcs import F1Eval, TargetUsageEval
from hyperparameters import hyperparameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def break_on_space_or_punctuation(text: str) -> str:
    text = text.lstrip(".,;!? ")

    for i, char in enumerate(text):
        if char in [" ", ".", ",", ";", "!", "?"]:
            return text[:i]
    return text


def eval_lambada(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunctionGPT],
    shuffle: bool,
    run_name: str,
) -> Tuple[List[List[str]], List[List[str]], torch.Tensor]:
    filename = f"local/gpt-results/lambada/lambada_outputs_{run_name}_b{batch_size}_n{n_batch_to_validate}_shuffle-{shuffle}.pt"
    all_outputs: List[List[str]] = [[] for _ in range(len(aq_funcs))]
    all_targets: List[List[str]] = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)

    if os.path.exists(filename):
        all_outputs, all_targets, all_uqs = torch.load(filename)
        logger.info("Loaded inference results from file.")
        return all_outputs, all_targets, all_uqs

    logger.info("Generating inference results...")

    dataloader = get_lambada_dataloader(batch_size, shuffle=shuffle)
    for i, (input_texts, targets, encoding_tensors) in tqdm(
        enumerate(dataloader),
        desc="Running lambada validation",
        total=len(dataloader) if n_batch_to_validate == -1 else n_batch_to_validate,
    ):
        output_texts, uq = generate_autoregressivly_gpt2_with_uq(
            model,
            tokenizer,
            encoding_tensors,
            topk_sampling_gpt,
            enable_mcdo=False,
            break_on_newline=True,
            aq_funcs=aq_funcs,
            max_tokens=20,
        )
        for aq in range(len(aq_funcs)):
            all_outputs[aq].extend(
                [break_on_space_or_punctuation(text) for text in output_texts[aq]]
            )
        all_targets.extend([targets])
        all_uqs = torch.cat((all_uqs, uq), dim=0)
        if i == n_batch_to_validate:
            break

    torch.save((all_outputs, all_targets, all_uqs), filename)
    logger.info("Saved inference results to file: %s", filename)
    return all_outputs, all_targets, all_uqs


if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    os.makedirs("local/gpt-results/lambada", exist_ok=True)
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name)
    model.to(hyperparameters.device)
    model.eval()

    run_name = "gpt2-testf1"

    n_batch_to_validate = -1
    batch_size = 1

    aq_funcs = [BeamScore(), BLEUVar(), BALD()]
    eval_function_lambada = F1Eval()  # TODO: change to F1Eval()

    all_outputs, all_targets, all_uqs = eval_lambada(
        model,
        tokenizer,
        batch_size,
        n_batch_to_validate,
        aq_funcs,
        shuffle=False,
        run_name=run_name,
    )
    stepsize = 1
    calc_retention_curve(
        all_outputs,
        all_targets,
        all_uqs,
        eval_function_lambada,
        [aq_func.__class__.__name__ for aq_func in aq_funcs],
        stepsize=stepsize,
        benchmark_name="lambada",
        model_name=model_name,
        folder="local/gpt-results/lambada",
        filename=f"plot_data_{run_name}_{eval_function_lambada.__class__.__name__}_b{batch_size}_n{n_batch_to_validate}_step{stepsize}.pt",
    )

    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/lambada/plot_data_{run_name}_{eval_function_lambada.__class__.__name__}_b{batch_size}_n{n_batch_to_validate}_step{stepsize}.pt",
        ],
        title="LAMBADA",
        save_filepath=f"local/gpt-results/lambada/lambada_ret_curve_{run_name}_{eval_function_lambada.__class__.__name__}_b{batch_size}_n{n_batch_to_validate}_step{stepsize}.svg",
    )
