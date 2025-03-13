import os
from typing import List, Tuple
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.data_processing.load_commongen import get_common_gen_dataloader
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import greedy_search_gpt, topk_sampling_gpt
from utils.general_plotter import plot_ret_curve
from gpt2project.uq.gpt_aq_funcs import BALD, AcquisitionFunctionGPT, BLEUVar, BeamScore
from gpt2project.uq.calc_plot_data import calc_retention_curve_cg
from gpt2project.utils.benchmark_eval_funcs import ConceptUsageEval
from hyperparameters import hyperparameters

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_commongen(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunctionGPT],
    shuffle: bool,
    run_name: str,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], torch.Tensor]:
    filename = f"local/gpt-results/commongen/commongen_outputs_{run_name}_b{batch_size}_n{n_batch_to_validate}_shuffle-{shuffle}.pt"
    all_outputs: List[List[str]] = [[] for _ in range(len(aq_funcs))]
    all_concepts: List[List[str]] = []
    all_targets: List[List[str]] = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)

    if os.path.exists(filename):
        all_outputs, all_concepts, all_targets, all_uqs = torch.load(filename)
        logger.info("Loaded inference results from file.")
        return all_outputs, all_concepts, all_targets, all_uqs

    logger.info("Generating inference results...")

    dataloader = get_common_gen_dataloader(batch_size, shuffle=shuffle)
    for i, (input_texts, concepts, targets, encoding_tensors) in tqdm(
        enumerate(dataloader),
        desc="Running commongen validation",
        total=len(dataloader) if n_batch_to_validate == -1 else n_batch_to_validate,
    ):
        output_texts, uq = generate_autoregressivly_gpt2_with_uq(
            model,
            tokenizer,
            encoding_tensors,
            greedy_search_gpt,
            enable_mcdo=True,
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


if __name__ == "__main__":
    # Load the  GPT-2 model and tokenizer
    os.makedirs("local/gpt-results/commongen", exist_ok=True)
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name)
    model.to(hyperparameters.device)
    model.eval()

    run_name = "gpt2-test-bald"

    n_batch_to_validate = -1
    batch_size = 1

    aq_funcs = [BeamScore(), BALD(), BLEUVar()]
    eval_function_commongen = ConceptUsageEval()

    all_outputs, all_concepts, all_targets, all_uqs = eval_commongen(
        model,
        tokenizer,
        batch_size,
        n_batch_to_validate,
        aq_funcs,
        shuffle=False,
        run_name=run_name,
    )
    stepsize = 25

    calc_retention_curve_cg(
        all_outputs,
        all_concepts,
        all_targets,
        all_uqs,
        eval_function_commongen,
        aq_func_names=[aq_func.__class__.__name__ for aq_func in aq_funcs],
        model_name=model_name,
        benchmark_name="commongen",
        stepsize=stepsize,
        folder="local/gpt-results/commongen",
        filename=f"plot_data_{run_name}_{eval_function_commongen.__class__.__name__}_b{batch_size}_n{n_batch_to_validate}_step{stepsize}.pt",
    )
    plot_ret_curve(
        plot_data_paths=[
            f"local/gpt-results/commongen/plot_data_{run_name}_{eval_function_commongen.__class__.__name__}_b{batch_size}_n{n_batch_to_validate}_step{stepsize}.pt",
        ],
        title="Commongen",
        save_filepath=f"local/gpt-results/commongen/commongen_ret_curve_{run_name}_{eval_function_commongen.__class__.__name__}_b{batch_size}_n{n_batch_to_validate}_step{stepsize}.svg",
    )
