import os
import pickle
from typing import Any, List, Tuple
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.data_processing.load_commongen import get_common_gen_dataloader
from gpt2project.gpt2_commongen import BLEU_eval, CommongenEval, ConceptUsageEval
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import greedy_search_gpt, topk_sampling_gpt
from gpt2project.uq.plot_uq import plot_retention_curve_cg
from gpt2project.utils.decode import decode_token_id_batch
from hyperparameters import hyperparameters
from uq.acquisition_func import AcquisitionFunction, BLEUVar, BeamScore

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evalutate_model_batch_with_uq(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    encoding_tensors: torch.Tensor,
    aq_funcs: List[Any],
) -> Tuple[List[List[str]], torch.Tensor]:
    hypothesis, uq = generate_autoregressivly_gpt2_with_uq(
        model,
        tokenizer,
        encoding_tensors,
        topk_sampling_gpt,
        True,
        aq_funcs,
        max_tokens=20,
    )
    return hypothesis, uq


def load_or_generate_inference_commongen(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunction],
    shuffle: bool,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], torch.Tensor]:
    filename = f"local/gpt-results/commongen/commongen_outputs_{run_name}_b{batch_size}_n{n_batch_to_validate}_shuffle-{shuffle}.pt"
    all_outputs: List[List[str]] = [[] for _ in range(len(aq_funcs))]
    all_concepts: List[List[str]] = []
    all_targets: List[List[str]] = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)

    if os.path.exists(filename):
        all_outputs, all_concepts, all_targets, all_uqs = torch.load(filename)
        print("Loaded inference results from file.")
        return all_outputs, all_concepts, all_targets, all_uqs

    print("Generating inference results...")

    dataloader = get_common_gen_dataloader(batch_size=batch_size, shuffle=shuffle)
    for i, (input_texts, concepts, targets, encoding_tensors) in tqdm(
        enumerate(dataloader),
        desc="Running commongen validation",
        total=len(dataloader),
    ):
        output_texts, uq = evalutate_model_batch_with_uq(
            model=model,
            tokenizer=tokenizer,
            encoding_tensors=encoding_tensors,
            aq_funcs=aq_funcs,
        )
        for aq in range(len(aq_funcs)):
            all_outputs[aq].extend(output_texts[aq])
        all_concepts.extend(concepts)
        all_targets.extend(targets)
        all_uqs = torch.cat((all_uqs, uq), dim=0)
        if i == n_batch_to_validate:
            break

    torch.save((all_outputs, all_concepts, all_targets, all_uqs), filename)
    print("Saved inference results to file:", filename)
    return all_outputs, all_concepts, all_targets, all_uqs


if __name__ == "__main__":
    # Load the  GPT-2 model and tokenizer
    os.makedirs("local/gpt-results/commongen", exist_ok=True)
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name)
    model.to(hyperparameters.device)
    model.eval()

    run_name = "gpt2-pretrained"

    n_batch_to_validate = 10
    batch_size = 1

    aq_funcs = [BeamScore(), BLEUVar()]
    eval_function_commongen = ConceptUsageEval()

    all_outputs, all_concepts, all_targets, all_uqs = (
        load_or_generate_inference_commongen(
            model, tokenizer, batch_size, n_batch_to_validate, aq_funcs, shuffle=False
        )
    )

    # Call the function for each acquisition function
    for i, aq_func in enumerate(aq_funcs):
        plot_retention_curve_cg(
            all_outputs[i],
            all_concepts,
            all_targets,
            all_uqs[:, i],
            eval_function_commongen,
            aq_func.__class__.__name__,
            filepath=f"local/gpt-results/commongen/cg_ret_curve_{run_name}_{aq_func.__class__.__name__}_{eval_function_commongen.__class__.__name__}.svg",
        )
