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
from gpt2project.utils.decode import decode_token_id_batch
from hyperparameters import hyperparameters
from uq.acquisition_func import BLEUVar, BeamScore

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evalutate_model_batch_with_uq(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    encoding_tensors: torch.Tensor,
    targets: List[List[str]],
    aq_funcs: List[Any],
) -> Tuple[List[str], List[List[str]], torch.Tensor]:
    outputs, uq = generate_autoregressivly_gpt2_with_uq(
        model,
        tokenizer,
        encoding_tensors,
        topk_sampling_gpt,
        True,
        aq_funcs,
        max_tokens=20,
    )
    output_texts = decode_token_id_batch(outputs, tokenizer)
    return output_texts, targets, uq


def plot_retention_curve(
    output_texts: List[str],
    concepts: List[str],
    targets: List[List[str]],
    uq: torch.Tensor,
    eval_function: Any,
    aq_func_name: str,
    filepath: str,
) -> None:
    # Sort the results based on UQ
    sorted_indices = sorted(range(len(uq)), key=lambda i: uq[i].item())
    assert sorted_indices != list(range(len(uq))), "UQ is not working"
    sorted_outputs = [output_texts[i] for i in sorted_indices]
    sorted_targets = [targets[i] for i in sorted_indices]
    sorted_concepts = [concepts[i] for i in sorted_indices]

    # Evaluate and plot retention curve
    import matplotlib.pyplot as plt

    cutoffs = range(1, len(sorted_outputs) + 1)
    retention_scores = []

    for cutoff in cutoffs:
        selected_outputs = sorted_outputs[:cutoff]
        selected_targets = sorted_targets[:cutoff]
        selected_concepts = sorted_concepts[:cutoff]
        score = eval_function(selected_outputs, selected_concepts, selected_targets)
        retention_scores.append(score)

    plt.figure()
    plt.plot(cutoffs, retention_scores)
    plt.xlabel("Number of Samples")
    plt.ylabel("Evaluation Score")
    plt.title(f"Retention Curve for {aq_func_name}")
    plt.savefig(filepath)
    plt.show()


if __name__ == "__main__":
    # Load the  GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name)
    model.to(hyperparameters.device)
    model.eval()

    run_name = "gpt2-pretrained"

    dataloader = get_common_gen_dataloader(batch_size=1, shuffle=False)
    print("Test examples:", len(dataloader))

    aq_funcs = [BeamScore(), BLEUVar()]
    eval_function_commongen = ConceptUsageEval()

    all_outputs: List[str] = []
    all_concepts: List[str] = []
    all_targets: List[List[str]] = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)
    for i, (input_texts, concepts, target_texts, encoding_tensors) in tqdm(
        enumerate(dataloader),
        desc="Running commongen validation",
        total=len(dataloader),
    ):
        output_texts, targets, uq = evalutate_model_batch_with_uq(
            model=model,
            tokenizer=tokenizer,
            encoding_tensors=encoding_tensors,
            targets=target_texts,
            aq_funcs=aq_funcs,
        )
        all_outputs.extend(output_texts)
        all_concepts.extend(concepts)
        all_targets.extend(targets)
        all_uqs = torch.cat((all_uqs, uq), dim=0)

    pickle.dump(
        {
            "all_outputs": all_outputs,
            "all_concepts": all_concepts,
            "all_targets": all_targets,
            "all_uqs": all_uqs,
        },
        open(f"local/gpt-results/cg_ret_curve_{run_name}.pkl", "wb"),
    )

    data = pickle.load(
        open(f"local/gpt-results/cg_ret_curve_{run_name}.pkl", "rb")
    )
    all_outputs = data["all_outputs"]
    all_concepts = data["all_concepts"]
    all_targets = data["all_targets"]
    all_uqs = data["all_uqs"]

    os.makedirs("local/gpt-results", exist_ok=True)
    # Call the function for each acquisition function
    for i, aq_func in enumerate(aq_funcs):
        plot_retention_curve(
            all_outputs,
            all_concepts,
            all_targets,
            all_uqs[:, i],
            eval_function_commongen,
            aq_func.__class__.__name__,
            filepath=f"local/gpt-results/cg_ret_curve_{run_name}_{aq_func.__class__.__name__}_{eval_function_commongen.__class__.__name__}.png",
        )
