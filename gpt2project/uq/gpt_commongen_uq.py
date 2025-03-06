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

    data = pickle.load(open(f"local/gpt-results/cg_ret_curve_{run_name}.pkl", "rb"))
    all_outputs = data["all_outputs"]
    all_concepts = data["all_concepts"]
    all_targets = data["all_targets"]
    all_uqs = data["all_uqs"]

    
    os.makedirs("local/gpt-results/commongen", exist_ok=True)
    # Call the function for each acquisition function
    for i, aq_func in enumerate(aq_funcs):
        plot_retention_curve_cg(
            all_outputs,
            all_concepts,
            all_targets,
            all_uqs[:, i],
            eval_function_commongen,
            aq_func.__class__.__name__,
            filepath=f"local/gpt-results/commongen/cg_ret_curve_{run_name}_{aq_func.__class__.__name__}_{eval_function_commongen.__class__.__name__}.svg",
        )
