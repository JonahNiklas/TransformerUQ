import os
from typing import List, Tuple
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.data_processing.load_squad import (
    create_squad_prompt_batched,
    get_squad_dataloader,
    TargetUsageEval,
)
from gpt2project.gpt2model import GPT
from gpt2project.uq.plot_uq import plot_retention_curve_squad
from hyperparameters import hyperparameters
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.search_methods_gpt import topk_sampling_gpt
from uq.acquisition_func import AcquisitionFunction, BLEUVar, BeamScore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_squad(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunction],
    shuffle: bool,
    run_name: str,
) -> Tuple[List[List[str]], List[List[str]], torch.Tensor]:
    filename = f"local/gpt-results/squad/squad_outputs_{run_name}_b{batch_size}_n{n_batch_to_validate}_shuffle-{shuffle}.pt"
    if os.path.exists(filename):
        all_output_texts, all_targets, all_uqs = torch.load(filename)
        logger.info("Loaded inference results from file.")
        return all_output_texts, all_targets, all_uqs

    logger.info("Generating inference results...")
    all_output_texts = [[] for _ in range(len(aq_funcs))]
    all_targets = []
    all_uqs = torch.empty((0, len(aq_funcs))).to(hyperparameters.device)

    dataloader = get_squad_dataloader(batch_size, shuffle=shuffle)
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
            topk_sampling_gpt,
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


if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    run_name = "gpt2-pre-1000"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name).to(hyperparameters.device)
    model.eval()

    n_batch_to_validate = 1000
    batch_size = 1
    aq_funcs = [BeamScore(), BLEUVar()]
    squad_eval_function = TargetUsageEval()

    all_outputs, all_targets, all_uqs = eval_squad(
        model,
        tokenizer,
        batch_size,
        n_batch_to_validate,
        aq_funcs,
        shuffle=False,
        run_name=run_name,
    )

    os.makedirs("local/gpt-results/squad", exist_ok=True)
    # Call the function for each acquisition function
    for i, aq_func in enumerate(aq_funcs):
        plot_retention_curve_squad(
            all_outputs[i],
            all_targets,
            all_uqs[:, i],
            squad_eval_function,
            aq_func.__class__.__name__,
            filepath=f"local/gpt-results/squad/squad_ret_curve_{run_name}_{aq_func.__class__.__name__}_{eval_squad.__class__.__name__}.svg",
        )
