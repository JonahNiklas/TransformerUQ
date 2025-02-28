import os
from typing import List, Tuple
import tiktoken
import torch
from gpt2project.data_processing.load_squad import (
    create_squad_prompt,
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


def evaluate_model_batch_with_uq(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    encoding_tensors: torch.Tensor,
    aq_funcs: List[AcquisitionFunction],
) -> Tuple[List[str], torch.Tensor]:
    # Use the padded encoding tensor directly to generate responses
    outputs, uq, hypotheses = generate_autoregressivly_gpt2_with_uq(
        model, tokenizer, encoding_tensors, topk_sampling_gpt, aq_funcs
    )
    token_ids = outputs

    return hypotheses, uq


if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    run_name = "gpt2-pre-1000test"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name).to(hyperparameters.device)
    model.eval()

    dataloader = get_squad_dataloader(1, shuffle=False, force_new_clean=True)
    print("Test examples:", len(dataloader))
    n_batch_to_validate = 1000

    aq_funcs = [BeamScore(), BLEUVar()]
    eval_squad = TargetUsageEval()

    all_outputs = [[]]*len(aq_funcs)
    all_targets = []
    all_uqs = torch.zeros((0, len(aq_funcs))).to(hyperparameters.device)
    for i, batch in enumerate(dataloader):
        if i == n_batch_to_validate:
            break
        context, question, targets = batch
        tokens = tokenizer.encode_batch(create_squad_prompt_batched(context, question))
        encoding_tensors = torch.tensor(tokens).to(hyperparameters.device)
        output, uq = evaluate_model_batch_with_uq(
            model, tokenizer, encoding_tensors, aq_funcs
        )
        for aq in range(len(aq_funcs)):
            all_outputs[aq].extend(output[aq])
        all_targets.extend(targets)
        all_uqs = torch.cat((all_uqs, uq), dim=0)

    os.makedirs("local/gpt-results/squad", exist_ok=True)
    # Call the function for each acquisition function
    for i, aq_func in enumerate(aq_funcs):
        plot_retention_curve_squad(
            all_outputs[i],
            all_targets,
            all_uqs[:, i],
            eval_squad,
            aq_func.__class__.__name__,
            filepath=f"local/gpt-results/squad/squad_ret_curve_{run_name}_{aq_func.__class__.__name__}_{eval_squad.__class__.__name__}.svg",
        )
