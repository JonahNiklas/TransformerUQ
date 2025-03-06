import os
from typing import List, Tuple
from idna import decode
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.data_processing.load_squad import (
    create_squad_prompt,
    create_squad_prompt_batched,
    get_squad_dataloader,
    TargetUsageEval,
)
from gpt2project.gpt2model import GPT
from gpt2project.uq.plot_uq import plot_retention_curve_squad
from gpt2project.utils.decode import decode_token_id_batch
from hyperparameters import hyperparameters
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.search_methods_gpt import topk_sampling_gpt
from uq.acquisition_func import AcquisitionFunction, BLEUVar, BeamScore


def evaluate_model_batch_with_uq(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    encoding_tensors: torch.Tensor,
    aq_funcs: List[AcquisitionFunction],
) -> Tuple[List[List[str]], torch.Tensor]:
    # Use the padded encoding tensor directly to generate responses
    output_texts, uq = generate_autoregressivly_gpt2_with_uq(
        model,
        tokenizer,
        encoding_tensors,
        topk_sampling_gpt,
        break_on_newline=False,
        aq_funcs=aq_funcs,
    )

    return output_texts, uq


def load_or_generate_inference(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    batch_size: int,
    n_batch_to_validate: int,
    aq_funcs: List[AcquisitionFunction],
    shuffle: bool,
) -> Tuple[List[List[str]], List[List[str]], torch.Tensor]:
    filename = f"local/gpt-results/squad/squad_outputs_{run_name}_b{batch_size}_n{n_batch_to_validate}_shuffle-{shuffle}.pt"
    if os.path.exists(filename):
        all_output_texts, all_targets, all_uqs = torch.load(filename)
        print("Loaded inference results from file.")
        return all_output_texts, all_targets, all_uqs

    print("Generating inference results...")
    all_output_texts = [[] for _ in range(len(aq_funcs))]
    all_targets = []
    all_uqs = torch.zeros((0, len(aq_funcs))).to(hyperparameters.device)

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
        output, uq = evaluate_model_batch_with_uq(
            model, tokenizer, encoding_tensors, aq_funcs
        )
        for aq in range(len(aq_funcs)):
            all_output_texts[aq].extend(
                output[aq]
            )  # should be output[aq] once we fix the returning blue-mean instead of first inference
        all_targets.extend(targets)
        all_uqs = torch.cat((all_uqs, uq), dim=0)

    os.makedirs("local/gpt-results/squad", exist_ok=True)
    torch.save((all_output_texts, all_targets, all_uqs), filename)
    print("Saved inference results to file:", filename)
    return all_output_texts, all_targets, all_uqs


if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    run_name = "gpt2-pre-1000"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name).to(hyperparameters.device)
    model.eval()

    n_batch_to_validate = 10
    batch_size = 1
    aq_funcs = [BeamScore(), BLEUVar()]
    eval_squad = TargetUsageEval()

    all_outputs, all_targets, all_uqs = load_or_generate_inference(
        model, tokenizer, batch_size, n_batch_to_validate, aq_funcs, shuffle=False
    )

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
