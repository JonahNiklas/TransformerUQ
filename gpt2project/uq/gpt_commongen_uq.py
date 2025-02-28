import os
from typing import Any, List, Tuple
import tiktoken
import torch
from tqdm import tqdm
from gpt2project.data_processing.load_commongen import get_common_gen_dataloader
from gpt2project.gpt2_commongen import BLEU_eval, CommongenEval, ConceptUsageEval
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2_with_uq
from gpt2project.gpt2model import GPT
from gpt2project.uq.plot_uq import plot_retention_curve_cg
from gpt2project.search_methods_gpt import topk_sampling_gpt
from hyperparameters import hyperparameters
from uq.acquisition_func import AcquisitionFunction, BLEUVar, BeamScore


def evalutate_model_batch_with_uq(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    encoding_tensors: torch.Tensor,
    remove_prefix_tokens: List[int],
    aq_funcs: List[AcquisitionFunction],
) -> Tuple[List[str],  torch.Tensor]:
    outputs, uq = generate_autoregressivly_gpt2_with_uq(
        model, tokenizer, encoding_tensors, topk_sampling_gpt, aq_funcs
    )
    token_ids = outputs
    new_line_token = tokenizer.encode("\n")[0]
    non_breaking_space_token = tokenizer.encode("\xa0")[0]

    output_texts = []
    for b in range(len(token_ids)):
        ids = token_ids[b]  # dim (max_len,)
        ids = ids[ids != non_breaking_space_token]
        while ids[0] in remove_prefix_tokens:
            ids = ids[1:]
        if new_line_token in ids:
            ids = ids[: ids.tolist().index(new_line_token)]
        output_texts.append(tokenizer.decode(ids.int().tolist()))
    return output_texts, uq

if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name)
    model.to(hyperparameters.device)
    model.eval()

    run_name = "gpt2-pretrained"

    dataloader = get_common_gen_dataloader(batch_size=1, shuffle=False)
    print("Test examples:", len(dataloader))
    n_batch_to_validate = -1

    # Example of iterating through the DataLoader
    remove_prefix_tokens = [
        tokenizer.encode("\n")[0],
        tokenizer.encode("~")[0],
        tokenizer.encode("~~")[0],
        tokenizer.encode(" ")[0],
    ]
    aq_funcs = [BeamScore(), BLEUVar()]
    eval_function_commongen = ConceptUsageEval()
    all_outputs = []
    all_concepts = []
    all_targets = []
    all_uqs = torch.zeros((0, len(aq_funcs))).to(hyperparameters.device)

    for i, batch in tqdm(
        enumerate(dataloader), desc="Running commongen validation", total=len(dataloader)
    ):
        if i == n_batch_to_validate:
            break
        input_texts, concepts, target_texts, encoding_tensors = batch
        encoding_tensors = torch.tensor(encoding_tensors).to(hyperparameters.device)

        output_texts, uq = evalutate_model_batch_with_uq(
            model=model,
            tokenizer=tokenizer,
            encoding_tensors=encoding_tensors,
            remove_prefix_tokens=remove_prefix_tokens,
            aq_funcs=aq_funcs,
        )
        all_outputs.extend(output_texts)
        all_concepts.extend(concepts)
        all_targets.extend(target_texts)
        all_uqs = torch.cat((all_uqs, uq), dim=0)


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
