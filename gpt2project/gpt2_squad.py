from typing import List
import tiktoken
import torch
from gpt2project.data_processing.load_squad import (
    SquadEval,
    TargetUsageEval,
    create_squad_prompt,
    create_squad_prompt_batched,
    get_squad_dataloader,
)
from gpt2project.gpt2model import GPT
from hyperparameters import hyperparameters
from gpt2project.gpt2_generate import generate_autoregressivly_gpt2
from gpt2project.search_methods_gpt import topk_sampling_gpt

def evaluate_model_batch(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    encoding_tensors: torch.Tensor,
    targets: List[List[str]],
    eval_function_squad: SquadEval,
    remove_prefix_tokens: List[int],
) -> List[float]:
    # Use the padded encoding tensor directly to generate responses
    outputs = generate_autoregressivly_gpt2(
        model, tokenizer, encoding_tensors, search_method=topk_sampling_gpt
    )
    token_ids = outputs.token_ids

    # remove the prompt from the tokens and decode them
    output_texts = [
        tokenizer.decode(token_ids[b][len(encoding_tensors[b]) :].tolist())
        for b in range(len(token_ids))
    ]
    score = eval_function_squad(output_texts, targets)
    return score

if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = tiktoken.get_encoding(model_name)
    model = GPT.from_pretrained(model_name).to(hyperparameters.device)

    dataloader = get_squad_dataloader(1, shuffle=False,force_new_clean=True)
    print("Test examples:", len(dataloader))
    n_batch_to_validate = 10
    # Example of iterating through the DataLoader
    outputs = []
    for i, batch in enumerate(dataloader):
        if i == n_batch_to_validate:
            break
        context, question, targets = batch
        tokens = tokenizer.encode_batch(create_squad_prompt_batched(context, question))
        encoding_tensors = torch.tensor(tokens).to(hyperparameters.device)
        output = evaluate_model_batch(
            model,
            tokenizer,
            encoding_tensors,
            targets,
            eval_function_squad=TargetUsageEval(),
            remove_prefix_tokens=[],
        )
        outputs.extend(output)

    print(f"{torch.mean(torch.tensor(outputs)).item():.5f}")
    # Output on all 10570 examples: 0.01315
    # Output on first 6000 examples: 0.01567
