from math import inf
from os import remove
from typing import Any, List, Tuple
import numpy as np
from requests import get
import tiktoken
import torch
import torch.nn as nn
import tiktoken
from tqdm.notebook import tqdm
from gpt2project.data_processing.load_commongen import get_common_gen_dataloader
from gpt2project.search_methods_gpt import (
    greedy_search_gpt,
    GPT_search_method,
    topk_sampling_gpt,
)
from sacrebleu import corpus_bleu
from hyperparameters import hyperparameters
from gpt2project.gpt2model import GPT
from beam_search import AutoregressiveInferenceResults
import random
from collections import Counter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class CommongenEval:
    def __call__(self, output_text: List[str], targets: List[List[str]]) -> float:
        raise NotImplementedError("Evaluation function not implemented.")

class BLEU_eval(CommongenEval):
    def __call__(self, output_text: List[str], targets: List[List[str]]) -> float:
        # Calculate BLEU score
        bleu = corpus_bleu(output_text, targets)
        return bleu.score



# Function to evaluate the model on a given input
def generate_model_single_example(model: GPT, tokenizer: tiktoken.Encoding, input_text: str) -> AutoregressiveInferenceResults:
    # Generate a response from the model
    encodings = tokenizer.encode(input_text)
    encoding_tensor = torch.tensor(encodings, dtype=torch.long).unsqueeze(0)
    encoding_tensor = encoding_tensor.to(hyperparameters.device)
    output = generate_autoregressivly_gpt2(
        model, tokenizer, encoding_tensor, search_method=topk_sampling_gpt
    )
    return output

def evaluate_model_batch(model: GPT, tokenizer: tiktoken.Encoding, encoding_tensors: torch.Tensor, targets: List[List[str]],eval_function_commongen: CommongenEval,remove_prefix_tokens: List[int]) -> float:
    # Use the padded encoding tensor directly to generate responses
    encoding_tensors = torch.tensor(encoding_tensors).to(hyperparameters.device)
    outputs = generate_autoregressivly_gpt2(
        model, tokenizer, encoding_tensors, search_method=topk_sampling_gpt
    )
    token_ids = outputs.token_ids
    new_line_token = tokenizer.encode("\n")[0]
    non_breaking_space_token = tokenizer.encode("\xa0")[0]

    output_texts = []
    # clean the output tokens and decode them
    for b in range(len(token_ids)): # iterate over batch
        ids = token_ids[b][len(encoding_tensors[b]):]
        ids = ids[ids != non_breaking_space_token]
        while ids[0] in remove_prefix_tokens:
            ids = ids[1:]
        if new_line_token in ids:
            ids = ids[:ids.tolist().index(new_line_token)]
        output_texts.append(tokenizer.decode(ids.tolist()))
    score = eval_function_commongen(output_texts, targets)
    return score


def generate_autoregressivly_gpt2(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    tgt_tokens: torch.Tensor,
    search_method: GPT_search_method,
    max_tokens: int = 32,
) -> AutoregressiveInferenceResults:
    model.eval()
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = tokenizer.n_vocab
    output = search_method(model, tgt_tokens, vocab_size, max_tokens)
    return output


# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = tiktoken.get_encoding(model_name)
model = GPT.from_pretrained(model_name)

dataloader = get_common_gen_dataloader(batch_size=8, shuffle=False)
n_batch_to_validate = -1

# Example of iterating through the DataLoader
outputs = []
remove_prefix_tokens = [
    tokenizer.encode("\n")[0],
    tokenizer.encode("~")[0],
    tokenizer.encode("~~")[0],
    tokenizer.encode(" ")[0]
    ]
for i, batch in tqdm(enumerate(dataloader), desc="Running commongen validation", total=len(dataloader)):
    if i == n_batch_to_validate:
        break
    input_texts, target_texts, encoding_tensors = batch
    output = evaluate_model_batch(
        model=model,
        tokenizer=tokenizer,
        encoding_tensors=encoding_tensors,
        targets=target_texts,
        eval_function_commongen=BLEU_eval(),
        remove_prefix_tokens=remove_prefix_tokens
    )
    outputs.append(output)

print("Average score: ",np.mean(outputs))
# Average score:  3.9192467438661587

# # Example input words
# words = ["tree", "car", "crash"]
# input_text = generate_input_text(words)

# # Generate a sentence using the input words
# output_sentence = evaluate_model_single_example(input_text)
# print(output_sentence)
