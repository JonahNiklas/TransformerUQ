from math import inf
from typing import Any, List, Tuple
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset
from tqdm.notebook import tqdm
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
from torch.utils.data import DataLoader, Dataset
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

def generate_input_text(words: List[str]) -> str:
    template = f"""Task: Generate a meaningful sentence using the provided words.

Example 1:
Words: dog, run, park.
Sentence: The dog ran happily in the park.

Example 2:
Words: chef, cook, kitchen.
Sentence: The chef cooked a delicious meal in the kitchen.

Now try:
Words: {", ".join(words)}.
Sentence: """
    return template


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

def evaluate_model_batch(model: GPT, tokenizer: tiktoken.Encoding, encoding_tensors: torch.Tensor, targets: List[List[str]],eval_function_commongen: CommongenEval) -> float:
    # Use the padded encoding tensor directly to generate responses
    encoding_tensors = torch.tensor(encoding_tensors).to(hyperparameters.device)
    outputs = generate_autoregressivly_gpt2(
        model, tokenizer, encoding_tensors, search_method=topk_sampling_gpt
    )
    token_ids = outputs.token_ids
    answer_only_token_ids = [token_ids[i][len(encoding_tensors[i]):] for i in range(len(token_ids))]
    output_texts = [tokenizer.decode(ids.tolist()) for ids in answer_only_token_ids]
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


# Load the CommonGen dataset
dataset = load_dataset("common_gen", split="validation")
print(dataset)
merged_dataset = []

first_example = dataset[0]
idx = first_example["concept_set_idx"]
input = generate_input_text(first_example["concepts"])
target = [first_example["target"]]

for example in dataset:
    idx_n = example["concept_set_idx"]
    if idx_n == idx:
        target.append(example["target"])
        continue

    merged_dataset.append({
        "concept_set_idx": idx,
        "input": input,
        "target": target
    })
    idx = idx_n
    input = generate_input_text(example["concepts"])
    target = [example["target"]]
    
print("Rows in merged dataset:", len(merged_dataset))

class CommonGenDataset(Dataset):
    def __init__(self, dataset: List[dict]):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        item = self.dataset[idx]
        input_text = item["input"]
        target_text = item["target"]
        return input_text, target_text

def collate_fn(batch: Any) -> Tuple[List[str], List[str], torch.Tensor]:
    input_texts, target_texts = zip(*batch)
    # Tokenize each input text
    encodings = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in input_texts]
    # Pad sequences with a padding value (e.g., 0)
    padded_encodings = pad_sequence(encodings, batch_first=True, padding_value=0)
    return list(input_texts), list(target_texts), padded_encodings

# Create a dataset object
commongen_dataset = CommonGenDataset(merged_dataset)

# Create a DataLoader object with batching
batch_size = 8
dataloader = DataLoader(commongen_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = tiktoken.get_encoding(model_name)
model = GPT.from_pretrained(model_name)

n_batch_to_validate = 5

# Example of iterating through the DataLoader
outputs = []


for i, batch in tqdm(enumerate(dataloader), desc="Running commongen validation", total=len(dataloader)):
    input_texts, target_texts, encoding_tensors = batch
    output = evaluate_model_batch(
        model=model,
        tokenizer=tokenizer,
        encoding_tensors=encoding_tensors,
        targets=target_texts,
        eval_function_commongen=BLEU_eval()
    )
    outputs.append(output)

print("Average score: ",np.mean(outputs))
# Average score:  1.587567555989891

# # Example input words
# words = ["tree", "car", "crash"]
# input_text = generate_input_text(words)

# # Generate a sentence using the input words
# output_sentence = evaluate_model_single_example(input_text)
# print(output_sentence)
