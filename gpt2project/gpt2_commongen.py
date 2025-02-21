from typing import List
import tiktoken
import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset
from gpt2project.search_methods_gpt import (
    greedy_search_gpt,
    GPT_search_method,
    topk_sampling_gpt,
)
from hyperparameters import hyperparameters
from gpt2project.gpt2model import GPT
from beam_search import AutoregressiveInferenceResults
import torch.nn.functional as F


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
def evaluate_model_single_example(input_text: str) -> str:
    # Generate a response from the model
    encodings = tokenizer.encode(input_text)
    encoding_tensor = torch.tensor(encodings, dtype=torch.long).unsqueeze(0)
    encoding_tensor = encoding_tensor.to(hyperparameters.device)
    output = generate_autoregressivly_gpt2(
        model, encoding_tensor, search_method=topk_sampling_gpt
    )
    return output


def generate_autoregressivly_gpt2(
    model: GPT,
    tgt_tokens: torch.Tensor,
    search_method: GPT_search_method,
    max_tokens: int = 32,
) -> str:
    model.eval()
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = tokenizer.n_vocab
    output = search_method(model, tgt_tokens, vocab_size, max_tokens)
    tokens = output.token_ids[0].tolist()
    decoded = tokenizer.decode(tokens)
    return decoded


# Load the CommonGen dataset
dataset = load_dataset("common_gen", split="test")
print(dataset)

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = tiktoken.get_encoding(model_name)
model = GPT.from_pretrained(model_name)

# Example input words
words = ["tree", "car", "crash"]
input_text = generate_input_text(words)

# Generate a sentence using the input words
output_sentence = evaluate_model_single_example(input_text)
print(output_sentence)
