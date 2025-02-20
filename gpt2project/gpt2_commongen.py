from typing import List
from sympy import hyper
import tiktoken
import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset
from data_processing.vocab import Vocabulary
from hyperparameters import hyperparameters
from gpt2project.gpt2model import GPT
from beam_search import AutoregressiveInferenceResults, greedy_search

# Load the CommonGen dataset
dataset = load_dataset("common_gen",split="test")
print(dataset)
# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = tiktoken.get_encoding(model_name)

model = GPT.from_pretrained(model_name)

def generate_input_text(words: List[str]) -> str:
    template = f"""
    Task: Generate a meaningful sentence using the provided words.

    Example 1:
    Words: dog, run, park.
    Sentence: The dog ran happily in the park.

    Example 2:
    Words: chef, cook, kitchen.
    Sentence: The chef cooked a delicious meal in the kitchen.

    Now try:
    Words: {", ".join(words)}.
    Sentence:
    """
    return template

# Function to evaluate the model on a given input
def evaluate_model_single_example(input_text: str) -> str:
    # Generate a response from the model
    encodings = tokenizer.encode(input_text)
    encoding_tensor = torch.tensor(encodings, dtype=torch.long).unsqueeze(0)
    encoding_tensor = encoding_tensor.to(hyperparameters.device)
    output = generate_autoregressivly_gpt2(model, encoding_tensor, max_tokens=20)
    return output

def generate_autoregressivly_gpt2(model: GPT, tgt_tokens: torch.Tensor, max_tokens: int = 100) -> str:
    model.eval()
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = tokenizer.n_vocab
    output = greedy_search_gpt(model, tgt_tokens, vocab_size, max_len=max_tokens)
    tokens = output.token_ids
    decoded = tokenizer.decode(tokens[0])

def greedy_search_gpt(
    model: nn.Module,
    tgt_tokens: torch.Tensor,
    vocab_size: int,
    max_len:int,
) -> AutoregressiveInferenceResults:
    with torch.no_grad():
        batch_size = tgt_tokens.size(0)
        tgt_tokens = torch.zeros(batch_size, max_len).long().to(hyperparameters.device)
        softmax_probs = torch.zeros(batch_size, max_len, vocab_size).to(hyperparameters.device)
        

        for t in range(1,max_len):
            output = model(tgt_tokens)
            assert output.shape == (batch_size, max_len, vocab_size)
            logits = output[:, t - 1, :]
            assert logits.shape == (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            assert probs.shape == (batch_size, vocab_size)
            softmax_probs[:, t, :] = probs
            predicted_tokens = torch.argmax(probs, dim=-1)
            tgt_tokens[:, t] = predicted_tokens
    assert tgt_tokens.shape == (batch_size, max_len)
    return AutoregressiveInferenceResults(tgt_tokens, softmax_probs)


# Example input words
words = ["apple", "tree", "fruit"]
input_text = generate_input_text(words)

# Generate a sentence using the input words
output_sentence = evaluate_model_single_example(input_text)
print(input_text)
print("=====================================")
print(output_sentence)


