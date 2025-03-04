import torch
import tiktoken
import numpy as np
from tqdm import tqdm
from torch import nn
from typing import List, Tuple

from beam_search import AutoregressiveInferenceResults
from gpt2project.gpt2model import GPT
from gpt2project.search_methods_gpt import GPT_search_method, topk_sampling_gpt
from hyperparameters import hyperparameters
from torch.functional import F

from uq.acquisition_func import AcquisitionFunction, BLEU_mean_output_batch

enc = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

def generate_from_model(model: nn.Module, context: str, max_length: int = 100, num_return_sequences: int = 5) -> None:
    model.eval()
    tokens: List[int] = enc.encode(context)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    tokens_tensor = tokens_tensor.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens_tensor.to(device)
    sample_rng = torch.Generator(device=device)
    pbar = tqdm(total=max_length - xgen.size(1), desc="Generating")
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
            pbar.update(1)
    pbar.close()
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"sample {i}: {decoded}")

# this is copied from the nano-gpt repo
def karpathy(model: nn.Module) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    toke = [15496, 11, 314, 1101, 257, 3303, 2746, 11] # "Hello, I'm a language model,"
    tokens = torch.tensor(toke, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(5, 1) # (5, 8)
    x = tokens.to(device)

    # generate!
    while x.size(1) < 30: # max_length=30
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x)[0] # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    for i in range(5):
        token_list = x[i, :30].tolist()
        decoded = enc.decode(token_list)
        print(">", decoded)  

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

def generate_autoregressivly_gpt2_with_uq(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    tgt_tokens: torch.Tensor,
    search_method: GPT_search_method,
    aq_funcs: List[AcquisitionFunction],
    max_tokens: int = 32,
) -> Tuple[torch.Tensor,torch.Tensor]:
    model.eval()
    tgt_tokens = tgt_tokens.to(hyperparameters.device)
    vocab_size = tokenizer.n_vocab
    batch_size = tgt_tokens.size(0)
    token_ids = torch.zeros(batch_size, hyperparameters.uq.num_inferences, max_tokens).to(hyperparameters.device)
    softmax_probs = torch.zeros(batch_size, hyperparameters.uq.num_inferences, max_tokens).to(hyperparameters.device)
    hypothesis :List[List[str]] = [[] for _ in range(batch_size)]
    uqs = torch.zeros(batch_size, len(aq_funcs)).to(hyperparameters.device)

    for n in range(hyperparameters.uq.num_inferences):
        output = search_method(model, tgt_tokens, vocab_size, max_tokens)
        
        token_ids[:, n, :] = output.token_ids[:,-max_tokens:]
        softmax_probs[:, n, :] = output.get_softmax_probs_for_selected_token()[:,-max_tokens:]
        for b in range(batch_size):
            hypothesis[b].append(tokenizer.decode(output.token_ids[b].tolist()))

    for i, aq_func in enumerate(aq_funcs):
        if aq_func.multiple_inference:
            hyp = BLEU_mean_output_batch(hypothesis)
        else:
            hyp = [hypothesis[b][0] for b in range(batch_size)]
        uq = aq_func(hypothesis, token_ids, softmax_probs)

        uqs[:, i] = uq
    
    return token_ids, uqs

# this is adapted from the karpathy method above to test how it works in the commongen pipeline 
def generate_karpathy(model: nn.Module, tgt_tokens: torch.Tensor, max_len:int) -> AutoregressiveInferenceResults:
    model.eval()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    length=tgt_tokens.size(1)
    x = tgt_tokens.to(device)

    # generate!
    while x.size(1) < length+max_len: # max_length=30
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x)[0] # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    return AutoregressiveInferenceResults(x, torch.ones(x.size(0), x.size(1)).to(hyperparameters.device))