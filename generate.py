import torch
import torch.nn as nn
from tqdm import tqdm

from hyperparameters import hyperparameters
from vocab import BOS_TOKEN, load_vocab, output_to_text
from constants import constants
from beam_search import BeamSearchFunction

def generate_autoregressivly(
    model: nn.Module,
    src_tokens: torch.Tensor,
    ground_truth: torch.Tensor,
    search_method: BeamSearchFunction,
    print_ex: int,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    vocab = load_vocab(constants.file_paths.vocab)
    batch_size = src_tokens.size(0)
    max_len = hyperparameters.transformer.max_len
    
    tgt_tokens = search_method(model, src_tokens, max_len, vocab, 4)

    random_indices = torch.randperm(batch_size)[:print_ex]
    for i in random_indices:
        print(f"Example {i+1} in batch")
        print(f"Source: {output_to_text(src_tokens[i].tolist(), lang='de')}")
        print(f"Source tokens: {src_tokens[i].tolist()}")
        print(f"Ground truth: {output_to_text(ground_truth[i].tolist())}")
        print(f"Ground truth tokens: {ground_truth[i].tolist()}")
        print(f"Generated text: {output_to_text(tgt_tokens[i].tolist())}")
        print(f"Generated tokens: {tgt_tokens[i].tolist()}")
        print("")

    return tgt_tokens
