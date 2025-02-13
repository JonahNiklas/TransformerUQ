from typing import List
import torch
import torch.nn as nn
from tqdm import tqdm

from hyperparameters import hyperparameters
from data_processing.vocab import BOS_TOKEN, Vocabulary, load_vocab, output_to_text
from constants import constants
from beam_search import BeamSearchFunction
from utils.print_random_generated_sentences import print_random_generated_sentences

def generate_autoregressivly(
    model: nn.Module,
    src_tokens: torch.Tensor,
    ground_truth: torch.Tensor,
    search_method: BeamSearchFunction,
    vocab: Vocabulary,
    print_ex: int,
) -> List[str]:
    model.eval()
    inference_results = search_method(model, src_tokens, vocab)
    tgt_tokens = inference_results.token_ids
    batch_size = src_tokens.size(0)
    output_sentences = [output_to_text(tgt_tokens[i].tolist()) for i in range(batch_size)]
    print_random_generated_sentences(src_tokens, ground_truth, tgt_tokens, print_ex)

    return output_sentences
