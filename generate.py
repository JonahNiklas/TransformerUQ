import torch
import torch.nn as nn
from tqdm import tqdm

from hyperparameters import hyperparameters
from vocab import  BOS_TOKEN, load_vocab, output_to_text
from constants import constants

def generate_autoregressivly(model: nn.Module, src_tokens: torch.Tensor, print_ex:int) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    vocab = load_vocab(constants.file_paths.vocab)
    batch_size = src_tokens.size(0)
    max_len = hyperparameters.transformer.max_len
    with torch.no_grad():
        tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
        tgt_tokens[:, 0] = vocab.token_to_id(BOS_TOKEN)
            
        for t in tqdm(range(1, max_len), desc="Generating tokens"):
            output = model(src_tokens, tgt_tokens)
            assert output.shape == (batch_size, max_len, len(vocab))
            output = output[:, t-1, :]
            assert output.shape == (batch_size, len(vocab))
            output = output.argmax(dim=1)
            assert output.shape == (batch_size,)
            tgt_tokens[:, t] = output

        for i in range(batch_size):
            for j in range(1, max_len):
                if tgt_tokens[i, j] == vocab.token_to_id("<eos>"):
                    tgt_tokens[i, j+1:] = vocab.token_to_id("<pad>")
                    break
        
        random_indices = torch.randperm(batch_size)[:print_ex]
        for i in random_indices:
            print(f"Example {i+1} in batch")
            print(f"Source: {output_to_text(src_tokens[i].tolist(), lang='de')}")
            print(f"Source tokens: {src_tokens[i].tolist()}")
            print(f"Ground truth: {output_to_text(tgt_tokens[i].tolist())}")
            print(f"Ground truth tokens: {tgt_tokens[i].tolist()}")
            print(f"Generated text: {output_to_text(tgt_tokens[i].tolist())}")
            print(f"Generated tokens: {tgt_tokens[i].tolist()}")
            print("")

    return tgt_tokens