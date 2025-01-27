import torch
import torch.nn as nn
from tqdm import tqdm

from hyperparameters import Hyperparameter
from vocab import  BOS_TOKEN, load_vocab, output_to_text

def generate_autoregressivly(model: nn.Module, src_tokens: torch.Tensor, print_ex:int) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    vocab_en = load_vocab("local/vocab_en.pkl")
    batch_size = src_tokens.size(0)
    max_len = Hyperparameter().max_len
    with torch.no_grad():
        tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
        tgt_tokens[:, 0] = vocab_en.token_to_id(BOS_TOKEN)
            
        for t in tqdm(range(1, max_len), desc="Generating tokens"):
            output = model(src_tokens, tgt_tokens)
            output = output.argmax(dim=-1)
            tgt_tokens[:, t] = output[:, t-1]
        
        for i in range(print_ex):
            print(f"Example {i+1}")
            print(f"Source: {output_to_text(src_tokens[i].tolist())}")
            print(f"Generated: {output_to_text(tgt_tokens[i].tolist())}")
            print("")

    return tgt_tokens