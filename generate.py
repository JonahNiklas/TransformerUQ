import torch
import torch.nn as nn
from tqdm import tqdm
from vocab import BOS_TOKEN, load_vocab, output_to_text
from acquisition_func import AcquisitionFunction
from hyperparameters import hyperparameters

def generate_autoregressivly(model: nn.Module, src_tokens: torch.Tensor, print_ex: int, aq_func: AcquisitionFunction) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    vocab_en = load_vocab("local/vocab_en.pkl")
    batch_size = src_tokens.size(0)
    max_len = hyperparameters.transformer.max_len
    with torch.no_grad():
        tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
        tgt_tokens[:, 0] = vocab_en.token_to_id(BOS_TOKEN)
        text_output = [""] * batch_size # generated full sentences
        probabilities = torch.zeros(batch_size, max_len).to(device) # probability of the most generated token at each time step
        
        if aq_func.multiple_inference:
            num_inferences = aq_func.num_inferences
            tgt_tokens = tgt_tokens.unsqueeze(1).expand(batch_size, num_inferences, max_len).contiguous()
            src_tokens = src_tokens.unsqueeze(1).expand(batch_size, num_inferences, src_tokens.size(1)).contiguous()
            text_output_n = [[""] * num_inferences for _ in range(batch_size)]
            probabilities = torch.zeros(batch_size, num_inferences, max_len).to(device)
            for n in range(num_inferences):
                for t in tqdm(range(1, max_len), desc="Generating tokens"):
                    output = model(src_tokens[:, n], tgt_tokens[:, n])
                    assert output.shape == (batch_size, max_len, len(vocab_en))
                    probabilities[:, n, t] = output[:, t-1, :].max(dim=1).values
                    output = output[:, t-1, :].argmax(dim=1)
                    assert output.shape == (batch_size,)
                    tgt_tokens[:, n, t] = output

                for i in range(batch_size):
                    for j in range(1, max_len):
                        if tgt_tokens[i, n, j] == vocab_en.token_to_id("<eos>"):
                            tgt_tokens[i, n, j+1:] = vocab_en.token_to_id("<pad>")
                            break
                    text_output_n[i][n] = output_to_text(tgt_tokens[i, n].tolist())
        else:
            for t in tqdm(range(1, max_len), desc="Generating tokens"):
                output = model(src_tokens, tgt_tokens)
                assert output.shape == (batch_size, max_len, len(vocab_en))
                probabilities[:, t] = output[:, t-1, :].max(dim=1).values
                output = output[:, t-1, :].argmax(dim=1)
                assert output.shape == (batch_size,)
                tgt_tokens[:, t] = output

            for i in range(batch_size):
                for j in range(1, max_len):
                    if tgt_tokens[i, j] == vocab_en.token_to_id("<eos>"):
                        tgt_tokens[i, j+1:] = vocab_en.token_to_id("<pad>")
                        break
                text_output[i] = output_to_text(tgt_tokens[i].tolist())
        
        if aq_func.multiple_inference:
            uq = aq_func.__call__(text_output_n, probabilities)
        else:
            uq = aq_func.__call__(text_output, probabilities) 
            
        random_indices = torch.randperm(batch_size)[:print_ex]
        for i in random_indices:
            print(f"Example {i+1} in batch")
            print(f"Source: {output_to_text(src_tokens[i].tolist(), lang='de')}")
            # print(f"Source tokens: {src_tokens[i].tolist()}")
            print(f"Ground truth: {output_to_text(tgt_tokens[i].tolist())}")
            # print(f"Ground truth tokens: {tgt_tokens[i].tolist()}")
            print(f"Generated text: {output_to_text(tgt_tokens[i].tolist())}")
            print(f"Uncertainty: {uq[i]}")
            # print(f"Generated tokens: {tgt_tokens[i].tolist()}")
            print("")

    return tgt_tokens
