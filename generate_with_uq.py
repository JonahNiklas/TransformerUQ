import torch
import torch.nn as nn
from tqdm import tqdm
from vocab import BOS_TOKEN, load_vocab, output_to_text
from acquisition_func import AcquisitionFunction, BLEU_mean_output_batch
from hyperparameters import hyperparameters
from constants import constants
from typing import List, Tuple

def _enable_test_time_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def generate_autoregressivly_with_uq(model: nn.Module, src_tokens: torch.Tensor, ground_truth: torch.Tensor, print_ex: int, aq_func: AcquisitionFunction) -> Tuple[List[str], torch.Tensor]:
    model.eval()
    device = hyperparameters.device
    vocab_shared = load_vocab(constants.file_paths.vocab)
    batch_size = src_tokens.size(0)
    max_len = hyperparameters.transformer.max_len
    with torch.no_grad():
        tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
        tgt_tokens[:, 0] = vocab_shared.token_to_id(BOS_TOKEN)
        text_output = [""] * batch_size # generated full sentences
        logits = torch.zeros(batch_size, max_len).to(device) # probability of the most generated token at each time step
        
        if aq_func.multiple_inference:
            num_inferences = hyperparameters.uq.num_inferences
            tgt_tokens = tgt_tokens.unsqueeze(1).expand(batch_size, num_inferences, max_len).contiguous()
            # tgt_token: (batch_size, num_inferences, max_len)
            src_tokens = src_tokens.unsqueeze(1).expand(batch_size, num_inferences, src_tokens.size(1)).contiguous()
            text_output_n = [[""] * num_inferences for _ in range(batch_size)]
            logits = torch.zeros(batch_size, num_inferences, max_len).to(device)
            _enable_test_time_dropout(model)
            for n in range(num_inferences):
                for t in tqdm(range(1, max_len), desc="Generating tokens"):
                    output = model(src_tokens[:, n, :], tgt_tokens[:, n, :])
                    assert output.shape == (batch_size, max_len, len(vocab_shared))
                    logits[:, n, t] = output[:, t-1, :].max(dim=1).values
                    output = output[:, t-1, :].argmax(dim=1)
                    assert output.shape == (batch_size,)
                    tgt_tokens[:, n, t] = output

                for i in range(batch_size):
                    for j in range(1, max_len):
                        if tgt_tokens[i, n, j] == vocab_shared.token_to_id("<eos>"):
                            tgt_tokens[i, n, j+1:] = vocab_shared.token_to_id("<pad>")
                            break
                    text_output_n[i][n] = output_to_text(tgt_tokens[i, n].tolist())
            model.eval()
        else:
            for t in tqdm(range(1, max_len), desc="Generating tokens"):
                output = model(src_tokens, tgt_tokens)
                assert output.shape == (batch_size, max_len, len(vocab_shared))
                logits[:, t] = output[:, t-1, :].max(dim=1).values
                output = output[:, t-1, :].argmax(dim=1)
                assert output.shape == (batch_size,)
                tgt_tokens[:, t] = output

            for i in range(batch_size):
                for j in range(1, max_len):
                    if tgt_tokens[i, j] == vocab_shared.token_to_id("<eos>"):
                        tgt_tokens[i, j+1:] = vocab_shared.token_to_id("<pad>")
                        break
                text_output[i] = output_to_text(tgt_tokens[i].tolist())
        
        if aq_func.multiple_inference:
            uq = aq_func.__call__(text_output_n, logits)
        else:
            uq = aq_func.__call__(text_output, logits) 
            
        random_indices = torch.randperm(batch_size)[:print_ex]
        for i in random_indices:
            if aq_func.multiple_inference:
                src = src_tokens[i, 0].tolist()
            else:
                src = src_tokens[i].tolist()
            print(f"Example {i+1} in batch")
            print(f"Source: {output_to_text(src, lang='de')}")
            print(f"Ground truth: {output_to_text(ground_truth[i].tolist())}")
            # print(f"Source tokens: {src_tokens[i, 0].tolist()}")
            if aq_func.multiple_inference:
                for n in range(aq_func.num_inferences):
                    print(f"Inference {n+1}:")
                    print(f"Generated text: {text_output_n[i][n]}")
                    # print(f"Generated tokens: {tgt_tokens[i, n].tolist()}")
                    print("")
                print("=====")
            else:
                # print(f"Ground truth tokens: {tgt_tokens[i].tolist()}")
                print(f"Generated text: {text_output[i]}")
            print(f"Uncertainty: {uq[i]}")
            # print(f"Generated tokens: {tgt_tokens[i].tolist()}")
            print("")

    if aq_func.multiple_inference:
        return BLEU_mean_output_batch(text_output_n), uq
    return text_output, uq
