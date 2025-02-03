import torch
import torch.nn as nn
from tqdm import tqdm
from data_processing.vocab import BOS_TOKEN, Vocabulary, load_vocab, output_to_text
from uq.acquisition_func import AcquisitionFunction, BLEU_mean_output_batch
from hyperparameters import hyperparameters
from constants import constants
from typing import List, Tuple, Union


def generate_autoregressivly_with_uq(
    model: nn.Module,
    src_tokens: torch.Tensor,
    ground_truth: torch.Tensor,
    print_ex: int,
    aq_func: AcquisitionFunction,
) -> Tuple[List[str], torch.Tensor]:
    model.eval()
    vocab_shared = load_vocab(constants.file_paths.vocab)
    batch_size = src_tokens.size(0)
    max_len = hyperparameters.transformer.max_len

    if aq_func.multiple_inference:
        text_output, logits = _generate_multiple_inference(model, src_tokens, vocab_shared, batch_size, max_len)
    else:
        text_output1, logits = _generate_single_inference(model, src_tokens, vocab_shared, batch_size, max_len)

    uq = aq_func.__call__(text_output, logits)

    print_sample_sentences(batch_size, src_tokens, ground_truth, text_output, uq, aq_func, print_ex)
    if aq_func.multiple_inference:
        return BLEU_mean_output_batch(text_output), uq
    else:
        return text_output1, uq

def _generate_single_inference(
    model: nn.Module,
    src_tokens: torch.Tensor,
    vocab_shared: Vocabulary,
    batch_size: int,
    max_len: int
) -> Tuple[List[str], torch.Tensor]:
    device = hyperparameters.device
    tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
    tgt_tokens[:, 0] = vocab_shared.token_to_id(BOS_TOKEN)
    text_output = [""] * batch_size
    logits = torch.zeros(batch_size, max_len).to(device)

    with torch.no_grad():
        for t in tqdm(range(1, max_len), desc="Generating tokens"):
            output = model(src_tokens, tgt_tokens)
            assert output.shape == (batch_size, max_len, len(vocab_shared))
            logits[:, t] = output[:, t - 1, :].max(dim=1).values
            output = output[:, t - 1, :].argmax(dim=1)
            assert output.shape == (batch_size,)
            tgt_tokens[:, t] = output

        for i in range(batch_size):
            for j in range(1, max_len):
                if tgt_tokens[i, j] == vocab_shared.token_to_id("<eos>"):
                    tgt_tokens[i, j + 1 :] = vocab_shared.token_to_id("<pad>")
                    break
            text_output[i] = output_to_text(tgt_tokens[i].tolist())
    return text_output, logits

def _generate_multiple_inference(
    model: nn.Module,
    src_tokens: torch.Tensor,
    vocab_shared: Vocabulary,
    batch_size: int,
    max_len: int
) -> Tuple[List[List[str]], torch.Tensor]:
    device = hyperparameters.device
    num_inferences = hyperparameters.uq.num_inferences
    tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
    tgt_tokens[:, 0] = vocab_shared.token_to_id(BOS_TOKEN)
    tgt_tokens = (
        tgt_tokens.unsqueeze(1)
        .expand(batch_size, num_inferences, max_len)
        .contiguous()
    )
    src_tokens = (
        src_tokens.unsqueeze(1)
        .expand(batch_size, num_inferences, src_tokens.size(1))
        .contiguous()
    )
    text_output_n = [[""] * num_inferences for _ in range(batch_size)]
    logits = torch.zeros(batch_size, num_inferences, max_len).to(device)

    _enable_test_time_dropout(model)
    with torch.no_grad():
        for n in range(num_inferences):
            for t in tqdm(range(1, max_len), desc="Generating tokens"):
                output = model(src_tokens[:, n, :], tgt_tokens[:, n, :])
                assert output.shape == (batch_size, max_len, len(vocab_shared))
                logits[:, n, t] = output[:, t - 1, :].max(dim=1).values
                output = output[:, t - 1, :].argmax(dim=1)
                assert output.shape == (batch_size,)
                tgt_tokens[:, n, t] = output

            for i in range(batch_size):
                for j in range(1, max_len):
                    if tgt_tokens[i, n, j] == vocab_shared.token_to_id("<eos>"):
                        tgt_tokens[i, n, j + 1 :] = vocab_shared.token_to_id("<pad>")
                        break
                text_output_n[i][n] = output_to_text(tgt_tokens[i, n].tolist())
    model.eval()
    return text_output_n, logits


def _enable_test_time_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def print_sample_sentences(batch_size: int, src_tokens: torch.Tensor, ground_truth: torch.Tensor, text_output: Union[List[str], List[List[str]]], uq: torch.Tensor, aq_func: AcquisitionFunction, print_ex: int) -> None:

    random_indices = torch.randperm(batch_size)[:print_ex]
    for i in random_indices:
        
        print(f"Example {i+1} in batch")
        print(src_tokens[i])
        print(f"Source: {output_to_text(src_tokens[i], lang='de')}")
        print(f"Ground truth: {output_to_text(ground_truth[i].tolist())}")
        # print(f"Source tokens: {src_tokens[i, 0].tolist()}")
        if aq_func.multiple_inference:
            for n in range(aq_func.num_inferences):
                print(f"Inference {n+1}:")
                print(f"Generated text: {text_output[i][n]}")
                # print(f"Generated tokens: {tgt_tokens[i, n].tolist()}")
                print("")
            print("=====")
        else:
            # print(f"Ground truth tokens: {tgt_tokens[i].tolist()}")
            print(f"Generated text: {text_output[i]}")
        print(f"Uncertainty: {uq[i]}")
        # print(f"Generated tokens: {tgt_tokens[i].tolist()}")
        print("")