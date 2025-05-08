from __future__ import annotations
import torch

from data_processing.vocab import output_to_text


def print_random_generated_sentences(
    src_tokens: torch.Tensor,
    ground_truth: torch.Tensor,
    tgt_tokens: torch.Tensor,
    print_ex: int = 2,
    seed: int | None = 42,
) -> None:
    batch_size = src_tokens.size(0)
    random_indices = torch.randperm(
        batch_size,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )[:print_ex]
    for i in random_indices:
        print(f"Example {i+1} in batch")
        print(f"Source: {output_to_text(src_tokens[i].tolist(), lang='de')}")
        print(f"Source tokens: {src_tokens[i].tolist()}")
        print(f"Ground truth: {output_to_text(ground_truth[i].tolist())}")
        print(f"Ground truth tokens: {ground_truth[i].tolist()}")
        print(f"Generated text: {output_to_text(tgt_tokens[i].tolist())}")
        print(f"Generated tokens: {tgt_tokens[i].tolist()}")
        print("")
