from __future__ import annotations
from typing import Callable, List, Tuple, cast
import torch
import torch.nn as nn

from vocab import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Vocabulary
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BeamSearchFunction = Callable[
    [nn.Module, torch.Tensor, int, Vocabulary, int],
    torch.Tensor
]

# This functino doesnt work because it doesnt end beams when <eos> is reached
def beam_search_batched(
    model: nn.Module,
    src_tokens: torch.Tensor,
    max_len: int,
    vocab: Vocabulary,
    beam_size: int = 4,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        batch_size = src_tokens.shape[0]
        beams = torch.zeros((batch_size, beam_size, 1), device=device, dtype=torch.int)
        beams[:, :, 0] = vocab.token_to_id(BOS_TOKEN)
        scores = torch.zeros((batch_size, beam_size), device=device)
        
        src_len = src_tokens.size(1)
        assert src_tokens.shape == (batch_size, src_len)

        for _ in tqdm(range(max_len - 1), desc="Generating tokens"):
            tgt_len = beams.size(2)
            assert beams.shape == (batch_size, beam_size, tgt_len)
            tgt_tokens = beams.view(batch_size * beam_size, tgt_len)
            assert tgt_tokens.shape == (batch_size * beam_size, tgt_len)
            src_tokens_expanded = src_tokens.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, src_len)
            assert src_tokens_expanded.shape == (batch_size * beam_size, src_len)
            output = model(src_tokens_expanded, tgt_tokens)
            assert output.shape == (batch_size * beam_size, tgt_len, len(vocab))
            output = output[:, -1, :]
            assert output.shape == (batch_size * beam_size, len(vocab))
            log_probs = torch.log_softmax(output, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)
            assert log_probs.shape == (batch_size, beam_size, len(vocab))

            # Expand beams and scores
            log_probs_topk, token_id_topk = log_probs.topk(beam_size, dim=-1)
            assert log_probs_topk.shape == token_id_topk.shape == (batch_size, beam_size, beam_size)
            log_probs_topk = log_probs_topk.view(batch_size, -1)
            token_id_topk = token_id_topk.view(batch_size, -1)
            scores_expanded = scores.unsqueeze(2).expand(-1, -1, beam_size).reshape(batch_size, beam_size * beam_size)
            assert scores_expanded.shape == (batch_size, beam_size * beam_size)
            total_scores = scores_expanded + log_probs_topk
            assert total_scores.shape == (batch_size, beam_size * beam_size)
            total_scores_topk, total_id_topk = total_scores.topk(beam_size, dim=-1)
            assert total_scores_topk.shape == total_id_topk.shape == (batch_size, beam_size)
            scores = total_scores_topk
            assert scores.shape == (batch_size, beam_size)
            tokens = token_id_topk.gather(dim=-1, index=total_id_topk)
            assert tokens.shape == (batch_size, beam_size)
            beams = torch.cat([beams, tokens.unsqueeze(2)], dim=2)
            assert beams.shape == (batch_size, beam_size, tgt_len + 1)

        # Select the best beam for each batch item
        final_tgt_tokens = beams[:, 0, :]
        
        # Clear out tokens after <eos>
        for i in range(batch_size):
            for j in range(1, max_len):
                if final_tgt_tokens[i, j] == vocab.token_to_id(EOS_TOKEN):
                    final_tgt_tokens[i, j + 1 :] = vocab.token_to_id(PAD_TOKEN)
                    break

    assert final_tgt_tokens.shape == (batch_size, max_len)
    return final_tgt_tokens


def beam_search_unbatched(
    model: nn.Module,
    src_tokens: torch.Tensor,
    max_len: int,
    vocab: Vocabulary,
    beam_size: int = 4,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        batch_size = src_tokens.size(0)

        # Initialize beams as a list of (log_prob, sequence) for each item
        beams = [
            [
                (
                    0.0,
                    torch.tensor(
                        [vocab.token_to_id(BOS_TOKEN)], device=device, dtype=torch.long
                    ),
                )
            ]
            for _ in range(batch_size)
        ]

        final_sequences: List[torch.Tensor | None] = [None] * batch_size
        for _ in tqdm(range(max_len - 1), desc="Generating tokens"):
            # Expand each beam, compute log probs for each possible next token
            new_beams: List[List[Tuple[float, torch.Tensor]]] = [
                [] for _ in range(batch_size)
            ]
            for i in range(batch_size):
                if final_sequences[i] is not None:
                    continue
                for log_prob, seq in beams[i]:
                    if seq[-1].item() == vocab.token_to_id("<eos>"):
                        final_sequences[i] = seq
                        continue
                    # Construct current tgt_tokens for model
                    tgt_for_model = seq.unsqueeze(0)  # shape [1, current_length]
                    output = model(src_tokens[i].unsqueeze(0), tgt_for_model)
                    assert output.shape == (1, tgt_for_model.size(1), len(vocab))
                    # Take the last time step from output
                    next_token_scores = torch.log_softmax(
                        output[:, -1, :], dim=-1
                    ).squeeze(0)
                    # Get top beam_size expansions
                    top_scores, top_ids = next_token_scores.topk(beam_size)
                    for s, idx in zip(top_scores, top_ids):
                        new_beams[i].append(
                            (log_prob + s.item(), torch.cat([seq, idx.view(1)]))
                        )
            # Prune to top beam_size for each item
            for i in range(batch_size):
                if final_sequences[i] is None and len(new_beams[i]) > 0:
                    new_beams[i].sort(key=lambda x: x[0], reverse=True)
                    beams[i] = new_beams[i][:beam_size]

        # Finalize sequences
        for i in range(batch_size):
            if final_sequences[i] is None:
                final_sequences[i] = max(beams[i], key=lambda x: x[0])[1]

        # Pad output sequences to max_len
        final_tgt_tokens = torch.full(
            (batch_size, max_len),
            vocab.token_to_id("<pad>"),
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(cast(List[torch.Tensor], final_sequences)):
            end = min(seq.size(0), max_len)
            final_tgt_tokens[i, :end] = seq[:end]

    assert final_tgt_tokens.shape == (batch_size, max_len)
    return final_tgt_tokens

def greedy_search(
    model: nn.Module,
    src_tokens: torch.Tensor,
    max_len: int,
    vocab: Vocabulary,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        batch_size = src_tokens.size(0)
        tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
        tgt_tokens[:, 0] = vocab.token_to_id(BOS_TOKEN)

        for t in tqdm(range(1, max_len), desc="Generating tokens"):
            output = model(src_tokens, tgt_tokens)
            assert output.shape == (batch_size, max_len, len(vocab))
            output = output[:, t - 1, :]
            assert output.shape == (batch_size, len(vocab))
            output = output.argmax(dim=1)
            assert output.shape == (batch_size,)
            tgt_tokens[:, t] = output

        for i in range(batch_size):
            for j in range(1, max_len):
                if tgt_tokens[i, j] == vocab.token_to_id(EOS_TOKEN):
                    tgt_tokens[i, j + 1 :] = vocab.token_to_id(PAD_TOKEN)
                    break

    assert tgt_tokens.shape == (batch_size, max_len)
    return tgt_tokens