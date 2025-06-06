from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, cast
import torch
import torch.nn as nn

from data_processing.vocab import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Vocabulary
from tqdm import tqdm
import torch.nn.functional as F

from hyperparameters import hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BeamSearchFunction = Callable[
    [nn.Module, torch.Tensor, Vocabulary], "AutoregressiveInferenceResults"
]


def beam_search_batched(
    model: nn.Module,
    src_tokens: torch.Tensor,
    vocab: Vocabulary,
) -> AutoregressiveInferenceResults:
    """
    Performs a batched beam search on a Transformer-based model and returns both the
    final token sequences and the corresponding softmax probability distributions
    (computed via teacher forcing on the final sequences).

    Args:
        model (nn.Module): Transformer model with `encode` and `decode` methods.
        src_tokens (torch.Tensor): Source sequences (batch_size, src_len).
        vocab (Vocabulary): Vocabulary for mapping tokens to ids.

    Returns:
        AutoregressiveInferenceResults: A dataclass containing:
            - tgt_tokens: A tensor of predicted token sequences (batch_size, max_len).
            - softmax_probs: A tensor of softmax probability distributions
              (batch_size, max_len, vocab_size).
    """
    batch_size = src_tokens.size(0)
    src_tokens = src_tokens.to(device)
    beam_size = hyperparameters.beam_search.beam_size
    max_len = hyperparameters.transformer.max_len

    # 2) Prepare beams
    #    Each beam is a partial hypothesis of shape (batch_size, beam_size, current_tgt_len).
    start_symbol = vocab.token_to_id(BOS_TOKEN)
    end_symbol = vocab.token_to_id(EOS_TOKEN)
    beam = torch.full(
        (batch_size, beam_size, 1), start_symbol, dtype=torch.long, device=device
    )
    # Beam scores for each partial hypothesis
    beam_scores = torch.zeros(
        (batch_size, beam_size), device=device
    )  # log probabilities
    # Keep track of finished beams
    # "finished" can be a boolean mask or a list tracking if the beam has ended
    finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)

    # We will store final output sequences here once they end or at max_len
    # In a more robust implementation, you may store multiple candidates or keep track of
    # best sequences. Here we only store the top beam for simplicity.
    final_sequences: List[List[int] | None] = [None] * batch_size
    final_scores = torch.full((batch_size,), float("-inf"), device=device)

    for step in range(max_len):
        # Prepare the current input to the decoder
        # shape: [batch_size * beam_size, step_so_far]
        decoder_input = beam.view(batch_size * beam_size, -1)

        # 3) Decode to get logits for the next token
        with torch.no_grad():
            # model output shape: [batch_size*beam_size, seq_len, vocab_size]
            encoder_input = (
                src_tokens.unsqueeze(1)
                .repeat(1, beam_size, 1)
                .view(batch_size * beam_size, -1)
            )
            model_output = model(encoder_input, decoder_input)
            next_token_logits = model_output[
                :, -1, :
            ]  # we only need the last time step

        # Convert logits to log probabilities
        log_probs = F.log_softmax(
            next_token_logits, dim=-1
        )  # [batch_size*beam_size, vocab_size]

        # 4) Select top beam_size expansions for each beam
        #    log_probs: [batch_size*beam_size, vocab_size]
        topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)
        # shape of both: [batch_size*beam_size, beam_size]

        # Reshape to group by (batch, old_beam, new_beam)
        topk_log_probs = topk_log_probs.view(batch_size, beam_size, beam_size)
        topk_ids = topk_ids.view(batch_size, beam_size, beam_size)

        # Compute expanded beam scores
        expanded_beam_scores = (
            beam_scores.unsqueeze(-1) + topk_log_probs
        )  # [batch_size, beam_size, beam_size]
        # Flatten to get best beam_size from all beam_size^2 expansions
        expanded_beam_scores = expanded_beam_scores.view(
            batch_size, -1
        )  # [batch_size, beam_size*beam_size]

        # 5) Get top beam_size across the combined dimension
        best_scores, best_positions = torch.topk(
            expanded_beam_scores, beam_size, dim=-1
        )
        # best_scores: [batch_size, beam_size]
        # best_positions: [batch_size, beam_size] indices in [0..(beam_size*beam_size - 1)]

        # Prepare to reorder beams
        # old_beam_idx = best_positions // beam_size  # which of the old beams
        # token_idx = best_positions % beam_size      # which token within topk_ids
        old_beam_idx = best_positions // beam_size
        token_idx = best_positions % beam_size

        # 6) Update beams with the chosen expansions
        # Gather the chosen old beams
        # shape of beam after gather: [batch_size, beam_size, step_so_far]
        chosen_beams = torch.gather(
            beam, 1, old_beam_idx.unsqueeze(-1).repeat(1, 1, beam.size(-1))
        )
        # Append the new chosen tokens
        chosen_tokens = torch.gather(topk_ids, 2, token_idx.unsqueeze(-1)).squeeze(
            -1
        )  # [batch_size, beam_size]

        # cat along the sequence dimension
        beam = torch.cat([chosen_beams, chosen_tokens.unsqueeze(-1)], dim=-1)

        # Update beam scores
        beam_scores = best_scores

        # 7) Check for finished beams and save final sequences if ended
        #    A beam is finished if the last token is the end_symbol
        newly_finished = chosen_tokens == end_symbol
        # Mark them as finished
        finished = finished | newly_finished

        # For any newly finished beams, compare to see if they have a better score than
        # the current final score for that batch item, and store them if so.
        for b in range(batch_size):
            for k in range(beam_size):
                if newly_finished[b, k]:
                    score_k = beam_scores[b, k].item()
                    if score_k > final_scores[b].item():
                        final_scores[b] = beam_scores[b, k]
                        final_sequences[b] = beam[b, k, :].tolist()

        # If all beams in a batch entry are finished, we can skip further decoding for that entry
        # This simple example does not do partial skipping, but you could optimize by ignoring
        # finished beams in further steps and just duplicating them.

        # If everything in the batch is finished, we can break early.
        if torch.all(finished):
            break

    # 8) If some sequences never generated an end_symbol, pick the highest-scoring beam as final
    for b in range(batch_size):
        if final_sequences[b] is None:
            # No beam ended with <eos>, take the beam with highest score
            best_k = torch.argmax(beam_scores[b])
            final_sequences[b] = beam[b, best_k, :].tolist()
    # update final_sequences type
    output_sequences = cast(List[List[int]], final_sequences)

    # Stop at fist <eos>
    for b in range(batch_size):
        if end_symbol in output_sequences[b]:
            output_sequences[b] = output_sequences[b][
                : output_sequences[b].index(end_symbol) + 1
            ]

    # Pad output sequences to max_len
    final_tgt_tokens = torch.full(
        (batch_size, max_len),
        vocab.token_to_id("<pad>"),
        dtype=torch.long,
        device=device,
    )
    for i, seq in enumerate(cast(List[List[int]], output_sequences)):
        end = min(len(seq), max_len)
        final_tgt_tokens[i, :end] = torch.tensor(seq[:end], device=device)

    # -------------------------------------------------------------------------
    # Compute softmax probabilities for the generated sequences using teacher forcing.
    # Here we re-run the model with the final sequence (which includes PAD after EOS)
    # so that we can collect the probability distributions.
    # -------------------------------------------------------------------------
    vocab_size = len(vocab)
    softmax_probs = torch.zeros(batch_size, max_len, vocab_size, device=device)
    with torch.no_grad():
        output = model(
            src_tokens, final_tgt_tokens
        )  # shape: (batch_size, max_len, vocab_size)
        for t in range(1, max_len):
            softmax_probs[:, t, :] = torch.softmax(output[:, t - 1, :], dim=-1)

    final_tgt_tokens, softmax_probs = _clean_inference_results(
        final_tgt_tokens, softmax_probs, vocab
    )

    assert final_tgt_tokens.shape == (batch_size, max_len)
    return AutoregressiveInferenceResults(final_tgt_tokens, softmax_probs)


def greedy_search(
    model: nn.Module,
    src_tokens: torch.Tensor,
    vocab: Vocabulary,
) -> AutoregressiveInferenceResults:
    max_len = hyperparameters.transformer.max_len
    with torch.no_grad():
        batch_size = src_tokens.size(0)
        tgt_tokens = torch.zeros(batch_size, max_len).long().to(device)
        softmax_probs = torch.zeros(batch_size, max_len, len(vocab)).to(device)
        tgt_tokens[:, 0] = vocab.token_to_id(BOS_TOKEN)

        for t in range(1, max_len):
            output = model(src_tokens, tgt_tokens)
            assert output.shape == (batch_size, max_len, len(vocab))
            logits = output[:, t - 1, :]
            assert logits.shape == (batch_size, len(vocab))
            probs = torch.softmax(logits, dim=-1)
            assert probs.shape == (batch_size, len(vocab))
            softmax_probs[:, t, :] = probs
            predicted_tokens = torch.argmax(probs, dim=-1)
            tgt_tokens[:, t] = predicted_tokens

        tgt_tokens, softmax_probs = _clean_inference_results(
            tgt_tokens, softmax_probs, vocab
        )

    assert tgt_tokens.shape == (batch_size, max_len)
    return AutoregressiveInferenceResults(tgt_tokens, softmax_probs)


def top_k_sampling(
    model: nn.Module,
    src_tokens: torch.Tensor,
    vocab: Vocabulary,
    k: int = 5,
    temperature: float = 0.4,
) -> AutoregressiveInferenceResults:
    max_len = hyperparameters.transformer.max_len
    with torch.no_grad():
        batch_size = src_tokens.size(0)
        tgt_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        softmax_probs = torch.zeros(batch_size, max_len, len(vocab), device=device)

        tgt_tokens[:, 0] = vocab.token_to_id(BOS_TOKEN)

        for t in range(1, max_len):
            output = model(src_tokens, tgt_tokens)
            logits = output[:, t - 1, :]

            full_probs = torch.softmax(logits, dim=-1)
            softmax_probs[:, t, :] = full_probs

            logits = logits / temperature
            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
            top_k_probs = torch.softmax(top_k_logits, dim=-1)

            sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
            predicted_tokens = top_k_indices.gather(1, sampled_indices).squeeze(-1)
            tgt_tokens[:, t] = predicted_tokens

            if torch.all((tgt_tokens == vocab.token_to_id(EOS_TOKEN)).any(dim=1)):
                break

        tgt_tokens, softmax_probs = _clean_inference_results(
            tgt_tokens, softmax_probs, vocab
        )

    assert tgt_tokens.shape == (batch_size, max_len)
    return AutoregressiveInferenceResults(tgt_tokens, softmax_probs)


def _clean_inference_results(
    tgt_tokens: torch.Tensor, softmax_probs: torch.Tensor, vocab: Vocabulary
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert (
        tgt_tokens[:, 0].eq(vocab.token_to_id(BOS_TOKEN)).all()
    ), "First token must be BOS"
    softmax_probs[:, 0, vocab.token_to_id(BOS_TOKEN)] = 1.0
    for i in range(tgt_tokens.size(0)):
        for j in range(1, tgt_tokens.size(1)):
            if tgt_tokens[i, j] == vocab.token_to_id(EOS_TOKEN):
                tgt_tokens[i, j + 1 :] = vocab.token_to_id(PAD_TOKEN)
                softmax_probs[i, j + 1 :, vocab.token_to_id(PAD_TOKEN)] = 1.0
                break
    return tgt_tokens, softmax_probs


@dataclass
class AutoregressiveInferenceResults:
    """
    Results of autoregressive inference (batch_size, max_len)
    """

    token_ids: torch.Tensor
    """
    Softmax probabilities for each token (batch_size, max_len, vocab_size)
    """
    softmax_probs: torch.Tensor

    def get_softmax_probs_for_selected_token(self) -> torch.Tensor:
        """
        Get the softmax probability for each token in token_ids by indexing into softmax_probs.
        For each batch and timestep, this returns the probability corresponding to the predicted token.

        Returns:
            A tensor of shape (batch_size, max_len) containing the probabilities for the selected tokens.
        """
        selected_probs = self.softmax_probs.gather(
            dim=2, index=self.token_ids.unsqueeze(2)
        )
        return selected_probs.squeeze(2)


def beam_search_unbatched(
    model: nn.Module,
    src_tokens: torch.Tensor,
    vocab: Vocabulary,
) -> torch.Tensor:
    """
    This function is deprecated. Use `beam_search_batched` instead.
    """
    max_len = hyperparameters.transformer.max_len
    beam_size = hyperparameters.beam_search.beam_size
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
        for _ in range(max_len - 1):
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
