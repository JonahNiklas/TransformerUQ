import pytest
import torch
import numpy as np
from gpt2project.gpt2model import GPT, GPTConfig

class TestGPTModel:
    @pytest.fixture
    def gpt_model(self) -> GPT:
        # Create a small GPT model for testing
        config = GPTConfig(
            block_size=8,  # Small sequence length for testing
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_embd=16,
        )
        return GPT(config)

    def test_create_padding_causal_mask(self, gpt_model: GPT) -> None:
        # Test case 1: No padding tokens
        batch_size = 2
        seq_length = 4
        n_head = gpt_model.config.n_head

        # Create input with no padding tokens
        idx = torch.tensor(
            [
                [1, 2, 3, 4],  # First sequence: no padding
                [5, 6, 7, 8],  # Second sequence: no padding
            ]
        )

        # Get the mask
        mask = gpt_model._create_padding_causal_mask(idx, pad_token_id=0, nhead=n_head)

        # Check shape
        assert mask.shape == (batch_size, n_head, seq_length, seq_length)

        # Since there are no padding tokens, the mask should only contain the causal mask
        # The causal mask should have 0 for positions where attention is allowed (lower triangle)
        # and -inf for positions where attention is not allowed (upper triangle)

        # Check first sequence, first head
        expected_causal_pattern = torch.tensor(
            [
                [0, -float("inf"), -float("inf"), -float("inf")],
                [0, 0, -float("inf"), -float("inf")],
                [0, 0, 0, -float("inf")],
                [0, 0, 0, 0],
            ]
        )

        # Check if the causal pattern is correct for both sequences
        for b in range(batch_size):
            for h in range(n_head):
                torch.testing.assert_close(mask[b, h], expected_causal_pattern)

        # Test case 2: With padding tokens
        idx_with_padding = torch.tensor(
            [
                [1, 2, 0, 0],  # First sequence: last two tokens are padding
                [5, 0, 0, 0],  # Second sequence: last three tokens are padding
            ]
        )

        # Get the mask
        mask_with_padding = gpt_model._create_padding_causal_mask(
            idx_with_padding, pad_token_id=0, nhead=n_head
        )

        # Check shape
        assert mask_with_padding.shape == (batch_size, n_head, seq_length, seq_length)

        # For the first sequence, positions (0,2), (0,3), (1,2), (1,3), (2,2), (2,3), (3,2), (3,3)
        # should have -inf due to padding
        expected_first_seq = torch.tensor(
            [
                [0, -float("inf"), -float("inf"), -float("inf")],
                [0, 0, -float("inf"), -float("inf")],
                [0, 0, -float("inf"), -float("inf")],
                [0, 0, -float("inf"), -float("inf")],
            ]
        )

        # For the second sequence, even more positions should have -inf due to more padding
        expected_second_seq = torch.tensor(
            [
                [0, -float("inf"), -float("inf"), -float("inf")],
                [0, -float("inf"), -float("inf"), -float("inf")],
                [0, -float("inf"), -float("inf"), -float("inf")],
                [0, -float("inf"), -float("inf"), -float("inf")],
            ]
        )

        # Check if the patterns are correct
        for h in range(n_head):
            torch.testing.assert_close(mask_with_padding[0, h], expected_first_seq)
            torch.testing.assert_close(mask_with_padding[1, h], expected_second_seq)

    def test_forward_with_and_without_padding(self, gpt_model: GPT) -> None:
        input_ids = torch.tensor([1, 2, 3, 4]).unsqueeze(0)
        output, loss = gpt_model.forward(input_ids)
        assert output.shape == (1, 4, 100)

        # Add padding tokens
        input_ids_with_padding = torch.cat(
            [input_ids, torch.zeros(1, 4, dtype=torch.long)], dim=1
        )
        output_with_padding, loss_with_padding = gpt_model.forward(
            input_ids_with_padding
        )
        assert output_with_padding.shape == (1, 8, 100)

        # Check that the output is the same for both sequences
        assert torch.allclose(output[:, -1, :], output_with_padding[:, -1, :])
