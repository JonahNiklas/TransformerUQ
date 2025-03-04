from typing import Tuple
from torch import Tensor
import torch


def create_transformer_masks(
    src: Tensor,
    tgt: Tensor,
    src_key_padding_mask: Tensor,
    tgt_key_padding_mask: Tensor,
    tgt_mask: Tensor,
    nhead: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size, src_seq_length, _ = src.shape
    tgt_seq_length = tgt.shape[1]

    # Encoder self-attention mask: shape [batch_size, num_heads, src_seq_length, src_seq_length]
    enc_src_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(
        2
    )  # [batch, 1, 1, src_seq_length]
    enc_src_mask = torch.where(enc_src_mask == True, -torch.inf, 0)
    enc_src_mask = enc_src_mask.expand(
        batch_size, nhead, src_seq_length, src_seq_length
    )

    # Decoder self-attention key padding mask: shape [batch_size, num_heads, tgt_seq_length, tgt_seq_length]
    tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)
    tgt_key_padding_mask = torch.where(tgt_key_padding_mask == True, -torch.inf, 0)
    tgt_key_padding_mask = tgt_key_padding_mask.expand(
        batch_size, nhead, tgt_seq_length, tgt_seq_length
    )

    # Decoder target mask (e.g., causal mask): shape [batch_size, num_heads, tgt_seq_length, tgt_seq_length]
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(
        1
    )  # from [tgt_seq_length, tgt_seq_length]
    tgt_mask = tgt_mask.expand(batch_size, nhead, tgt_seq_length, tgt_seq_length)

    # Combine decoder masks
    tgt_mask = tgt_mask + tgt_key_padding_mask

    # MEMORY MASK for cross attention: shape [batch_size, num_heads, tgt_seq_length, src_seq_length]
    memory_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(
        2
    )  # [batch, 1, 1, src_seq_length]
    memory_mask = torch.where(memory_mask == True, -torch.inf, 0)
    memory_mask = memory_mask.expand(batch_size, nhead, tgt_seq_length, src_seq_length)

    assert isinstance(tgt_mask, torch.Tensor)
    return enc_src_mask, tgt_mask, memory_mask
