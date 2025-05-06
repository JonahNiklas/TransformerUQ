import tiktoken
import torch
from typing import List

from gpt2project.constants import PADDING_TOKEN_ID


def decode_token_id_batch(
    token_ids: torch.Tensor, tokenizer: tiktoken.Encoding
) -> List[str]:
    return [
        decode_token_list(token_ids[b].tolist(), tokenizer)
        for b in range(len(token_ids))
    ]


def decode_token_list(token_list: List[int], tokenizer: tiktoken.Encoding) -> str:
    return tokenizer.decode(_remove_padding_tokens(token_list))


def _remove_padding_tokens(
    token_list: List[int], padding_id: int = PADDING_TOKEN_ID
) -> List[int]:
    return [token for token in token_list if token != padding_id]
