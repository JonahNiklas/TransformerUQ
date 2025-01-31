from __future__ import annotations
import logging
import os
import pickle
from collections import Counter
from typing import Dict, List

import torch
from sacremoses import MosesDetokenizer, MosesTokenizer

from constants import constants


logger = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


class Vocabulary:
    def __init__(self, min_freq: int, specials: List[str] | None = None) -> None:
        if specials is None:
            specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.min_freq = min_freq
        self.specials = specials
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []
        self.freqs: Counter = Counter()

    def build_vocab_from_freqs(self) -> None:
        """
        Once self.freqs is updated with all tokens, build vocabulary.
        """
        # Initialize with specials
        idx = 0
        for sp in self.specials:
            self.token2id[sp] = idx
            idx += 1

        # Sort by frequency descending, then lexicographically
        for token, freq in sorted(self.freqs.items(), key=lambda x: (-x[1], x[0])):
            if freq >= self.min_freq and token not in self.token2id:
                self.token2id[token] = idx
                idx += 1

        # Build id2token
        self.id2token = [None] * len(self.token2id) # type: ignore
        for t, i in self.token2id.items():
            self.id2token[i] = t

    def update_freqs_from_file(self, filepath: str) -> None:
        """
        Updates the frequency counter by reading tokens line by line.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                self.freqs.update(tokens)

    def __len__(self) -> int:
        return len(self.token2id)

    def token_to_id(self, token: str) -> int:
        return self.token2id.get(token, self.token2id[UNK_TOKEN])

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.id2token):
            return self.id2token[idx]
        return UNK_TOKEN

    def encode(self, tokens: List[str], add_bos: bool, add_eos: bool) -> List[int]:
        out = []
        if add_bos:
            out.append(self.token2id[BOS_TOKEN])
        out.extend([self.token_to_id(t) for t in tokens])
        if add_eos:
            out.append(self.token2id[EOS_TOKEN])
        return out

    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:
        tokens = [self.id2token[i] if i < len(self) else UNK_TOKEN for i in ids]
        if remove_special:
            tokens = [t for t in tokens if t not in self.specials]
        return tokens


def build_and_save_vocab(
    train_en_path: str,
    train_de_path: str,
    min_freq: int,
    save_path: str,
) -> None:
    logger.info("Building shared vocabulary...")
    shared_vocab = Vocabulary(min_freq=min_freq)
    shared_vocab.update_freqs_from_file(train_en_path)
    shared_vocab.update_freqs_from_file(train_de_path)
    shared_vocab.build_vocab_from_freqs()

    with open(save_path, "wb") as f:
        pickle.dump(shared_vocab, f)

    logger.info(f"Saved shared vocab to {save_path}")


def load_vocab(vocab_file: str) -> Vocabulary:
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    assert isinstance(vocab, Vocabulary)
    return vocab


_vocab_shared = None

def output_to_text(output: List[int], lang: str="en") -> str:
    global _vocab_shared
    if _vocab_shared is None:
        logger.debug("Loading shared vocab")
        _vocab_shared = load_vocab(constants.file_paths.vocab)

    tokens = _vocab_shared.decode(output)

    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].endswith("@@") and i + 1 < len(tokens):
            tokens[i] = tokens[i][:-2] + tokens.pop(i + 1)

    detokenizer = MosesDetokenizer(lang=lang)
    text: str = detokenizer.detokenize(tokens)
    return text


if __name__ == "__main__":
    # Test output_to_text
    
    sample_sentence = "hi don't im such a cool person"
    tokenizer = MosesTokenizer(lang="en")
    tokenized_text = tokenizer.tokenize(sample_sentence)
    print("Tokenized text:")
    print(tokenized_text)
    vocab = load_vocab(constants.file_paths.vocab)
    encoded = vocab.encode(tokenized_text, add_bos=True, add_eos=True)
    print([vocab.id2token[enc] for enc in encoded])
    print("Encoded text:")
    print(encoded)
    decoded = vocab.decode(encoded)
    print("Decoded text:")
    print(decoded)
    
    
    detokenizer = MosesDetokenizer(lang="en")
    detokenized_text = detokenizer.detokenize(decoded)
    print("Detokenized text:")
    print(detokenized_text)
    