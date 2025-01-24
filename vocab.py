import os
import torch
import pickle
from collections import Counter
import logging
from sacremoses import MosesDetokenizer

logger = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

class Vocabulary:
    def __init__(self, min_freq=1, specials=None):
        if specials is None:
            specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.min_freq = min_freq
        self.specials = specials
        self.token2id = {}
        self.id2token = []
        self.freqs = Counter()

    def build_vocab_from_freqs(self):
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
        self.id2token = [None] * len(self.token2id)
        for t, i in self.token2id.items():
            self.id2token[i] = t

    def update_freqs_from_file(self, filepath):
        """
        Updates the frequency counter by reading tokens line by line.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                self.freqs.update(tokens)

    def __len__(self):
        return len(self.token2id)

    def token_to_id(self, token):
        return self.token2id.get(token, self.token2id[UNK_TOKEN])

    def id_to_token(self, idx):
        if 0 <= idx < len(self.id2token):
            return self.id2token[idx]
        return UNK_TOKEN

    def encode(self, tokens, add_bos=False, add_eos=False):
        out = []
        if add_bos:
            out.append(self.token2id[BOS_TOKEN])
        out.extend([self.token_to_id(t) for t in tokens])
        if add_eos:
            out.append(self.token2id[EOS_TOKEN])
        return out

    def decode(self, ids, remove_special=True):
        tokens = [self.id2token[i] if i < len(self) else UNK_TOKEN for i in ids]
        if remove_special:
            tokens = [t for t in tokens if t not in self.specials]
        return tokens


def build_and_save_vocab(train_en_path, train_de_path, min_freq=1, 
                         save_en_path="vocab_en.pkl", save_de_path="vocab_de.pkl"):
    """
    Build English & German vocabularies from streaming of training data.
    Saves them to disk as pickle or torch file.
    """
    print("[INFO] Building English vocabulary...")
    en_vocab = Vocabulary(min_freq=min_freq)
    en_vocab.update_freqs_from_file(train_en_path)
    en_vocab.build_vocab_from_freqs()

    print("[INFO] Building German vocabulary...")
    de_vocab = Vocabulary(min_freq=min_freq)
    de_vocab.update_freqs_from_file(train_de_path)
    de_vocab.build_vocab_from_freqs()

    # Save (pickle or torch.save)
    # Option 1: Using pickle
    with open(save_en_path, "wb") as f:
        pickle.dump(en_vocab, f)
    with open(save_de_path, "wb") as f:
        pickle.dump(de_vocab, f)

    # Option 2 (Alternative): Using torch.save
    # torch.save(en_vocab, save_en_path)
    # torch.save(de_vocab, save_de_path)

    print(f"[INFO] Saved English vocab to {save_en_path}")
    print(f"[INFO] Saved German vocab  to {save_de_path}")


def load_vocab(vocab_file):
    import pickle
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    return vocab


_vocab_en = None
def output_to_text(output, lang="en"):
    global _vocab_en
    if _vocab_en is None:
        logger.debug("Loading vocab")
        _vocab_en = load_vocab("local/vocab_en.pkl")
    tokens = [_vocab_en.id_to_token(i) for i in output]
    detokenizer = MosesDetokenizer(lang=lang)
    return detokenizer.detokenize(tokens)


if __name__ == "__main__":
    # Test output_to_text
    output = [i for i in range(15)]
    text = output_to_text(output)
    print("The 15 most common words in our vocabulary are:")
    print(text)
    
