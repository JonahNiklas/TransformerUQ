import os
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Special Tokens
# -----------------------
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

class Vocabulary:
    """
    Simple vocabulary class to map tokens to IDs and vice versa.
    """
    def __init__(self, min_freq=1, specials=None):
        if specials is None:
            specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.min_freq = min_freq
        self.specials = specials
        self.token2id = {}
        self.id2token = []
        self.freqs = Counter()

    def build_vocab(self, list_of_token_lists):
        """
        Build vocabulary from a list of lists of tokens.
        """
        # Count frequencies
        for tokens in list_of_token_lists:
            self.freqs.update(tokens)

        # Initialize with specials
        idx = 0
        for sp in self.specials:
            self.token2id[sp] = idx
            idx += 1

        # Add all tokens that pass the min_freq threshold
        # sort by frequency (descending) then lexicographically
        for token, freq in sorted(self.freqs.items(), key=lambda x: (-x[1], x[0])):
            if freq >= self.min_freq and token not in self.token2id:
                self.token2id[token] = idx
                idx += 1

        # Build id2token list from token2id
        self.id2token = [None] * len(self.token2id)
        for t, i in self.token2id.items():
            self.id2token[i] = t

    def __len__(self):
        return len(self.token2id)

    def token_to_id(self, token):
        """
        Return the ID if token is in vocabulary, otherwise return <unk> ID.
        """
        return self.token2id.get(token, self.token2id[UNK_TOKEN])

    def id_to_token(self, idx):
        """
        Reverse lookup: ID -> token.
        """
        if 0 <= idx < len(self.id2token):
            return self.id2token[idx]
        return UNK_TOKEN

    def encode(self, tokens, add_bos=False, add_eos=False):
        """
        Convert a list of tokens into IDs. Optionally add <bos> and <eos>.
        """
        out = []
        if add_bos:
            out.append(self.token2id[BOS_TOKEN])
        out.extend([self.token_to_id(t) for t in tokens])
        if add_eos:
            out.append(self.token2id[EOS_TOKEN])
        return out

    def decode(self, ids, remove_special=True):
        """
        Convert a list of IDs back into tokens. Optionally remove special tokens.
        """
        tokens = [self.id2token[i] if i < len(self) else UNK_TOKEN for i in ids]
        if remove_special:
            tokens = [t for t in tokens if t not in self.specials]
        return tokens


# -----------------------
# Dataset & Collate
# -----------------------
class ParallelDataset(Dataset):
    """
    A simple parallel dataset with source (e.g. DE) and target (e.g. EN).
    Each item is (src_tensor, tgt_tensor).
    """
    def __init__(self, src_id_lines, tgt_id_lines):
        assert len(src_id_lines) == len(tgt_id_lines), \
            "Source and target must have the same number of lines."
        self.src_id_lines = src_id_lines
        self.tgt_id_lines = tgt_id_lines

    def __len__(self):
        return len(self.src_id_lines)

    def __getitem__(self, idx):
        return (
            self.src_id_lines[idx],  # list of IDs
            self.tgt_id_lines[idx]   # list of IDs
        )

def collate_fn(batch, pad_id=0):
    """
    Custom collate function to pad source and target sequences
    to the longest in the batch.
    """
    # batch = list of (src_ids, tgt_ids)
    src_batch, tgt_batch = zip(*batch)

    # Find max lengths
    max_src_len = max(len(x) for x in src_batch)
    max_tgt_len = max(len(x) for x in tgt_batch)

    # Pad
    padded_src = []
    padded_tgt = []
    for src_ids, tgt_ids in zip(src_batch, tgt_batch):
        padded_src.append(src_ids + [pad_id]*(max_src_len - len(src_ids)))
        padded_tgt.append(tgt_ids + [pad_id]*(max_tgt_len - len(tgt_ids)))

    # Convert to tensors
    src_tensor = torch.tensor(padded_src, dtype=torch.long)
    tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long)

    return src_tensor, tgt_tensor


# -----------------------
# Main tokens_to_tensor function
# -----------------------
def tokens_to_tensor(
    train_en_path,
    train_de_path,
    test_en_path,
    test_de_path,
    batch_size=32,
    min_freq=1,
    add_bos_eos=True,
):
    """
    1. Reads tokenized (BPE) parallel files for source=DE and target=EN.
    2. Builds separate vocabularies for DE and EN.
    3. Converts all lines into lists of IDs.
    4. Returns (train_loader, test_loader).

    :param train_en_path: Path to the *English* training data (BPE).
    :param train_de_path: Path to the *German* training data (BPE).
    :param test_en_path:  Path to the *English* test data (BPE).
    :param test_de_path:  Path to the *German* test data (BPE).
    :param batch_size:    Batch size for DataLoader.
    :param min_freq:      Minimum frequency for a token to appear in vocab.
    :param add_bos_eos:   Whether to add <bos>, <eos> tokens around each sequence.
    :return: (train_loader, test_loader)
    """

    # ---------------------
    # 1. Read lines
    # ---------------------
    # Each file is expected to have one sentence per line, tokens space-separated.
    def read_tokenized_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().split() for line in f]

    train_en_tokens = read_tokenized_file(train_en_path)
    train_de_tokens = read_tokenized_file(train_de_path)
    test_en_tokens = read_tokenized_file(test_en_path)
    test_de_tokens = read_tokenized_file(test_de_path)

    # ---------------------
    # 2. Build vocabs
    # ---------------------
    en_vocab = Vocabulary(min_freq=min_freq)
    en_vocab.build_vocab(train_en_tokens)  # Build from training data only
    de_vocab = Vocabulary(min_freq=min_freq)
    de_vocab.build_vocab(train_de_tokens)

    # ---------------------
    # 3. Convert tokens -> IDs
    # ---------------------
    def convert_lines_to_ids(token_lines, vocab):
        return [vocab.encode(tokens, add_bos=add_bos_eos, add_eos=add_bos_eos)
                for tokens in token_lines]

    train_en_ids = convert_lines_to_ids(train_en_tokens, en_vocab)
    train_de_ids = convert_lines_to_ids(train_de_tokens, de_vocab)
    test_en_ids = convert_lines_to_ids(test_en_tokens, en_vocab)
    test_de_ids = convert_lines_to_ids(test_de_tokens, de_vocab)

    # ---------------------
    # 4. Build Datasets and DataLoaders
    # ---------------------
    train_dataset = ParallelDataset(train_de_ids, train_en_ids)  # DE->EN
    test_dataset  = ParallelDataset(test_de_ids, test_en_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id=en_vocab.token2id[PAD_TOKEN])  
        # *Note:* For a multi-lingual approach, we might need separate pad IDs for DE & EN,
        # but typically we store the same PAD ID in both vocabs at index 0.
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_id=en_vocab.token2id[PAD_TOKEN])
    )

    # Return DataLoaders (and optionally we might want to return vocabs too)
    return train_loader, test_loader, en_vocab, de_vocab
