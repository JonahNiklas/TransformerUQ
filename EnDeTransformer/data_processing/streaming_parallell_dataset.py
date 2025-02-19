from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import os
import logging

from EnDeTransformer.data_processing.vocab import Vocabulary

logger = logging.getLogger(__name__)


class StreamingParallelDataset(Dataset):
    """
    A dataset that streams source & target lines from disk without loading everything into memory.
    We'll store the file offsets of each line for random access (if needed).
    Otherwise, we can read sequentially if shuffle=False, but DataLoader shuffle complicates that.
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        vocab: Vocabulary,
        max_len: int,
        add_bos_eos: bool,
        store_offsets: bool,
    ) -> None:
        super().__init__()
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.vocab = vocab
        self.add_bos_eos = add_bos_eos
        self.max_len = max_len

        # We will store (byte_offset_src[i], byte_offset_tgt[i]) for line i.
        # This allows random access in __getitem__.
        self.offsets_src = []
        self.offsets_tgt = []

        cache_path = self.src_file.rpartition("/")[0] + "/offsets.pkl"
        if os.path.exists(cache_path):
            logger.debug("Load offsets from disk pkl")
            self.offsets_src, self.offsets_tgt = torch.load(cache_path)
            return

        if store_offsets:
            with open(self.src_file, "r", encoding="utf-8") as f_src:
                offset = f_src.tell()
                line = f_src.readline()
                while line:
                    self.offsets_src.append(offset)
                    offset = f_src.tell()
                    line = f_src.readline()

            with open(self.tgt_file, "r", encoding="utf-8") as f_tgt:
                offset = f_tgt.tell()
                line = f_tgt.readline()
                while line:
                    self.offsets_tgt.append(offset)
                    offset = f_tgt.tell()
                    line = f_tgt.readline()

            assert len(self.offsets_src) == len(
                self.offsets_tgt
            ), "Source and target must have the same number of lines."

            logger.debug("Save offsets to disk pkl")
            torch.save((self.offsets_src, self.offsets_tgt), cache_path)

    def __len__(self) -> int:
        return len(self.offsets_src)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        # Move to the correct offset for source & target
        with open(self.src_file, "r", encoding="utf-8") as f_src:
            f_src.seek(self.offsets_src[idx])
            src_line = f_src.readline().strip().split()

        with open(self.tgt_file, "r", encoding="utf-8") as f_tgt:
            f_tgt.seek(self.offsets_tgt[idx])
            tgt_line = f_tgt.readline().strip().split()

        # Encode
        src_ids = self.vocab.encode(
            src_line, add_bos=self.add_bos_eos, add_eos=self.add_bos_eos
        )
        tgt_ids = self.vocab.encode(
            tgt_line, add_bos=self.add_bos_eos, add_eos=self.add_bos_eos
        )

        # Clip to max_len
        src_ids = src_ids[: self.max_len]
        tgt_ids = tgt_ids[: self.max_len]

        return src_ids, tgt_ids
