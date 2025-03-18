from tqdm import tqdm
from data_processing.collate import collate_fn
from hyperparameters import hyperparameters
from data_processing.streaming_parallell_dataset import StreamingParallelDataset
from data_processing.vocab import PAD_TOKEN, Vocabulary, load_vocab, output_to_text
from torch.utils.data import DataLoader
from constants import constants


def get_data_loader(
    src_file: str,
    tgt_file: str,
    src_vocab: Vocabulary, 
    tgt_vocab: Vocabulary,
    batch_size: int,
    add_bos_eos: bool,
    shuffle: bool,
    max_len: int,
) -> DataLoader:
    dataset = StreamingParallelDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        add_bos_eos=add_bos_eos,
        max_len=max_len,
        store_offsets=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
    )
    return loader

