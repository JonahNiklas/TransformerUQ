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
    vocab: Vocabulary,
    batch_size: int,
    add_bos_eos: bool,
    shuffle: bool,
    max_len: int,
) -> DataLoader:
    dataset = StreamingParallelDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        vocab=vocab,
        add_bos_eos=add_bos_eos,
        max_len=max_len,
        store_offsets=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    # Paths
    train_en_path = "local/data/training/bpe_train.en"
    train_de_path = "local/data/training/bpe_train.de"
    test_en_path = "local/data/test/bpe_test.en"
    test_de_path = "local/data/test/bpe_test.de"

    # Load the saved vocabularies
    vocab = load_vocab(constants.file_paths.vocab)

    # Build PyTorch DataLoaders for training, test
    train_loader = get_data_loader(
        src_file=train_de_path,
        tgt_file=train_en_path,
        vocab=vocab,
        batch_size=2,
        add_bos_eos=True,
        shuffle=True,
        max_len=hyperparameters.transformer.max_len,
    )

    test_loader = get_data_loader(
        src_file=test_de_path,
        tgt_file=test_en_path,
        vocab=vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    # Time the data loading
    import time

    start = time.time()
    for batch_idx, (src_batch, tgt_batch) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        for i in range(src_batch.size(0)):
            print(f"First src sentence: {src_batch[i]}")
            print(f"First tgt sentence: {tgt_batch[i]}")
            print(vocab.decode(src_batch[i]))
            print(vocab.decode(tgt_batch[i]))
            print(output_to_text(src_batch[i], "de"))
            print(output_to_text(tgt_batch[i], "en"))
        import os

        os._exit(0)
    print(f"Time taken: {time.time() - start:.2f} seconds")
