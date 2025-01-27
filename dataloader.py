from tqdm import tqdm
from collate import collate_fn
from streaming_parallell_dataset import StreamingParallelDataset
from vocab import PAD_TOKEN, load_vocab
from torch.utils.data import DataLoader

def get_data_loader(
    src_file, 
    tgt_file,
    src_vocab, 
    tgt_vocab,
    batch_size,
    add_bos_eos,
    shuffle,
    max_len,
):
    dataset = StreamingParallelDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        add_bos_eos=add_bos_eos,
        max_len=max_len,
        store_offsets=True
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

if __name__ == "__main__":
    # Paths
    train_en_path = "local/data/training/bpe_train.en"
    train_de_path = "local/data/training/bpe_train.de"
    test_en_path  = "local/data/test/bpe_test.en"
    test_de_path  = "local/data/test/bpe_test.de"

    # Load the saved vocabularies
    en_vocab = load_vocab("local/vocab_en.pkl")
    de_vocab = load_vocab("local/vocab_de.pkl")

    # Build PyTorch DataLoaders for training, test
    train_loader = get_data_loader(
        src_file=train_de_path,
        tgt_file=train_en_path,
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=512
    )

    test_loader = get_data_loader(
        src_file=test_de_path,
        tgt_file=test_en_path,
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=512
    )

    # Time the data loading
    import time
    start = time.time()
    for batch_idx, (src_batch, tgt_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if batch_idx == 0:
            print(f"First src sentence: {src_batch[0]}")
            print(f"First tgt sentence: {tgt_batch[0]}")
            import os; os._exit(0)
    print(f"Time taken: {time.time() - start:.2f} seconds")




