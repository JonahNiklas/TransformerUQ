from collate import collate_fn
from streaming_parallell_dataset import StreamingParallelDataset
from vocab import PAD_TOKEN, load_vocab
from torch.utils.data import DataLoader

def get_data_loader(
    src_file, 
    tgt_file,
    src_vocab, 
    tgt_vocab,
    batch_size=32,
    add_bos_eos=True,
    shuffle=True
):
    dataset = StreamingParallelDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        add_bos_eos=add_bos_eos,
        store_offsets=True  # build the offset lists
    )

    # We'll use the EN vocab's PAD token ID (assuming both vocabs have the same PAD ID)
    pad_id = tgt_vocab.token2id[PAD_TOKEN]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_id=pad_id)
    )
    return loader

if __name__ == "__main__":
    # Paths
    train_en_path = "train.bpe.en"
    train_de_path = "train.bpe.de"
    test_en_path  = "test.bpe.en"
    test_de_path  = "test.bpe.de"

    # Load the saved vocabularies
    en_vocab = load_vocab("vocab_en.pkl")
    de_vocab = load_vocab("vocab_de.pkl")

    # Build PyTorch DataLoaders for training, test
    train_loader = get_data_loader(
        src_file=train_de_path,
        tgt_file=train_en_path,
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=True
    )

    test_loader = get_data_loader(
        src_file=test_de_path,
        tgt_file=test_en_path,
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False
    )

    # Example usage:
    for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
        # src_batch.shape = [B, T_src], tgt_batch.shape = [B, T_tgt]
        print(src_batch.shape, tgt_batch.shape)
        # break or continue ...




