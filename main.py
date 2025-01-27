import os
import torch
from dataloader import get_data_loader, load_vocab
from models.transformer import Transformer
from tokenizer import ParallelCorpusTokenizer
from train import train
import logging
from torch.utils.data import DataLoader

from vocab import build_and_save_vocab

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')

def main():
    logger.info("Tokenize data")
    tokenizer = ParallelCorpusTokenizer()
    tokenizer.tokenize_files(
        train_en_path="local/data/training/train.de",
        train_de_path="local/data/training/train.en",
        test_en_path="local/data/test/test.de",
        test_de_path="local/data/test/test.en",
        output_train_en="local/data/training/tokenized_train.de",
        output_train_de="local/data/training/tokenized_train.en",
        output_test_en="local/data/test/tokenized_test.de",
        output_test_de="local/data/test/tokenized_test.en",
    )
    languages = ["en", "de"]
    for lang in languages:
        logger.info(f"Learning BPE codes for {lang.upper()}")
        tokenizer.learn_bpe(
            input_path=f"local/data/training/tokenized_train.{lang}",
            output_codes_path=f"local/data/training/{lang}_bpe_codes.txt",
        )
        logger.info(f"Applying BPE to {lang.upper()} data")
        tokenizer.apply_bpe(
            input_path=f"local/data/training/tokenized_train.{lang}",
            output_path=f"local/data/training/bpe_train.{lang}",
            codes_path=f"local/data/training/{lang}_bpe_codes.txt",
        )
        logger.info(f"Applying BPE to {lang.upper()} test data")
        tokenizer.apply_bpe(
            input_path=f"local/data/test/tokenized_test.{lang}",
            output_path=f"local/data/test/bpe_test.{lang}",
            codes_path=f"local/data/training/{lang}_bpe_codes.txt",
        )

    logger.info("Build and save vocab")
    if not os.path.exists("local/vocab_en.pkl") or not os.path.exists("local/vocab_de.pkl"):
        build_and_save_vocab(
            train_en_path="local/data/training/bpe_train.en",
            train_de_path="local/data/training/bpe_train.de",
            min_freq=2000,
            save_en_path="local/vocab_en.pkl",
            save_de_path="local/vocab_de.pkl"
        )
        logger.warning("Vocab files not found. Building vocab from training data.")
    en_vocab = load_vocab("local/vocab_en.pkl")
    de_vocab = load_vocab("local/vocab_de.pkl")
    logger.info(f"English vocab size: {len(en_vocab)}")
    logger.info(f"German vocab size: {len(de_vocab)}")

    logger.info("Create data loaders")
    max_len=512
    training_loader = get_data_loader(
        src_file="local/data/training/bpe_train.de",
        tgt_file="local/data/training/bpe_train.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=max_len,
    )

    test_loader = get_data_loader(
        src_file="local/data/test/bpe_test.de",
        tgt_file="local/data/test/bpe_test.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=64,
        add_bos_eos=True,
        shuffle=False,
        max_len=max_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating model")
    class Hyperparameter:
        def __init__(self):
            self.encoder_embed_dim: int = 512
            self.encoder_ffn_embed_dim: int = 1024
            self.encoder_attention_heads: int = 4
            self.encoder_layers: int = 6

    hyperparameters = Hyperparameter()
    model = Transformer(
        src_vocab_size=len(de_vocab),
        tgt_vocab_size=len(en_vocab),
        d_model=hyperparameters.encoder_embed_dim,
        num_heads=hyperparameters.encoder_attention_heads,
        d_ff=hyperparameters.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.encoder_layers,
        num_decoder_layers=hyperparameters.encoder_layers,
        dropout=0.1,
        max_len=max_len,
    )
    number_of_params = sum(p.numel() for p in model.parameters())
    print(f"Model number of parameters: {number_of_params/1e6:.2f}M")
    model.to(device)
    if torch.cuda.is_available():
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    train(
        model,
        training_loader,
        test_loader,
        optimizer,
        criterion,
        max_steps=500_000,
    )


if __name__ == "__main__":
    main()
