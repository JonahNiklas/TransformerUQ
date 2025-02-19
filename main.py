import math
import os
from typing import Literal
import torch
import wandb
from constants import constants
from data_processing.dataloader import get_data_loader, load_vocab
from data_processing.tokenizer import ParallelCorpusTokenizer
from hyperparameters import hyperparameters
from models.transformer_model import TransformerModel
from train import train
import logging
from torch import nn
from data_processing.vocab import build_and_save_vocab, load_vocab

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

torch_compile = os.getenv("TORCH_COMPILE", "TRUE") != "FALSE"
use_wandb = os.getenv("USE_WANDB", "TRUE") != "FALSE"

def main() -> None:
    logger.info("Tokenize data")
    tokenizer = ParallelCorpusTokenizer()
    tokenizer.tokenize_files(
        train_en_path=constants.file_paths.train_en,
        train_de_path=constants.file_paths.train_de,
        dev_en_path=constants.file_paths.dev_en,
        dev_de_path=constants.file_paths.dev_de,
        test_en_path=constants.file_paths.test_en,
        test_de_path=constants.file_paths.test_de,
        output_train_en=constants.file_paths.tokenized_train_en,
        output_train_de=constants.file_paths.tokenized_train_de,
        output_dev_en=constants.file_paths.tokenized_dev_en,
        output_dev_de=constants.file_paths.tokenized_dev_de,
        output_test_en=constants.file_paths.tokenized_test_en,
        output_test_de=constants.file_paths.tokenized_test_de,
        test_ood_en_path=constants.file_paths.ood_en,
        test_ood_nl_path=constants.file_paths.ood_nl,
        output_test_ood_en=constants.file_paths.tokenized_ood_en,
        output_test_ood_nl=constants.file_paths.tokenized_ood_nl,
    )
    for lang in ["en", "de"]:
        logger.info(f"Learning BPE codes for {lang.upper()}")
        tokenizer.learn_bpe(
            input_path=f"local/iwslt/training/tokenized_train.{lang}",
            output_codes_path=f"local/iwslt/training/{lang}_bpe_codes.txt",
        )
        logger.info(f"Applying BPE to {lang.upper()} data")
        tokenizer.apply_bpe(
            input_path=f"local/iwslt/training/tokenized_train.{lang}",
            output_path=f"local/iwslt/training/bpe_train.{lang}",
            codes_path=f"local/iwslt/training/{lang}_bpe_codes.txt",
        )
        logger.info(f"Applying BPE to {lang.upper()} test data")
        tokenizer.apply_bpe(
            input_path=f"local/iwslt/test/tokenized_test.{lang}",
            output_path=f"local/iwslt/test/bpe_test.{lang}",
            codes_path=f"local/iwslt/training/{lang}_bpe_codes.txt",
        )
        logger.info(f"Applying BPE to {lang.upper()} dev data")
        tokenizer.apply_bpe(
            input_path=f"local/iwslt/dev/tokenized_dev.{lang}",
            output_path=f"local/iwslt/dev/bpe_dev.{lang}",
            codes_path=f"local/iwslt/training/{lang}_bpe_codes.txt",
        )

    # Apply BPE to OOD data
    for lang in ["en", "nl"]:
        logger.info(f"Applying BPE to {lang} out of distribution test data")
        tokenizer.apply_bpe(
            input_path=f"local/data/test_ood/tokenized_test_ood.{lang}",
            output_path=f"local/data/test_ood/bpe_test_ood.{lang}",
            codes_path=f"local/data/training/{"de" if lang=="nl" else lang}_bpe_codes.txt",
        )


    logger.info("Build and save vocab")
    if not os.path.exists("local/iwslt/vocab_en.pkl") or not os.path.exists(
        "local/iwslt/vocab_de.pkl"
    ):
        build_and_save_vocab(
            train_en_path=constants.file_paths.bpe_train_en,
            train_de_path=constants.file_paths.bpe_train_de,
            min_freq=hyperparameters.vocab.token_min_freq,
            save_en_path="local/iwslt/vocab_en.pkl",
            save_de_path="local/iwslt/vocab_de.pkl",
        )
        logger.warning("Vocab files not found. Building vocab from training data.")
    en_vocab = load_vocab("local/iwslt/vocab_en.pkl")
    de_vocab = load_vocab("local/iwslt/vocab_de.pkl")
    logger.info(f"English vocab size: {len(en_vocab)}")
    logger.info(f"German vocab size: {len(de_vocab)}")
    assert 6600 <= len(en_vocab) <= 6700, f"Expected 6628 English vocab size, got {len(en_vocab)}"
    assert 8800 <= len(de_vocab) <= 8900, f"Expected 8844 German vocab size, got {len(de_vocab)}"

    logger.info("Create data loaders")
    training_loader = get_data_loader(
        src_file="local/iwslt/training/bpe_train.de",
        tgt_file="local/iwslt/training/bpe_train.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=hyperparameters.training.batch_size,
        add_bos_eos=True,
        shuffle=hyperparameters.training.shuffle,
        max_len=hyperparameters.transformer.max_len,
    )

    test_loader = get_data_loader(
        src_file="local/iwslt/test/bpe_test.de",
        tgt_file="local/iwslt/test/bpe_test.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=124,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    dev_loader = get_data_loader(
        src_file="local/iwslt/dev/bpe_dev.de",
        tgt_file="local/iwslt/dev/bpe_dev.en",
        src_vocab=de_vocab,
        tgt_vocab=en_vocab,
        batch_size=124,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )

    logger.info(f"Training set size: {len(training_loader.dataset)}")  # type: ignore
    logger.info(f"Dev set size: {len(dev_loader.dataset)}")  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating model")

    model: nn.Module = TransformerModel(
        src_vocab_size=len(de_vocab),
        tgt_vocab_size=len(en_vocab),
        d_model=hyperparameters.transformer.hidden_size,
        num_heads=hyperparameters.transformer.num_heads,
        d_ff=hyperparameters.transformer.encoder_ffn_embed_dim,
        num_encoder_layers=hyperparameters.transformer.num_hidden_layers,
        num_decoder_layers=hyperparameters.transformer.num_hidden_layers,
        dropout=hyperparameters.transformer.dropout,
        max_len=hyperparameters.transformer.max_len,
    )
    number_of_params = sum(p.numel() for p in model.parameters())
    print(f"Model number of parameters: {number_of_params/1e6:.2f}M")
    assert 34_000_000 <= number_of_params <= 35_000_000, f"Expected ~34.5M parameters, got {number_of_params/1e6:.2f}M"
    model.to(device)
    if torch.cuda.is_available() and torch_compile:
        logger.info("Compiling model with torch compile")
        model = torch.compile(model)  # type: ignore

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1,
        betas=hyperparameters.training.adam_betas,
        eps=hyperparameters.training.adam_eps,
    )

    def get_lr(step: int) -> float:
        d_model = hyperparameters.transformer.hidden_size
        warmup_steps = hyperparameters.training.learning_rate_warm_up_steps
        step = max(1, step) # Avoid division by zero
        out: float = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        return out

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=hyperparameters.training.label_smoothing, ignore_index=0
    )

    logger.info("Setting up weights and biases")
    wandb_mode: Literal["online", "disabled"] = "online" if use_wandb else "disabled"
    wandb.init(
        project="TransformerUQ",
        entity="sondresorbye-magson",
        config=hyperparameters.model_dump(),
        dir="local",
        mode=wandb_mode,
    )

    train(
        model,
        training_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        max_steps=hyperparameters.training.max_steps,
    )


if __name__ == "__main__":
    main()
