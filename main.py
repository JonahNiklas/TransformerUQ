import math
import os
from typing import Literal
from sympy import hyper
import torch
import wandb
from constants import constants
from data_processing.dataloader import get_data_loader, load_vocab
from hyperparameters import hyperparameters
from models.transformer import Transformer
from models.transformer_pytorch import TransformerPyTorch
from data_processing.tokenizer import ParallelCorpusTokenizer
from train import train
import logging
from torch.utils.data import DataLoader
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
        test_en_path=constants.file_paths.test_en,
        test_de_path=constants.file_paths.test_de,
        output_train_en=constants.file_paths.tokenized_train_en,
        output_train_de=constants.file_paths.tokenized_train_de,
        output_test_en=constants.file_paths.tokenized_test_en,
        output_test_de=constants.file_paths.tokenized_test_de,
        test_ood_en_path=constants.file_paths.ood_en,
        test_ood_nl_path=constants.file_paths.ood_nl,
        output_test_ood_en=constants.file_paths.tokenized_ood_en,
        output_test_ood_nl=constants.file_paths.tokenized_ood_nl,
    )

    logger.info("Merge the tokenized training data")
    if not os.path.exists(constants.file_paths.tokenized_train_merged):
        with open(constants.file_paths.tokenized_train_en, "r", encoding="utf-8") as f_en, \
            open(constants.file_paths.tokenized_train_de, "r", encoding="utf-8") as f_de, \
            open(constants.file_paths.tokenized_train_merged, "w", encoding="utf-8") as f_out:
            for line in f_en:
                f_out.write(line)
            for line in f_de:
                f_out.write(line)

    logger.info("Learn a single set of BPE codes from merged data")
    shared_bpe_codes = "local/data/training/shared_bpe_codes.txt"
    tokenizer.learn_bpe(
        input_path=constants.file_paths.tokenized_train_merged,
        output_codes_path=shared_bpe_codes,
    )

    logger.info("Apply BPE to each language using the single shared code")
    for lang in ["en", "de"]:
        logger.info(f"Applying BPE to training data for {lang}")
        tokenizer.apply_bpe(
            input_path=f"local/data/training/tokenized_train.{lang}",
            output_path=f"local/data/training/bpe_train.{lang}",
            codes_path=shared_bpe_codes,
        )
        logger.info(f"Applying BPE to test data for {lang}")
        tokenizer.apply_bpe(
            input_path=f"local/data/test/tokenized_test.{lang}",
            output_path=f"local/data/test/bpe_test.{lang}",
            codes_path=shared_bpe_codes,
        )

    # Apply BPE to OOD data
    for lang in ["en", "nl"]:
        logger.info(f"Applying BPE to {lang} out of distribution test data")
        tokenizer.apply_bpe(
            input_path=f"local/data/test_ood/tokenized_test_ood.{lang}",
            output_path=f"local/data/test_ood/bpe_test_ood.{lang}",
            codes_path=shared_bpe_codes,
        )

    logger.info("Build and save vocab")
    if not os.path.exists(constants.file_paths.vocab):
        build_and_save_vocab(
            train_en_path="local/data/training/bpe_train.en",
            train_de_path="local/data/training/bpe_train.de",
            min_freq=hyperparameters.vocab.token_min_freq,
            save_path=constants.file_paths.vocab,
        )
        logger.warning("Shared vocab file not found. Building vocab from training data.")

    shared_vocab = load_vocab(constants.file_paths.vocab)
    logger.info(f"Shared vocab size: {len(shared_vocab)}")

    logger.info("Create data loaders")
    training_loader = get_data_loader(
        src_file="local/data/training/bpe_train.de",
        tgt_file="local/data/training/bpe_train.en",
        vocab=shared_vocab,
        batch_size=hyperparameters.training.batch_size,
        add_bos_eos=True,
        shuffle=hyperparameters.training.shuffle,
        max_len=hyperparameters.transformer.max_len,
    )

    test_loader = get_data_loader(
        src_file="local/data/test/bpe_test.de",
        tgt_file="local/data/test/bpe_test.en",
        vocab=shared_vocab,
        batch_size=124,
        add_bos_eos=True,
        shuffle=False,
        max_len=hyperparameters.transformer.max_len,
    )
    logger.info(f"Training set size: {len(training_loader.dataset)}")  # type: ignore
    logger.info(f"Test set size: {len(test_loader.dataset)}")  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating model")

    # model: nn.Module = Transformer(
    #     vocab_size=len(shared_vocab),
    #     d_model=hyperparameters.transformer.hidden_size,
    #     num_heads=hyperparameters.transformer.num_heads,
    #     d_ff=hyperparameters.transformer.encoder_ffn_embed_dim,
    #     num_encoder_layers=hyperparameters.transformer.num_hidden_layers,
    #     num_decoder_layers=hyperparameters.transformer.num_hidden_layers,
    #     dropout=hyperparameters.transformer.dropout,
    #     max_len=hyperparameters.transformer.max_len,
    # )
    model: nn.Module = TransformerPyTorch(
        vocab_size=len(shared_vocab),
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
        validate_every=hyperparameters.training.validate_every,
    )


if __name__ == "__main__":
    main()
