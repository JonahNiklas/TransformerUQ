import torch
from models.transformer import Transformer
from tokenizer import ParallelCorpusTokenizer
from tokens_to_tensor import tokens_to_tensor
import train
import logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

    logger.info("Convert tokenized data to tensors")
    training_loader, test_loader = tokens_to_tensor(
        train_en_path="local/data/training/bpe_train.en",
        train_de_path="local/data/training/bpe_train.de",
        test_en_path="local/data/test/bpe_test.en",
        test_de_path="local/data/test/bpe_test.de",
        batch_size=32,
        min_freq=1,
        add_bos_eos=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating model")
    model = Transformer()
    model.to(device)
    model = model.compile()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    train(
        model,
        training_loader,
        test_loader,
        optimizer,
        criterion,
        max_updates=500_000,
    )


if __name__ == "__main__":
    main()
