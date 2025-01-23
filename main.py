import torch
from models.transformer import Transformer
from tokenizer import ParallelCorpusTokenizer
import train
import logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    logger.info("Tokenize data")
    tokenizer = ParallelCorpusTokenizer()
    tokenizer.tokenize_files(        
        train_en_path='local/data/training/train.de',
        train_de_path='local/data/training/train.en',
        test_en_path='local/data/test/test.de',
        test_de_path='local/data/test/test.en',
        output_train_en='local/data/training/tokenized_train.de',
        output_train_de='local/data/training/tokenized_train.en',
        output_test_en='local/data/test/tokenized_test.de',
        output_test_de='local/data/test/tokenized_test.en'
    )
    logger.info("Learning BPE codes")
    tokenizer.learn_bpe(
        input_path='local/data/training/tokenized_train_en.txt',
        output_codes_path='local/data/training/en_bpe_codes.txt'
    )
    logger.info("Applying BPE")
    tokenizer.apply_bpe(
        input_path='local/data/training/tokenized_train_en.txt',
        output_path='local/data/training/bpe_train_en.txt',
        codes_path='local/data/training/en_bpe_codes.txt'
    )

    # Convert tokenized data to tensors
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating model")
    model = Transformer()
    model.to(device)
    model = model.compile()

    train(model, training_data, test_data)

if __name__ == "__main__":
    main()