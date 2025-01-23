import torch
from models.transformer import Transformer
import train
import logging
from fairseq.data import Dictionary, data_utils, LanguagePairDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_data():
    # Load data
    data_dir = 'local/data-bin/iwslt14.tokenized.de-en'
    src_dict = Dictionary.load(f'{data_dir}/dict.de.txt')
    tgt_dict = Dictionary.load(f'{data_dir}/dict.en.txt')
    dataset = LanguagePairDataset(
        src=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.de', src_dict),
        src_sizes=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.de', src_dict).sizes,
        src_dict=src_dict,
        tgt=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.en', tgt_dict),
        tgt_sizes=data_utils.load_indexed_dataset(f'{data_dir}/train.de-en.en', tgt_dict).sizes,
        tgt_dict=tgt_dict,
    )
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    # Load test data
    test_dataset = LanguagePairDataset(
        src=data_utils.load_indexed_dataset(f'{data_dir}/test.de-en.de', src_dict),
        src_sizes=data_utils.load_indexed_dataset(f'{data_dir}/test.de-en.de', src_dict).sizes,
        src_dict=src_dict,
        tgt=data_utils.load_indexed_dataset(f'{data_dir}/test.de-en.en', tgt_dict),
        tgt_sizes=data_utils.load_indexed_dataset(f'{data_dir}/test.de-en.en', tgt_dict).sizes,
        tgt_dict=tgt_dict,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    return dataloader, test_dataloader

def main():
    logger.info("Loading data")
    training_data, test_data = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Creating model")
    model = Transformer()
    model.to(device)
    model = model.compile()

    train(model, training_data, test_data)

if __name__ == "__main__":
    main()