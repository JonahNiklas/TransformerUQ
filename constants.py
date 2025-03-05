from pydantic import BaseModel
class FilePaths(BaseModel):
    vocab: str = "local/vocab_shared.pkl"
    train_en: str = "local/data/training/train.en"
    train_de: str = "local/data/training/train.de"
    test_en: str = "local/data/test/test.en"
    test_de: str = "local/data/test/test.de"
    ood_en: str = "local/data/test_ood/test_ood.en"
    ood_nl: str = "local/data/test_ood/test_ood.nl"

    tokenized_train_en: str = "local/data/training/tokenized_train.en"
    tokenized_train_de: str = "local/data/training/tokenized_train.de"
    tokenized_test_en: str = "local/data/test/tokenized_test.en"
    tokenized_test_de: str = "local/data/test/tokenized_test.de"
    tokenized_train_merged: str = "local/data/training/tokenized_train_merged.txt"
    tokenized_ood_en: str = "local/data/test_ood/tokenized_test_ood.en"
    tokenized_ood_nl: str = "local/data/test_ood/tokenized_test_ood.nl"

    dev_en: str = "local/data/dev/dev.en"
    dev_de: str = "local/data/dev/dev.de"
    tokenized_dev_en: str = "local/data/dev/tokenized_dev.en"
    tokenized_dev_de: str = "local/data/dev/tokenized_dev.de"

    bpe_train_en: str = "local/data/training/bpe_train.en"
    bpe_train_de: str = "local/data/training/bpe_train.de"
    bpe_test_en: str = "local/data/test/bpe_test.en"
    bpe_test_de: str = "local/data/test/bpe_test.de"
    bpe_test_ood_en: str = "local/data/test_ood/bpe_test_ood.en"
    bpe_test_ood_nl: str = "local/data/test_ood/bpe_test_ood.nl"
    bpe_dev_en: str = "local/data/dev/bpe_dev.en"
    bpe_dev_de: str = "local/data/dev/bpe_dev.de"


class Constants(BaseModel):
    file_paths: FilePaths = FilePaths()


constants = Constants()
