from pydantic import BaseModel


class FilePaths(BaseModel):
    src_vocab: str = "local/iwslt/vocab_de.pkl"
    tgt_vocab: str = "local/iwslt/vocab_en.pkl"

    train_en: str = "local/iwslt/training/train.en"
    train_de: str = "local/iwslt/training/train.de"
    test_en: str = "local/iwslt/test/test.en"
    test_de: str = "local/iwslt/test/test.de"
    ood_en: str = "local/data/test_ood/test_ood.en"
    ood_nl: str = "local/data/test_ood/test_ood.nl"

    tokenized_train_en: str = "local/iwslt/training/tokenized_train.en"
    tokenized_train_de: str = "local/iwslt/training/tokenized_train.de"
    tokenized_test_en: str = "local/iwslt/test/tokenized_test.en"
    tokenized_test_de: str = "local/iwslt/test/tokenized_test.de"
    tokenized_train_merged: str = "local/iwslt/training/tokenized_train_merged.txt"
    tokenized_ood_en: str = "local/data/test_ood/tokenized_test_ood.en"
    tokenized_ood_nl: str = "local/data/test_ood/tokenized_test_ood.nl"

    dev_en: str = "local/iwslt/dev/dev.en"
    dev_de: str = "local/iwslt/dev/dev.de"
    tokenized_dev_en: str = "local/iwslt/dev/tokenized_dev.en"
    tokenized_dev_de: str = "local/iwslt/dev/tokenized_dev.de"

    bpe_train_en: str = "local/iwslt/training/bpe_train.en"
    bpe_train_de: str = "local/iwslt/training/bpe_train.de"
    bpe_test_en: str = "local/iwslt/test/bpe_test.en"
    bpe_test_de: str = "local/iwslt/test/bpe_test.de"
    bpe_test_ood_en: str = "local/data/test_ood/bpe_test_ood.en"
    bpe_test_ood_nl: str = "local/data/test_ood/bpe_test_ood.nl"
    bpe_dev_en: str = "local/iwslt/dev/bpe_dev.en"
    bpe_dev_de: str = "local/iwslt/dev/bpe_dev.de"


class Constants(BaseModel):
    file_paths: FilePaths = FilePaths()


constants = Constants()
