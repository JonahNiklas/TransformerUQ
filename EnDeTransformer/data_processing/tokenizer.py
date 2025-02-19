from __future__ import annotations

import multiprocessing
import os
from functools import partial
from venv import logger

import sacremoses
import subword_nmt.apply_bpe
import subword_nmt.learn_bpe
from tqdm import tqdm

from EnDeTransformer.hyperparameters import hyperparameters


class ParallelCorpusTokenizer:
    def __init__(self, num_processes: int | None =None, chunksize: int=1000) -> None:
        """
        :param num_processes: Number of processes to use for tokenization.
                              Defaults to all available CPU cores.
        :param chunksize: Number of lines to send to each worker process at a time.
        """
        self.en_tokenizer = sacremoses.MosesTokenizer(lang="en")
        self.de_tokenizer = sacremoses.MosesTokenizer(lang="de")
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.chunksize = chunksize

    def _tokenize_line(self, line: str, tokenizer: sacremoses.MosesTokenizer) -> str:
        """Tokenize a single line (helper for multiprocessing)."""
        return " ".join(tokenizer.tokenize(line.strip())) + "\n"

    def tokenize_file(self, input_path: str, output_path: str, lang: str ="en") -> None:
        """
        Tokenize a file line by line using multiprocessing.

        :param input_path: Path to the input file.
        :param output_path: Path to save the tokenized output.
        :param lang: Language identifier ("en" or "de").
        """

        if os.path.exists(output_path):
            logger.warning(f"Output file {output_path} already exists. Skipping tokenization.")
            return

        tokenizer = self.en_tokenizer if lang == "en" else self.de_tokenizer
        tokenize_func = partial(self._tokenize_line, tokenizer=tokenizer)

        # Get total lines for tqdm progress bar
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        with open(input_path, "r", encoding="utf-8") as input_file, open(
            output_path, "w", encoding="utf-8"
        ) as output_file:
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                # imap returns results one by one in order, while processing in parallel
                for tokenized_line in tqdm(
                    pool.imap(tokenize_func, input_file, chunksize=self.chunksize),
                    total=total_lines,
                    desc=f"Tokenizing {os.path.basename(input_path)}",
                ):
                    output_file.write(tokenized_line)

    def tokenize_files(
        self,
        train_en_path: str,
        train_de_path: str,
        dev_en_path: str,
        dev_de_path: str,
        test_en_path: str,
        test_de_path: str,
        output_train_en: str,
        output_train_de: str,
        output_dev_en: str,
        output_dev_de: str,
        output_test_en: str,
        output_test_de: str,
        test_ood_en_path: str,
        test_ood_nl_path: str,
        output_test_ood_en: str,
        output_test_ood_nl: str,
    ) -> None:
        """
        Tokenize training and test files.

        :param train_en_path: Path to the English training file.
        :param train_de_path: Path to the German training file.
        :param dev_en_path: Path to the English dev file.
        :param dev_de_path: Path to the German dev file.
        :param test_en_path: Path to the English test file.
        :param test_de_path: Path to the German test file.
        :param output_train_en: Path to save the tokenized English training file.
        :param output_train_de: Path to save the tokenized German training file.
        :param output_dev_en: Path to save the tokenized English dev file.
        :param output_dev_de: Path to save the tokenized German dev file.
        :param output_test_en: Path to save the tokenized English test file.
        :param output_test_de: Path to save the tokenized German test file.
        """
        # Tokenize training files
        self.tokenize_file(train_en_path, output_train_en, "en")
        self.tokenize_file(train_de_path, output_train_de, "de")

        # Tokenize dev files
        self.tokenize_file(dev_en_path, output_dev_en, "en")
        self.tokenize_file(dev_de_path, output_dev_de, "de")

        # Tokenize test files
        self.tokenize_file(test_en_path, output_test_en, "en")
        self.tokenize_file(test_de_path, output_test_de, "de")

        # Tokenize out-of-domain test files
        self.tokenize_file(test_ood_en_path, output_test_ood_en, "en")
        self.tokenize_file(test_ood_nl_path, output_test_ood_nl, "de")

    def learn_bpe(self, input_path: str, output_codes_path: str) -> None:
        """
        Learn BPE codes from a tokenized file.

        :param input_file: Path to the tokenized file (usually a concatenation of training data).
        :param output_codes_path: Where to save the learned BPE codes.
        :param num_symbols: Number of BPE symbols to learn.
        """

        if os.path.exists(output_codes_path):
            logger.warning(f"Output file {output_codes_path} already exists. Skipping BPE learning.")
            return

        with open(input_path, "r", encoding="utf-8") as infile, open(
            output_codes_path, "w", encoding="utf-8"
        ) as outfile:
            subword_nmt.learn_bpe.learn_bpe(infile, outfile, num_symbols=hyperparameters.vocab.bpe_num_symbols)

    def _apply_bpe_line(self, line: str, bpe: subword_nmt.apply_bpe.BPE) -> str:
        """Apply BPE to a single line."""
        bpe_line: str = bpe.process_line(line.strip())
        return bpe_line

    def apply_bpe(self, input_path: str, output_path: str, codes_path: str) -> None:
        """
        Apply BPE to a tokenized corpus line by line, using multiprocessing.
        """
        if os.path.exists(output_path):
            logger.warning(f"Output file {output_path} already exists. Skipping BPE application.")
            return

        # Load the BPE codes
        with open(codes_path, "r", encoding="utf-8") as codes_file:
            bpe = subword_nmt.apply_bpe.BPE(codes_file)

        # Count lines for progress bar
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        with open(input_path, "r", encoding="utf-8") as input_file, open(
            output_path, "w", encoding="utf-8"
        ) as output_file, multiprocessing.Pool(processes=self.num_processes) as pool:
            apply_bpe_func = partial(self._apply_bpe_line, bpe=bpe)
            for bpe_line in tqdm(
                pool.imap(apply_bpe_func, input_file, chunksize=self.chunksize),
                total=total_lines,
                desc=f"Applying BPE to {os.path.basename(input_path)}",
            ):
                output_file.write(bpe_line + "\n")
