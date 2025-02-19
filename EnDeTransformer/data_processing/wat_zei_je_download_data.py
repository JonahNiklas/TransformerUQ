# %% [markdown]
# ## Download all the data we need for Wat zei je benchmarks
#
# Training
# - news-commentary-v13.de-en
# - wmt13-commoncrawl.de-en
# - wmt13-europarl.de-en
#
# Test
# - wmt14-newstest2014-de-en
#
# Test OOD
# - news-commentary-v14.nl-en

# %%
from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel
import requests
import tarfile
import gzip
import shutil
import os
import logging

logger = logging.getLogger(__name__)

data_directory = "./EnDeTransformer/local/data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

training_sets: List[Dict[str, str]] = [
    {
        "link": "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz",
        "subfolder": "training-parallel-nc-v13/",
        "de_file": "news-commentary-v13.de-en.de",
        "en_file": "news-commentary-v13.de-en.en",
    },
    {
        "link": "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        "subfolder": "",
        "de_file": "commoncrawl.de-en.de",
        "en_file": "commoncrawl.de-en.en",
    },
    {
        "link": "https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        "subfolder": "training/",
        "de_file": "europarl-v7.de-en.de",
        "en_file": "europarl-v7.de-en.en",
    },
]


dev_set = {
    "link": "https://www.statmt.org/wmt14/dev.tgz",
    "subfolder": "dev/",
    "de_file": "newstest2013.de",
    "en_file": "newstest2013.en",
}

test_set = {
    "link": "https://www.statmt.org/wmt14/test-full.tgz",
    "subfolder": "test-full/",
    "de_file": "newstest2014-deen-src.de.sgm",
    "en_file": "newstest2014-deen-src.en.sgm",
}

test_ood = {
    "link": "https://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-nl.tsv.gz",
    "subfolder": "",
    "file_name": "news-commentary-v14.en-nl.tsv",
    "file_extension": ".tsv",
    "en_file": "news-commentary-v14.en-nl.en",
    "nl_file": "news-commentary-v14.en-nl.nl",
}


# %%
def download_and_extract(
    url: str, de_file: str | None = None, en_file: str | None = None
) -> None:
    local_filename = url.split("/")[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    if local_filename.endswith(".tgz") or local_filename.endswith(".tar.gz"):
        with tarfile.open(local_filename, "r:gz") as tar:
            if de_file is None or en_file is None:
                tar.extractall(path=data_directory)
            else:
                for member in tar.getmembers():
                    if member.name.endswith(de_file) or member.name.endswith(en_file):
                        tar.extract(member, path=data_directory)

    if local_filename.endswith(".gz") and not local_filename.endswith(".tar.gz"):
        with gzip.open(local_filename, "rb") as f_in:
            with open(os.path.join(data_directory, local_filename[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    os.remove(local_filename)
    logger.info(f"Downloaded and extracted {local_filename}")


# Download and extract training sets
for dataset in training_sets:
    download_and_extract(dataset["link"], dataset["de_file"], dataset["en_file"])

# Download and extract dev set
download_and_extract(dev_set["link"], dev_set["de_file"], dev_set["en_file"])

# Download and extract test set
download_and_extract(test_set["link"], test_set["de_file"], test_set["en_file"])

# Download and extract ood test set
download_and_extract(test_ood["link"])

# %%
# convert test ood from tsv to txt
test_ood_file_path = os.path.join(data_directory, test_ood["file_name"])
if os.path.exists(test_ood_file_path):
    en_lines = []
    nl_lines = []
    with open(test_ood_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tab_split = line.split("\t")
            nl_lines.append(tab_split[-1])
            en_lines.append(
                " ".join(tab_split[:-1]) + "\n"
            )  # 32 lines have a tab in the english sentence that needs to be merged

    with open(
        os.path.join(data_directory, test_ood["en_file"]), "w", encoding="utf-8"
    ) as de_f:
        de_f.writelines(en_lines)
    with open(
        os.path.join(data_directory, test_ood["nl_file"]), "w", encoding="utf-8"
    ) as en_f:
        en_f.writelines(nl_lines)
    os.remove(test_ood_file_path)

# %%
from bs4 import BeautifulSoup


def sgm_to_txt(sgm_file: str) -> None:
    if not os.path.exists(sgm_file):
        return
    with open(sgm_file, "r", encoding="utf-8") as sgm:
        soup = BeautifulSoup(sgm, "html.parser")
        with open(sgm_file.replace(".sgm", ""), "w", encoding="utf-8") as txt:
            for seg in soup.find_all("seg"):
                txt.write(seg.get_text() + "\n")
    os.remove(sgm_file)


sgm_to_txt(os.path.join(data_directory, test_set["subfolder"], test_set["de_file"]))
sgm_to_txt(os.path.join(data_directory, test_set["subfolder"], test_set["en_file"]))
test_set["de_file"] = test_set["de_file"].replace(".sgm", "")
test_set["en_file"] = test_set["en_file"].replace(".sgm", "")


# %%
def merge_files(
    datasets: List[Dict[str, str]],
    subfolder: str,
    filename: str,
    second_lang: str = "de",
) -> None:

    target_folder = os.path.join(data_directory, subfolder)
    de_files = []
    en_files = []
    for dataset in datasets:
        de_file_path = os.path.join(
            data_directory, dataset["subfolder"], dataset[second_lang + "_file"]
        )
        en_file_path = os.path.join(
            data_directory, dataset["subfolder"], dataset["en_file"]
        )
        de_files.append(de_file_path)
        en_files.append(en_file_path)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    with open(
        os.path.join(target_folder, filename + "." + second_lang), "w", encoding="utf-8"
    ) as de_f:
        for de_file in de_files:
            with open(de_file, "r", encoding="utf-8") as f:
                de_f.write(f.read())

    with open(
        os.path.join(target_folder, filename + ".en"), "w", encoding="utf-8"
    ) as en_f:
        for en_file in en_files:
            with open(en_file, "r", encoding="utf-8") as f:
                en_f.write(f.read())

    for file in de_files + en_files:
        os.remove(file)
    print(f"Merged files into {filename}")


# %%
merge_files(training_sets, "training", "train")
merge_files([dev_set], "dev", "dev")
merge_files([test_set], "test", "test")
merge_files([test_ood], "test_ood", "test_ood", second_lang="nl")
# Remove empty directories in data_directory
for root, dirs, files in os.walk(data_directory, topdown=False):
    for name in dirs:
        dir_path = os.path.join(root, name)
        if not os.listdir(dir_path):
            os.rmdir(dir_path)

print("Data downloaded and merged successfully")

# %%
# Ensure test_ood is the same length as test
test_file_path = os.path.join(data_directory, "test", "test.en")
test_ood_en_file_path = os.path.join(data_directory, "test_ood", "test_ood.en")
test_ood_nl_file_path = os.path.join(data_directory, "test_ood", "test_ood.nl")

with open(test_file_path, "r", encoding="utf-8") as test_f:
    test_lines = test_f.readlines()

with open(test_ood_en_file_path, "r", encoding="utf-8") as test_ood_en_f:
    test_ood_en_lines = test_ood_en_f.readlines()

with open(test_ood_nl_file_path, "r", encoding="utf-8") as test_ood_nl_f:
    test_ood_nl_lines = test_ood_nl_f.readlines()

min_length = min(len(test_lines), len(test_ood_en_lines), len(test_ood_nl_lines))

test_ood_en_lines_short = test_ood_en_lines[:min_length]
test_ood_nl_lines_short = test_ood_nl_lines[:min_length]

with open(test_ood_en_file_path, "w", encoding="utf-8") as test_ood_en_f:
    test_ood_en_f.writelines(test_ood_en_lines_short)

with open(test_ood_nl_file_path, "w", encoding="utf-8") as test_ood_nl_f:
    test_ood_nl_f.writelines(test_ood_nl_lines_short)

# Write the old test_ood files as test_ood_long.en and test_ood_long.nl
test_ood_long_en_file_path = os.path.join(data_directory, "test_ood", "test_ood_long.en")
test_ood_long_nl_file_path = os.path.join(data_directory, "test_ood", "test_ood_long.nl")

with open(test_ood_long_en_file_path, "w", encoding="utf-8") as test_ood_long_en_f:
    test_ood_long_en_f.writelines(test_ood_en_lines)

with open(test_ood_long_nl_file_path, "w", encoding="utf-8") as test_ood_long_nl_f:
    test_ood_long_nl_f.writelines(test_ood_nl_lines)

print("Test OOD set adjusted to the same length as test set")
# %%
