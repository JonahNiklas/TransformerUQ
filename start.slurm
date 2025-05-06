#! /bin/sh
#SBATCH --job-name="concrete dropout wmt14"
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-55:00:00     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # 2 CPU cores
#SBATCH --mem=64G              # 5 gigabytes memory
#SBATCH --gres=gpu:a100:1
#SBATCH --output=TransformerUQ/concrete_dropout_wmt14_log.txt

module load Python/3.12.3-GCCcore-13.3.0
python -m venv venv
source venv/bin/activate
poetry install
python data_processing/wat_zei_je_download_data.py
python main.py
