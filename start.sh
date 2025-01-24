#! /bin/bash
#SBATCH –-job-name=”WMT14”
#SBATCH -–time=00:10:00
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load Python/3.12.3-GCCcore-13.3.0
python -m venv venv
source venv/bin/activate
poetry install
python wat_zei_je_download_data.py > wat_zei_je_download_data.log
python main.py > run.log
