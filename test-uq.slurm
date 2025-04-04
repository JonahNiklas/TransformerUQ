#! /bin/sh
#SBATCH --job-name="iwslt bayesformer eval"
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # 2 CPU cores
#SBATCH --mem=64G              # 5 gigabytes memory
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/iwslt_bayesformer_eval_20250331.txt

module load Python/3.12.3-GCCcore-13.3.0
source venv/bin/activate
poetry install
python -m uq.test_model_uq
