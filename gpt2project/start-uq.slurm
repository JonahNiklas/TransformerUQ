#! /bin/sh
#SBATCH --job-name="reduced bayesgpt dropout ood"
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # 2 CPU cores
#SBATCH --mem=64G              # 5 gigabytes memory
#SBATCH --gres=gpu:h100:1
#SBATCH --output=gpt2project/local/gpt2_log/bayesgpt_reduced_dropout_ood.txt

export TRANSFORMER_IMPL=transformer

module load Python/3.12.3-GCCcore-13.3.0
python -m venv venv
source venv/bin/activate
poetry install
python -m gpt2project.uq.run_all_uq
