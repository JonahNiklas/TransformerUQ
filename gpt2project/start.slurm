#! /bin/sh
#SBATCH --job-name="gpt2"
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-20:00:00     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # 2 CPU cores
#SBATCH --mem=64G              # 5 gigabytes memory
#SBATCH --gres=gpu:a100:8
#SBATCH --output=gpt2project/local/gpt2_log/output.txt

export TRANSFORMER_IMPL=transformer

module load Python/3.12.3-GCCcore-13.3.0
python -m venv venv
source venv/bin/activate
poetry install
torchrun --standalone --nproc_per_node=gpu -m gpt2project.train
