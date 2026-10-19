#!/bin/bash
#SBATCH --job-name=xyz-train
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/tanmoyhazra/xyz/output/logs/gt_forgery_%j.out
#SBATCH --error=/home/tanmoyhazra/xyz/output/logs/gt_forgery_%j.err

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID | CPUs: $SLURM_CPUS_PER_TASK | Start: $(date)"

# ---- Modules + conda (same pattern as the working llama job) ----
module load anaconda3-2024.2
module load cuda-12.8
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate gt_forgery

# Strict mode goes AFTER conda activation (conda's activate scripts break under set -u)
set -eo pipefail

cd /home/tanmoyhazra/xyz || { echo "cd failed"; exit 1; }
export PYTHONPATH=/home/tanmoyhazra/xyz:$PYTHONPATH

mkdir -p output/logs output/checkpoints output/metrics
mkdir -p data/models_cache data/raw data/processed data/features_cache

# ---- Sanity checks ----
echo "Python: $(which python3) ($(python3 --version))"
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU:', p.name, '| VRAM:', round(p.total_memory/1e9,1), 'GB')
"
nvidia-smi

# ---- Environment ----
export HF_HOME=/home/tanmoyhazra/xyz/data/models_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Train (--resume is harmless on a fresh run) ----
python3 src/train.py --resume --project_root /home/tanmoyhazra/xyz

echo "Training finished: $(date)"
nvidia-smi