#!/bin/bash

#SBATCH --job-name=gt-forgery
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # 16 CPU workers for DataLoader
#SBATCH --mem=128G               # H100 nodes have 512GB+ RAM
#SBATCH --gres=gpu:h100:1        # request one H100 -- adjust gres name to your cluster
#SBATCH --time=12:00:00          # realistic: ~6-9h on H100 (see time estimates)
#SBATCH --output=/home/%u/GT/output/logs/gt_forgery_%j.out
#SBATCH --error=/home/%u/GT/output/logs/gt_forgery_%j.err

set -euo pipefail

cd "$HOME/GT"

# Create output dirs if not present
mkdir -p output/logs output/checkpoints output/metrics
mkdir -p data/models_cache data/raw data/processed data/features_cache

# Load Conda
eval "$(/apps/compilers/anaconda3-2024.06/bin/conda shell.bash hook)"
conda activate gt_forgery

echo "========================================"
echo "GT Forgery Detection — Full Pipeline"
echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $SLURM_JOB_NAME"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "========================================"

echo ""
echo "Python:"
python3 --version

echo ""
echo "PyTorch + GPU:"
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('VRAM:', round(props.total_memory / 1e9, 1), 'GB')
    print('Compute capability:', props.major, props.minor)
"

echo ""
echo "GPU status:"
nvidia-smi

echo ""
echo "========================================"
echo "Starting training pipeline"
echo "========================================"

# HF_HOME: keep all model/dataset downloads inside the project
export HF_HOME=$HOME/GT/data/models_cache
export TRANSFORMERS_CACHE=$HOME/GT/data/models_cache/transformers
export HF_DATASETS_CACHE=$HOME/GT/data/models_cache/datasets

# Pass --resume to auto-continue from last checkpoint if it exists.
# On a fresh run the flag is harmless (no checkpoint found -> starts fresh).
# H100-specific env vars
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python3 src/train.py --resume --project_root "$HOME/GT"

echo ""
echo "========================================"
echo "Training finished"
echo "End time: $(date)"
echo "========================================"

echo ""
echo "Final GPU status:"
nvidia-smi