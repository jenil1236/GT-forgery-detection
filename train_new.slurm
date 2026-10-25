#!/bin/bash
#SBATCH --job-name=xyz-qformer
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/tanmoyhazra/xyz/output_new/logs/qformer_%j.out
#SBATCH --error=/home/tanmoyhazra/xyz/output_new/logs/qformer_%j.err

echo "=== Q-FORMER TRAINING START: $(date) ==="
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"

module load anaconda3-2024.2
module load cuda-12.8
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate gt_forgery

set -eo pipefail

cd /home/tanmoyhazra/xyz || { echo "cd failed"; exit 1; }
export PYTHONPATH=/home/tanmoyhazra/xyz:$PYTHONPATH

mkdir -p output_new/logs output_new/checkpoints output_new/metrics output_new/errors

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

export HF_HOME=/home/tanmoyhazra/xyz/data/models_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 train_new.py --resume --project_root /home/tanmoyhazra/xyz

echo "Training finished: $(date)"
nvidia-smi
