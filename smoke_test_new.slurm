#!/bin/bash
#SBATCH --job-name=xyz-qf-smoke
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/tanmoyhazra/xyz/output_new/logs/smoke_qf_%j.out
#SBATCH --error=/home/tanmoyhazra/xyz/output_new/logs/smoke_qf_%j.err

echo "=== Q-FORMER SMOKE TEST START: $(date) ==="
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
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU:', p.name, '| VRAM:', round(p.total_memory/1e9,1), 'GB')
"
nvidia-smi

export HF_HOME=/home/tanmoyhazra/xyz/data/models_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
python3 train_new.py --smoke_test --skip_stage1 --project_root /home/tanmoyhazra/xyz
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================"
    echo "  SMOKE TEST PASSED - safe to run train_new.slurm"
    echo "================================================"
else
    echo "================================================"
    echo "  SMOKE TEST FAILED (exit $EXIT_CODE)"
    echo "  Check: output_new/errors/"
    echo "================================================"
    ls -lh output_new/errors/ 2>/dev/null || echo "(empty)"
    LATEST=$(ls -t output_new/errors/*.txt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && tail -40 "$LATEST"
    exit $EXIT_CODE
fi
echo "Finished: $(date)"
