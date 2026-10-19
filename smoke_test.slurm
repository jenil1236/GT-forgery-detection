#!/bin/bash
#SBATCH --job-name=xyz-smoke
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/tanmoyhazra/xyz/output/logs/smoke_%j.out
#SBATCH --error=/home/tanmoyhazra/xyz/output/logs/smoke_%j.err

# ── identical setup to train.slurm (which is known to work) ────────────────
echo "=== SMOKE TEST START: $(date) ==="
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"

module load anaconda3-2024.2
module load cuda-12.8
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate gt_forgery

# strict mode AFTER conda (same as train.slurm)
set -eo pipefail

cd /home/tanmoyhazra/xyz || { echo "cd failed"; exit 1; }
export PYTHONPATH=/home/tanmoyhazra/xyz:$PYTHONPATH

mkdir -p output/logs output/checkpoints output/metrics output/errors

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

export HF_HOME=/home/tanmoyhazra/xyz/
data/models_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── smoke test run ──────────────────────────────────────────────────────────
set +e
python3 src/train.py --smoke_test --project_root /home/tanmoyhazra/xyz
EXIT_CODE=$?
set -e

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "================================================"
    echo "  SMOKE TEST PASSED - safe to run train.slurm"
    echo "================================================"
else
    echo "================================================"
    echo "  SMOKE TEST FAILED (exit code: $EXIT_CODE)"
    echo "  Check: output/errors/ for traceback files"
    echo "================================================"
    echo ""
    echo "--- output/errors/ ---"
    ls -lh output/errors/ 2>/dev/null || echo "(empty)"
    LATEST=$(ls -t output/errors/*.txt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "--- $LATEST ---"
        tail -40 "$LATEST"
    fi
    exit $EXIT_CODE
fi

echo "Finished: $(date)"
