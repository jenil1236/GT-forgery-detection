# How to Run

## 1. Clone / Copy project to HPC
`
scp -r ./GT user@hpc-login:/home/user/GT
`

## 2. Create conda environment (first time only)
`
conda create -n gt_forgery python=3.10
conda activate gt_forgery
pip install -r requirements.txt
`

## 3. Run on HPC via SLURM
`
cd ~/GT
sbatch train.slurm
`

Monitor logs:
`
tail -f output/logs/gt_forgery_<JOBID>.out
`

## 4. Run locally (development / debugging)
`
conda activate gt_forgery
python src/train.py --resume
`

Skip already-done steps:
`
python src/train.py --resume --skip_download --skip_forge
python src/train.py --resume --skip_stage1   # skip encoder pretraining
python src/train.py --eval_only              # evaluation only
`

## 5. Check results
`
cat output/metrics/eval_report.txt
python -c "import json; r=json.load(open(chr(39)output/metrics/eval_metrics.json{chr(39)})); print(r[chr(39)seen{chr(39)][chr(39)auc{chr(39)])"
`

## Key flags
| Flag | Effect |
|------|--------|
| --resume | Auto-continue from latest checkpoint |
| --skip_download | Skip dataset download |
| --skip_forge | Skip forgery generation |
| --skip_stage1 | Skip encoder pretraining |
| --skip_stage2 | Skip MARL training |
| --eval_only | Run evaluation only |
| --device cpu | Force CPU (for debugging) |

## Expected output structure
`
output/
  logs/gt_forgery_<JOBID>.out   <- SLURM stdout
  logs/gt_forgery_<JOBID>.err   <- SLURM stderr
  checkpoints/
    best_stage1_visual.pt
    best_stage1_text.pt
    best_stage1_signal.pt
    best_stage2.pt
  metrics/
    run_YYYYMMDD_HHMMSS.jsonl   <- per-step training metrics
    eval_metrics.json           <- final evaluation
    eval_report.txt             <- human-readable summary
`
