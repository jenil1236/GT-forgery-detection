# Quick Review - GT Forgery Detection (1-Page Cheat Sheet)

## What This Project Does
Reference-conditioned document forgery detection. Given:
- A **reference image** (authoritative, always real, same document type)
- A **test image** (real or forged)

Output: **genuine / forged** verdict.
Trains on some document types, generalises to **completely unseen types** at inference.

## System Architecture

`
Reference Image ---> Visual Expert  ---> belief q1 ---+
                ---> Text Expert    ---> belief q2 ---+---> Judge ---> declare
                ---> Signal Expert  ---> belief q3 ---+
Test Image ----> (same 3 experts)
`

Judge: log-odds accumulator lambda_t += log(q/(1-q)); b_t = sigmoid(lambda_t)
Episode: max 4 steps - judge queries experts sequentially, then declares.

## Datasets
- CORD-v2 (receipts, ~800 images): train doc type
- RVL-CDIP 6 classes (~3000 images): train doc types
- RVL-CDIP 2 classes (~600 images): zero-shot test (never seen during training)
- Forgeries: programmatically generated (text replace, copy-move, noise injection)

## Training Stages
- Stage 1 (30 epochs): Encoder pretraining with InfoNCE + ContrastiveAlignment
- Stage 2 (50 epochs): MAPPO cooperative MARL with alternating judge/expert schedule

Stage 2 alternating: 5 epochs train judge (freeze experts) -> 5 epochs train experts (freeze judge) -> repeat.
This prevents private code collapse.

## Key Metrics
- AUC: overall detection quality
- EER: equal error rate (threshold-free)
- ECE: calibration quality
- Retention ratio > 0.6: experts learned real features (not private code)
- Shapley values: per-expert contribution (exact, n=3)

## File Map
`
src/train.py           <- single entry point
src/data/              <- download, forge, dataset, transforms
src/models/            <- encoders, agents, judge
src/training/          <- pretrain, marl, losses, shapley
src/evaluation/        <- metrics, evaluate
output/checkpoints/    <- .pt checkpoints
output/metrics/        <- eval_metrics.json, eval_report.txt
data/models_cache/     <- HuggingFace cache (stays in project)
`

## Run Command
`ash
python src/train.py --resume
`
