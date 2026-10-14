# Training Documentation

## Setup
`
conda create -n gt_forgery python=3.10
conda activate gt_forgery
pip install -r requirements.txt
`

## Stage 1 Key Params
- Epochs: 30, Batch: 64, LR: 1e-4
- InfoNCE temp: 0.07, Triplet margin: 0.5
- LR schedule: CosineAnnealing

## Stage 2 Key Params
- Total epochs: 50 (5 cycles of 5+5)
- PPO clip: 0.2, GAE lambda: 0.95
- Query cost: 0.1, Actor LR: 3e-4, Critic LR: 1e-3

## Alternating Schedule
Epochs 0-4: train judge, freeze experts
Epochs 5-9: train experts, freeze judge
Repeat 5 cycles.

## Resume
`
python src/train.py --resume
`
