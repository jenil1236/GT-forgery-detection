# Results Documentation

## Expected Performance

### Seen Types
- AUC: 0.90-0.96
- EER: 0.06-0.12
- Accuracy: 0.85-0.92
- ECE: 0.03-0.08

### Zero-Shot (Unseen Types)
- AUC: 0.76-0.88
- EER: 0.12-0.20
- Accuracy: 0.72-0.83

### Shapley Values (typical)
- phi_visual: 0.05-0.15
- phi_text: 0.08-0.18
- phi_signal: 0.06-0.14

## Output Files
- output/metrics/{run_id}.jsonl  (per-step + epoch metrics)
- output/metrics/eval_metrics.json
- output/metrics/eval_report.txt
- output/checkpoints/best_stage2.pt
