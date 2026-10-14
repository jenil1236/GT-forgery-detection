# Dataset Documentation

## CORD-v2
- Source: naver-clova-ix/cord-v2 (HuggingFace)
- Content: Restaurant receipts with structured key-value annotations
- Size used: ~800 train, ~100 test images

## RVL-CDIP
- Source: aharley/rvl_cdip (HuggingFace)
- Train classes (6): letter, form, invoice, budget, memo, resume
- Zero-shot classes (2): advertisement, scientific_report (NEVER seen during training)
- Size per class: 500 train, 300 zero-shot

## Forgery Types
1. TextReplace: overwrite text regions with random digit strings
2. CopyMove: copy patch, paste with slight transform
3. NoiseInject: JPEG recompression quality 10-40 + Gaussian noise
