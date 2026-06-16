# Agent loop progress — autoresearch/jun15 (A10G 24GB)
Started 2026-06-15 23:45 UTC | 12h cap -> auto-stop 2026-06-16 11:45 UTC
Baseline: DEPTH=8 BATCH=32, val_bpb=1.3230, peak 11.7GB

| # | idea | val_bpb | vs best | verdict |
|---|------|---------|---------|---------|
| 0 | baseline | 1.3230 | — | KEEP (best) |
| 1 | MATRIX_LR 0.05 | 1.2964 | -0.027 | KEEP (best) |
| 2 | MATRIX_LR 0.07 | 1.2738 | -0.023 | KEEP (best) |
| 3 | MATRIX_LR 0.10 | 1.2646 | -0.009 | KEEP (best) |
| 4 | MATRIX_LR 0.13 | 1.2792 | +0.015 | DISCARD (reverted to 0.10) |
| 5 | EMBEDDING_LR 0.8 | 1.2639 | -0.0006 | KEEP (best, marginal) |
| 6 | WARMUP_RATIO 0.1 | 1.3569 | +0.093 | DISCARD (reverted) |
| 7 | WINDOW SSLL | 1.2687 | +0.005 | DISCARD (reverted) |
| 8 | ADAM b1 0.9 | 1.2791 | +0.015 | DISCARD (reverted) |
| 9 | WEIGHT_DECAY 0.05 | 1.2724 | +0.008 | DISCARD (reverted) |
| 10 | UNEMBEDDING_LR 0.008 | 1.2479 | -0.016 | KEEP (new best) |
| 11 | UNEMBEDDING_LR 0.015 | 1.2775 | +0.030 | DISCARD (reverted to 0.008) |
| 12 | ASPECT_RATIO 48 | 1.1917 | -0.056 | KEEP (new best, 175 steps) |
| 13 | ASPECT_RATIO 40 | 1.1917 | -0.00003 | KEEP (tie, less VRAM) |
| 14 | DEPTH 10 | 1.3084 | +0.117 | DISCARD (reverted) |
| 15 | MATRIX_LR 0.13 @narrow | 1.1928 | +0.001 | DISCARD (reverted) |
| 15 | MATRIX_LR 0.13 @narrow | 1.1928 | +0.001 | DISCARD (reverted) |
| 16 | MQA n_kv_head=1 | 1.1982 | +0.006 | DISCARD (reverted) |
| 17 | MLP 6x | 1.2086 | +0.017 | DISCARD (reverted) |
| 18 | MLP 3x | 1.1821 | -0.010 | KEEP (new best, 191 steps) |
| 19 | MLP 2x | 1.1762 | -0.006 | KEEP (new best, 213 steps) |
| 20 | MLP 1x | 1.1762 | +0.00002 | DISCARD (tie, 2x optimal) |
