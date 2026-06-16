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
