# autoresearch v2 — A10G-tuned

This is an updated version of [karpathy/autoresearch](https://github.com/karpathy/autoresearch),
re-baselined for a single **NVIDIA A10G (24 GB)** and seeded with the results of a 25-experiment
autonomous research loop. **It is the v2 of the autoresearcher**: instead of starting from upstream's
H100-sized defaults, v2 starts from the best configuration that loop found.

## What's different from upstream / v1

| | Upstream (v1) | v2 (this branch) |
|---|---|---|
| Target GPU | H100 80 GB | A10G 24 GB (self-serve) |
| Model shape | wide (`ASPECT_RATIO=64`), MLP 4× | lean (`ASPECT_RATIO=32`), MLP 2× |
| Key LRs | `MATRIX_LR=0.04`, `UNEMBEDDING_LR=0.004` | `MATRIX_LR=0.10`, `UNEMBEDDING_LR=0.012` |
| `DEVICE_BATCH_SIZE` | 128 | 32 |
| Steps in 5-min budget | ~125 | ~335 |
| **val_bpb** | **1.3230** | **1.1749 (−11.2%)** |
| Peak VRAM | 11.7 GB | 5.1 GB |

## Why v2 looks the way it does

The eval is scored under a **fixed 5-minute budget**, and the A10G is **compute-bound**. Under that
constraint the winning move is to make each step cheaper so more steps fit — the model got *smaller*,
not bigger. The full reasoning, the wins, and the confirmed dead ends are in the **Learning Report**
(`README.A10G.md`) and `program.md` carries them forward as guidance for the next agent.

`program.md` now opens with a **"Prior learnings"** section so a fresh research loop starts informed:
which knobs are tuned, which directions are dead, and where the remaining upside likely is
(architectural changes that lower loss without adding per-step cost).

## Run it

```bash
uv sync
uv run prepare.py
uv run train.py                     # starts from the v2 best config
SAVE_CHECKPOINT=1 uv run train.py   # also writes checkpoints/model.pt
```

Best-config weights: see GitHub Releases (`a10g-best-v2`).
