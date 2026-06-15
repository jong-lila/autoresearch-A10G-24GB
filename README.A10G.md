# autoresearch on A10G (24 GB)

This fork adapts [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — which is
**tested on H100 (80 GB)** — to run on a single **NVIDIA A10G (24 GB)**. Upstream runs unmodified
right up to the training step, then dies with `CUDA OutOfMemoryError`: its `DEVICE_BATCH_SIZE` is
sized for an 80 GB card. The changes here make it fit in 24 GB **without altering the experiment's
semantics** (the `val_bpb` metric stays comparable).

## Machine specs (verified)

| | |
|---|---|
| Instance | AWS `g5.xlarge` (Lila `ML-Ops-dev`, `us-east-1`) |
| GPU | NVIDIA A10G, 23 GB (22.06 GiB usable), compute capability **8.6** (Ampere) |
| NVIDIA driver | 595.71.05 (CUDA 13.x runtime) |
| vCPU / RAM | 4 vCPU / 15 GiB |
| OS | Amazon Linux 2023 |
| Python / PyTorch | 3.10.20 / 2.9.1+cu128 |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (AL2023) |

## What changed vs. upstream

All changes are in `train.py`. Nothing in `prepare.py` was touched.

| # | Change | Why |
|---|--------|-----|
| 1 | `DEVICE_BATCH_SIZE` **128 → 16** | The OOM fix. 128 needs ~22 GB+; 16 peaks at **~6.0 GB**. `TOTAL_BATCH_SIZE` (2**19 tokens/step) is unchanged — `grad_accum_steps` rises from 2 → 16 to compensate, so tokens-per-optimizer-step and therefore `val_bpb` stay directly comparable to an H100 run. |
| 2 | Added `GPU_BF16_PEAK_FLOPS`, device-auto-selected (~70 TFLOPS for A10G, else H100's 989.5) | MFU was hardcoded to the H100's peak FLOPS and would report nonsense on any other card. Now MFU reflects the actual GPU (~66% on the A10G). |
| 3 | Opt-in checkpoint save at end of run, gated by `SAVE_CHECKPOINT=1` | Upstream is ephemeral by design (the artifact is the *metric*, not weights). This adds a `checkpoints/model.pt` (state_dict + config + val_bpb) only when explicitly requested, leaving the default research loop unchanged. |

### FlashAttention-3 note
`train.py` imports FlashAttention-3, which is Hopper-only. On the A10G it transparently falls back
to `kernels-community/flash-attn3`, which **works on Ampere** — no code change was needed here.

## Result (baseline run on this hardware)

```
val_bpb:          1.323710
peak_vram_mb:     6150   (of 24 GB — lots of headroom; batch size could go higher)
mfu_percent:      ~66
tok/sec:          ~193,000
num_steps:        122 (full 300s budget)
```

## Run it

```bash
uv sync
uv run prepare.py            # downloads data + trains tokenizer (~2 min)
uv run train.py              # one 5-min experiment
SAVE_CHECKPOINT=1 uv run train.py   # same, but also writes checkpoints/model.pt
```

See `logs/` for raw training stdout and `results.tsv` for the run in the format
`analysis.ipynb` reads. (`results.tsv` is gitignored upstream; it is force-committed here so the
notebook works out of the box.)
