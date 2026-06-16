# Changelog — autoresearch on A10G (24 GB)

A running changelog of everything this fork changes relative to
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) (which is tested on H100 80 GB).
All changes are in `train.py`; `prepare.py` is never touched (it holds the fixed eval/metric).

Format: newest first. `val_bpb` is the metric — **lower is better**. Baseline = upstream defaults
running on this A10G.

---

## [Unreleased] — agent research loop (branch `autoresearch/jun15`)

Autonomous experiment loop on top of the A10G port. Each entry is one experiment; KEEP = advanced
the branch, DISCARD = reverted (`git reset --hard`). Best result tracked as we go.

### Kept (cumulative improvements over baseline)
- **MATRIX_LR (Muon) 0.04 → 0.10** — the big win. `1.3230 → 1.2646`. Swept 0.05/0.07/0.10 (all kept);
  0.13 overshot and was reverted.
- **EMBEDDING_LR 0.6 → 0.8** — marginal, `1.2646 → 1.2639`. Kept (lower with no added complexity).

**Best so far: `val_bpb = 1.2639` (−4.5% vs baseline 1.3230).**

### Tried and reverted (dead ends, documented so they aren't re-tried)
- `MATRIX_LR 0.13` — overshoot (1.2792).
- `WARMUP_RATIO 0.0 → 0.1` — wastes the fixed 5-min budget at low LR (1.3569, worst).
- `WINDOW_PATTERN SSSL → SSLL` — more global attention, slower steps, no gain (1.2687).
- `ADAM_BETAS β1 0.8 → 0.9` — slightly worse (1.2791).
- `WEIGHT_DECAY 0.2 → 0.05` — trains faster early, overfits the short run (1.2724).

> Live progress table: see [`PROGRESS.md`](./PROGRESS.md). Full per-experiment log feeding
> `analysis.ipynb`: see `results.tsv` / `results.loop.tsv`.

---

## [0.1.0] — A10G port (make it run on 24 GB)

Upstream runs unmodified right up to the first training step, then dies with `CUDA OutOfMemoryError`
— `DEVICE_BATCH_SIZE` is sized for an 80 GB card. These three changes make it fit in 24 GB
**without changing the experiment's semantics** (the `val_bpb` metric stays comparable to an H100 run).

### Changed
- **`DEVICE_BATCH_SIZE` 128 → 32** *(the OOM fix)*. At 128 the model needs ~22 GB+; at 32 it peaks
  at ~11.7 GB. `TOTAL_BATCH_SIZE` (2¹⁹ tokens/step) is **unchanged** — `grad_accum_steps` rises to
  compensate, so tokens-per-optimizer-step and therefore `val_bpb` stay directly comparable.
  - Sweep that set this: bs16 (6.0 GB, 1.3243) → **bs32 (11.7 GB, 1.3230)** → bs64 **OOM**.
    The A10G is compute-bound (MFU ~68%), so larger batches barely change throughput; 32 is the pick.
- **MFU denominator made device-aware** — added `GPU_BF16_PEAK_FLOPS`, auto-selected (~70 TFLOPS for
  A10G, else the H100's 989.5). Upstream hardcoded the H100 peak, so MFU was nonsense on any other card.

### Added
- **Opt-in checkpoint save**, gated by `SAVE_CHECKPOINT=1` (off by default). Writes
  `checkpoints/model.pt` (state_dict + config + val_bpb). Upstream is ephemeral by design — the
  artifact is the *metric*, not weights — so this leaves the default research loop unchanged.
  Released weights: see the repo's GitHub Releases.
- `README.A10G.md` (this changelog), `logs/` (raw run stdout), and `results.tsv` (force-committed;
  gitignored upstream) so `analysis.ipynb` works on a fresh clone.

### Not changed (but worth noting)
- **FlashAttention-3** is imported by `train.py` and is Hopper-only. On the A10G (Ampere, cap 8.6)
  it transparently falls back to `kernels-community/flash-attn3`, which works — **no code change needed**.
- **`DEPTH`** stays at 8. Tried 12 → it also scales width (50M → 135M params), runs 2.7× slower, and
  trains far fewer tokens in the fixed budget → much worse (1.5943). Reverted.

### Rejected/explored hardware paths
- True H100 (`p5.48xlarge`) is not self-serve at Lila — needs a P-instance quota bump + capacity
  block. This fork targets the self-serve A10G (`g5.xlarge`) instead.

---

## Verified machine specs

| | |
|---|---|
| Instance | AWS `g5.xlarge` (Lila `ML-Ops-dev`, `us-east-1`) |
| GPU | NVIDIA A10G, 23 GB (22.06 GiB usable), compute capability **8.6** (Ampere) |
| NVIDIA driver | 595.71.05 (CUDA 13.x runtime) |
| vCPU / RAM | 4 vCPU / 15 GiB |
| OS | Amazon Linux 2023 |
| Python / PyTorch | 3.10.20 / 2.9.1+cu128 |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (AL2023) |

## Run it

```bash
uv sync
uv run prepare.py                   # downloads data + trains tokenizer (~2 min)
uv run train.py                     # one 5-min experiment
SAVE_CHECKPOINT=1 uv run train.py   # same, but also writes checkpoints/model.pt
```
