# Fine-tuning — synthetic DPO

This branch adds a **DPO (Direct Preference Optimization)** stage on top of the A10G-tuned base model
(branch `autoresearch-v2`). It is self-contained: two scripts, no new dependencies, no reward model,
no external API. It runs on the same single A10G (24 GB).

## Why DPO (not PPO/GRPO)

The preference pairs here are **offline and pre-labeled**, so we need neither rollouts nor a reward
model nor a value network. DPO directly optimizes the policy against a frozen reference using the pairs,
which makes it the cheapest and most stable RLHF-family method on a small single GPU. (PPO would add a
value net + reward model + online generation — far heavier and unstable at 50M params. GRPO would still
need online sampling + a programmatic reward.) DPO holds just **2 models** (policy + frozen reference),
both of which fit comfortably in 24 GB.

## How the synthetic preferences are made (`gen_dpo_data.py`)

No human labels and no judge model. Each `(prompt, chosen, rejected)` triple is built from the real
pretraining corpus, giving a **verifiable** preference (the corruption is objectively worse text):

- **prompt** — a 128-token prefix of a real document
- **chosen** — the document's true continuation (coherent, in-distribution)
- **rejected** — a degraded continuation, one of three corruption modes (rotated evenly):
  - `repeat` — loops a 3-gram (the classic degenerate-repetition failure mode)
  - `shuffle` — randomly permutes the continuation tokens (incoherent)
  - `truncate` — keeps a quarter, pads the rest with a filler token

```bash
uv run gen_dpo_data.py --n 2000 --out dpo_data.pt
```

## The DPO trainer (`dpo.py`)

Standard DPO loss, reusing the repo's `GPT` model and tokenizer:

```
L = -log σ( β · [ (logπ(chosen)   − logπ_ref(chosen))
                − (logπ(rejected) − logπ_ref(rejected)) ] )
```

It loads one checkpoint as **both** the trainable policy and the frozen reference, then trains.
Metrics reported: `loss`, preference `margin`, and `pref_acc` (fraction where the policy already
prefers chosen over rejected).

```bash
# 1. produce a base checkpoint with THIS branch's train.py (shapes must match)
SAVE_CHECKPOINT=1 uv run train.py
# 2. generate preferences and run DPO
uv run gen_dpo_data.py --n 2000
SAVE_CHECKPOINT=1 uv run dpo.py --ckpt checkpoints/model.pt --data dpo_data.pt
```

## Results (smoke run)

A first end-to-end run on the A10G: **600 synthetic preference triples** (200 each of
repeat/shuffle/truncate), base = the v2 checkpoint (val_bpb 1.1748), `BETA=0.1`, `LR=1e-5`, 1 epoch,
75 steps. Per-step `pref_acc` = fraction of the batch where the policy already prefers *chosen*.

| step | loss | margin | pref_acc |
|-----:|-----:|-------:|---------:|
| 10 | 1.80 | +4.41 | 0.75 |
| 20 | 5.62 | +3.00 | 0.75 |
| 30 | 2.48 | +6.06 | 0.62 |
| 40 | 7.92 | −1.93 | 0.50 |
| 50 | 8.77 | +1.33 | 0.62 |
| 60 | 2.87 | +7.64 | 0.75 |
| 70 | 3.16 | +3.13 | 0.50 |

**Final pref_acc: 0.619** over the last batches. Checkpoint saved to `checkpoints/dpo_model.pt`.

**Interpretation:** the pipeline is correct — preferences generate, the DPO loss computes, the policy
trains and saves. But the run is **not converged**: loss swings wildly (1.8 → 8.8 → 2.9), the margin
even flips negative at step 40, and pref_acc just oscillates around 0.6 rather than climbing. That's
the signature of an **LR that's too high for the batch/`β`** on only 600 pairs — the optimizer is
overshooting each preference rather than accumulating signal. Treat this as a *plumbing-verified
baseline*, not a result to draw conclusions from.

## Status & caveats

- **Verified end-to-end**: data generation, DPO loss, training, and checkpoint save all run on the A10G.
- **Hyperparameters are a starting point, not tuned.** `BETA=0.1`, `LR=1e-5`, 1 epoch. As the table above
  shows, the loss is noisy and `pref_acc` hovers around 0.6 — the optimization needs an LR/β sweep and
  more data (≥2k pairs) to converge cleanly. Lowering LR (e.g. 2e-6) and raising the pair count is the
  natural next step.
- **Checkpoint must come from this branch's `train.py`** — the MLP ratio and model shape are baked into
  the model code, so a checkpoint trained with a different shape will fail to load (the script raises a
  clear error pointing you to regenerate it).
- One supporting change in `train.py`: its executable body is now under `if __name__ == "__main__":`
  so `dpo.py` can `import GPT` without triggering a training run. No behavior change when run directly.
