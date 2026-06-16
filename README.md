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

## Status & caveats

- **Verified end-to-end**: data generation, DPO loss, training, and checkpoint save all run on the A10G.
- **Hyperparameters are a starting point, not tuned.** `BETA=0.1`, `LR=1e-5`, 1 epoch. In smoke tests the
  loss is noisy and `pref_acc` hovers around 0.6 — the plumbing is correct but the optimization needs an
  LR/β sweep (and more data) to converge cleanly. That sweep is the natural next step.
- **Checkpoint must come from this branch's `train.py`** — the MLP ratio and model shape are baked into
  the model code, so a checkpoint trained with a different shape will fail to load (the script raises a
  clear error pointing you to regenerate it).
- One supporting change in `train.py`: its executable body is now under `if __name__ == "__main__":`
  so `dpo.py` can `import GPT` without triggering a training run. No behavior change when run directly.
