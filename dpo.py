"""
DPO fine-tuning on synthetic preference data — single GPU, single file.

Reuses the GPT model + tokenizer from this repo. Loads a pretrained checkpoint as
BOTH the policy (trainable) and a frozen reference, then optimizes the standard DPO
loss over (prompt, chosen, rejected) triples produced by gen_dpo_data.py.

DPO (no reward model, no value network — fits 24 GB easily):
    L = -log sigmoid( beta * [ (logp_pi(chosen)   - logp_ref(chosen))
                             - (logp_pi(rejected) - logp_ref(rejected)) ] )

Why DPO here (vs PPO/GRPO): the preference pairs are offline and pre-labeled, so we
need no rollouts and no reward model. That's the cheapest, most stable RLHF-family
method on a single small GPU. See README.finetuning.md for the full rationale.

Usage: SAVE_CHECKPOINT=1 uv run dpo.py --ckpt checkpoints/model.pt --data dpo_data.pt
"""

import argparse
import os

import torch
import torch.nn.functional as F

from prepare import Tokenizer
from train import GPT, GPTConfig

BETA = 0.1               # DPO temperature (KL strength vs the reference)
LR = 1e-5                # conservative; DPO is sensitive to LR
EPOCHS = 1
BATCH = 8


def seq_logprob(model, prompt, resp, device):
    """Sum of log p(resp | prompt) under `model`. prompt/resp are token id lists."""
    ids = torch.tensor(prompt + resp, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(ids[:, :-1])                       # forward returns logits when targets=None
    logprobs = F.log_softmax(logits.float(), dim=-1)
    targets = ids[:, 1:]
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    # only score the response tokens, not the prompt
    start = len(prompt) - 1
    return tok_lp[0, start:].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/model.pt")
    ap.add_argument("--data", type=str, default="dpo_data.pt")
    ap.add_argument("--out", type=str, default="checkpoints/dpo_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**blob["config"])

    def load_model():
        with torch.device("meta"):
            m = GPT(cfg)
        m.to_empty(device=device)
        try:
            m.load_state_dict(blob["model"])
        except RuntimeError as e:
            raise RuntimeError(
                "Checkpoint shapes don't match this branch's GPT (e.g. MLP ratio / "
                "model shape differ). DPO must run on a checkpoint produced by THIS "
                "branch's train.py: `SAVE_CHECKPOINT=1 uv run train.py` first.\n" + str(e)
            )
        return m

    # cast both models to bf16 so the FlashAttention-3 kernel sees matching q/k/v dtypes
    # (the base train.py achieves this implicitly by running everything under one autocast)
    policy = load_model().to(torch.bfloat16)
    ref = load_model().to(torch.bfloat16)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    data = torch.load(args.data, weights_only=False)
    triples = data["triples"]
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    step = 0
    acc_hist = []
    for epoch in range(EPOCHS):
        for i in range(0, len(triples), BATCH):
            batch = triples[i:i + BATCH]
            losses, margins, correct = [], 0.0, 0
            for t in batch:
                with autocast:
                    pi_c = seq_logprob(policy, t["prompt"], t["chosen"], device)
                    pi_r = seq_logprob(policy, t["prompt"], t["rejected"], device)
                with torch.no_grad(), autocast:
                    ref_c = seq_logprob(ref, t["prompt"], t["chosen"], device)
                    ref_r = seq_logprob(ref, t["prompt"], t["rejected"], device)
                logits = BETA * ((pi_c - ref_c) - (pi_r - ref_r))
                losses.append(-F.logsigmoid(logits))
                margins += logits.item()
                correct += int(logits.item() > 0)        # policy prefers chosen
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            acc = correct / len(batch)
            acc_hist.append(acc)
            step += 1
            if step % 10 == 0:
                print(f"step {step:04d} | loss {loss.item():.4f} | "
                      f"margin {margins/len(batch):+.3f} | pref_acc {acc:.2f}")

    final_acc = sum(acc_hist[-20:]) / max(1, len(acc_hist[-20:]))
    print(f"---\nfinal_pref_acc: {final_acc:.3f}\nsteps: {step}")
    if os.environ.get("SAVE_CHECKPOINT") == "1":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"model": policy.state_dict(), "config": blob["config"]}, args.out)
        print(f"saved DPO policy to {args.out}")


if __name__ == "__main__":
    main()
