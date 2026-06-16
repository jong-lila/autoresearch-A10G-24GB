"""
Rejection-sampling SFT trainer — fine-tune the model on its own best samples.

Loads a checkpoint, then plain cross-entropy SFT on the (prompt + best-completion)
sequences from gen_rsft_data.py. The prompt tokens are masked out (ignore_index=-1)
so the loss only trains the completion — we teach the model to produce its own
high-diversity continuations, not to re-predict the prompt.

Reuses the repo's GPT.forward(idx, targets, reduction) loss path directly.

Usage: SAVE_CHECKPOINT=1 uv run rsft.py --ckpt checkpoints/model.pt --data rsft_data.pt
"""

import argparse
import os

import torch

from train import GPT, GPTConfig

LR = 3e-4
EPOCHS = 2
BATCH = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/model.pt")
    ap.add_argument("--data", type=str, default="rsft_data.pt")
    ap.add_argument("--out", type=str, default="checkpoints/rsft_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**blob["config"])
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.load_state_dict(blob["model"])
    model = model.to(torch.bfloat16)

    data = torch.load(args.data, weights_only=False)
    samples = data["samples"]
    plen = data["prompt_len"]

    # build (input, target) tensors; mask prompt positions in the target with -1
    seqs, tgts = [], []
    for s in samples:
        ids = s["prompt"] + s["completion"]
        ids = ids[: cfg.sequence_len + 1]
        x = ids[:-1]
        y = ids[1:]
        y = [-1] * (plen - 1) + y[plen - 1:]   # only score completion tokens
        seqs.append(torch.tensor(x, dtype=torch.long))
        tgts.append(torch.tensor(y, dtype=torch.long))
    X = torch.stack(seqs)
    Y = torch.stack(tgts)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    n = X.size(0)
    step = 0
    losses = []
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            with autocast:
                loss = model(xb, targets=yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(loss.item())
            if step % 10 == 0:
                print(f"epoch {epoch} step {step:04d} | sft_loss {loss.item():.4f}")

    print(f"---\nfinal_sft_loss: {sum(losses[-10:])/max(1,len(losses[-10:])):.4f}\nsteps: {step}")
    if os.environ.get("SAVE_CHECKPOINT") == "1":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"model": model.state_dict(), "config": blob["config"]}, args.out)
        print(f"saved RSFT model to {args.out}")


if __name__ == "__main__":
    main()
