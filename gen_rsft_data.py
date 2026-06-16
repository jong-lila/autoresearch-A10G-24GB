"""
Rejection-sampling SFT data generator (self-improvement, no judge, no external API).

For each real-document prompt, sample N continuations FROM THE MODEL ITSELF, score each
with a programmatic, verifiable proxy for quality, and keep the best one. The kept
(prompt + best-completion) sequences become SFT targets — i.e. the model is trained on
its own best samples (the STaR / rejection-sampling-fine-tuning recipe).

Why this beats the synthetic DPO we tried first: the candidates are ON-POLICY (drawn from
the current model), so they sit right at the model's ability frontier. That gives real
learning signal, unlike real-vs-corrupted preference pairs which the base model already
separates trivially.

Scoring (no reward model):
  primary  = distinct-bigram ratio   -> directly punishes degenerate repetition
  fluency  = mean token logprob      -> floor against pure gibberish
  combined = distinct_ratio + LAMBDA * normalized_fluency
NOTE: we do NOT rank by perplexity alone — repetition has LOW perplexity and would be
rewarded. Diversity is the primary signal; fluency is only a floor.

Usage: uv run gen_rsft_data.py --prompts 400 --samples 4 --out rsft_data.pt
"""

import argparse
import os

import torch
import torch.nn.functional as F

from prepare import Tokenizer
from train import GPT, GPTConfig

PROMPT_LEN = 128
GEN_LEN = 128
TEMP = 0.9
TOP_K = 50
LAMBDA = 0.3          # weight of fluency floor relative to diversity


@torch.no_grad()
def sample_batch(model, prompt_ids, n, device, max_ctx):
    """Autoregressively sample n continuations for one prompt. Forward-only (no KV cache)."""
    ctx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(n, 1)
    gen = torch.empty((n, 0), dtype=torch.long, device=device)
    for _ in range(GEN_LEN):
        inp = torch.cat([ctx, gen], dim=1)[:, -max_ctx:]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inp)[:, -1, :].float()
        logits = logits / TEMP
        if TOP_K:
            v, _ = torch.topk(logits, TOP_K, dim=-1)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        gen = torch.cat([gen, nxt], dim=1)
    return gen  # [n, GEN_LEN]


def distinct_bigram_ratio(ids):
    if len(ids) < 2:
        return 0.0
    bigrams = list(zip(ids[:-1], ids[1:]))
    return len(set(bigrams)) / len(bigrams)


def max_unigram_frac(ids):
    """Fraction of the completion taken by its single most frequent token.
    Directly catches 'flu flu flu ...' domination that bigram diversity misses."""
    if not ids:
        return 1.0
    from collections import Counter
    return max(Counter(ids).values()) / len(ids)


def quality_score(ids, flu):
    """Diversity-first, with a hard repetition penalty and a fluency floor.
    Returns a scalar; higher is better."""
    div_bi = distinct_bigram_ratio(ids)
    dom = max_unigram_frac(ids)
    # heavy penalty once any token exceeds 15% of the completion (repetition)
    rep_penalty = max(0.0, dom - 0.15) * 4.0
    return div_bi - rep_penalty + LAMBDA * (flu / 10.0)


@torch.no_grad()
def mean_logprob(model, prompt_ids, comp_ids, device, max_ctx):
    seq = torch.tensor(prompt_ids + comp_ids, dtype=torch.long, device=device).unsqueeze(0)[:, -max_ctx:]
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(seq[:, :-1]).float()
    lp = F.log_softmax(logits, dim=-1)
    tgt = seq[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    return tok_lp[-len(comp_ids):].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/model.pt")
    ap.add_argument("--prompts", type=int, default=400)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--out", type=str, default="rsft_data.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**blob["config"])
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.load_state_dict(blob["model"])
    model = model.to(torch.bfloat16).eval()
    max_ctx = cfg.sequence_len

    tok = Tokenizer.from_directory()
    from prepare import _document_batches
    doc_iter = _document_batches("train")

    kept = []
    diversities = []
    while len(kept) < args.prompts:
        batch, _ = next(doc_iter)
        for ids in tok.encode(batch):
            if len(ids) < PROMPT_LEN + 1:
                continue
            prompt = ids[:PROMPT_LEN]
            samples = sample_batch(model, prompt, args.samples, device, max_ctx)
            best, best_score, best_div = None, -1e9, 0.0
            for r in range(samples.size(0)):
                comp = samples[r].tolist()
                flu = mean_logprob(model, prompt, comp, device, max_ctx)
                score = quality_score(comp, flu)
                if score > best_score:
                    best, best_score, best_div = comp, score, max_unigram_frac(comp)
            kept.append({"prompt": prompt, "completion": best})
            diversities.append(best_div)  # now tracks max-unigram-frac of kept (lower=better)
            if len(kept) >= args.prompts:
                break

    torch.save({"samples": kept, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN}, args.out)
    print(f"wrote {len(kept)} rejection-sampled SFT examples to {args.out}")
    print(f"mean max-unigram-frac of kept completions: {sum(diversities)/len(diversities):.3f} (lower=less repetitive)")


if __name__ == "__main__":
    main()
