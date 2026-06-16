"""
Synthetic DPO preference data generator (no reward model, no external API).

Builds (prompt, chosen, rejected) triples from the real pretraining corpus:
- prompt:   a short prefix of a real document
- chosen:   the document's true continuation (coherent, in-distribution)
- rejected: a degraded continuation produced by a corruption that the base model
            should learn to disprefer — one of:
              * repetition   (loop a short n-gram, the classic degenerate failure)
              * shuffle      (continuation tokens randomly permuted -> incoherent)
              * truncate+pad (cut short then pad with a filler token)

This gives a clean, verifiable preference signal: the corruptions are objectively
worse text than the real continuation, so DPO has a well-defined target without
needing human labels or a judge model. Output is a .pt of tokenized triples.

Usage: uv run gen_dpo_data.py [--n 2000] [--out dpo_data.pt]
"""

import argparse
import pickle
import os

import torch

from prepare import Tokenizer, MAX_SEQ_LEN, _document_batches

PROMPT_LEN = 128          # tokens of real prefix used as the prompt
RESP_LEN = 128            # tokens of continuation (chosen / rejected)


def make_rejected(chosen_ids, vocab_size, mode, gen):
    """Corrupt a chosen continuation into a dispreferred one. Pure tensor ops."""
    n = len(chosen_ids)
    if mode == "repeat":
        # loop a 3-gram from the start to fill the response (degenerate repetition)
        k = 3
        seed = chosen_ids[:k] if n >= k else chosen_ids
        reps = (n + len(seed) - 1) // len(seed)
        return (seed * reps)[:n]
    if mode == "shuffle":
        perm = torch.randperm(n, generator=gen).tolist()
        return [chosen_ids[i] for i in perm]
    if mode == "truncate":
        # keep a quarter, pad the rest with token 0 (filler)
        keep = max(1, n // 4)
        return chosen_ids[:keep] + [0] * (n - keep)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of preference triples")
    ap.add_argument("--out", type=str, default="dpo_data.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gen = torch.Generator().manual_seed(args.seed)
    tok = Tokenizer.from_directory()
    vocab_size = tok.get_vocab_size()

    modes = ["repeat", "shuffle", "truncate"]
    triples = []

    # stream real documents, tokenize, and slice prompt + true continuation
    need = PROMPT_LEN + RESP_LEN
    doc_iter = _document_batches("train")
    while len(triples) < args.n:
        batch, _epoch = next(doc_iter)
        ids_batch = tok.encode(batch)
        for ids in ids_batch:
            if len(ids) < need:
                continue
            prompt = ids[:PROMPT_LEN]
            chosen = ids[PROMPT_LEN:PROMPT_LEN + RESP_LEN]
            mode = modes[len(triples) % len(modes)]  # rotate corruption types
            rejected = make_rejected(chosen, vocab_size, mode, gen)
            triples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "mode": mode,
            })
            if len(triples) >= args.n:
                break

    torch.save({"triples": triples, "prompt_len": PROMPT_LEN, "resp_len": RESP_LEN}, args.out)
    by_mode = {m: sum(1 for t in triples if t["mode"] == m) for m in modes}
    print(f"wrote {len(triples)} preference triples to {args.out}")
    print(f"by corruption mode: {by_mode}")


if __name__ == "__main__":
    main()
