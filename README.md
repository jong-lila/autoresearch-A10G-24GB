# Fine-tuning v3 — rejection-sampling SFT (self-improvement)

This branch tries a different synthetic fine-tuning approach than the DPO branch: **rejection-sampling
SFT** (a.k.a. the STaR recipe). Instead of preference pairs, the model samples several continuations of
its own, a programmatic scorer keeps the best one, and the model is then plain-SFT'd on its own best
samples. No reward model, no judge, no external API, no new dependencies — runs on the same A10G (24 GB).

## Why this, after DPO

The synthetic-DPO branch failed for a specific reason: its preference pairs (real text vs.
repeat/shuffle/truncate corruption) were **trivially separable** — the base model already preferred the
real text by a wide margin, so DPO had almost no gradient to learn from. Rejection-sampling SFT fixes
that by making the candidates **on-policy** (drawn from the current model), so they sit right at the
model's ability frontier — where the learning signal actually is.

## Pipeline

**`gen_rsft_data.py`** — for each real-document prompt (128 tokens), sample N continuations from the
model (temp 0.9, top-k 50), score each, keep the best, save `(prompt, best_completion)` as an SFT target.

Scoring is programmatic and **deliberately not perplexity** (repetition has *low* perplexity and would be
rewarded). It combines:
- distinct-bigram ratio (diversity),
- a **hard repetition penalty** when any single token exceeds 15% of the completion (catches the
  `flu flu flu …` domination that bigram diversity alone misses),
- a mild mean-logprob fluency floor.

**`rsft.py`** — plain cross-entropy SFT on the kept sequences, with the prompt tokens masked
(`ignore_index=-1`) so loss trains only the completion. Reuses the repo's `GPT.forward` loss path.

```bash
SAVE_CHECKPOINT=1 uv run train.py                 # base checkpoint (this branch's shapes)
uv run gen_rsft_data.py --prompts 200 --samples 4 # sample + filter -> rsft_data.pt
SAVE_CHECKPOINT=1 uv run rsft.py --data rsft_data.pt
```

## Results (smoke run) — and the honest finding

Verified end-to-end on the A10G:
- **Data gen:** 200 examples, mean max-unigram-frac of kept completions **0.118** (i.e. the scorer
  successfully avoids picking heavily-repetitive samples; a first version without the repetition penalty
  let `flu flu flu …` through at ~0.87 single-token domination).
- **SFT training:** loss 3.57 → 3.17 over 2 epochs (26 steps), model saved.

**But the core finding is negative, and it's about the base model, not the pipeline:** even the
best-of-4 sampled completions are weak — either creeping repetition or fluent-looking gibberish (sample
decodes are in the commit history). A 50M model trained for 5 minutes simply cannot produce coherent
128-token continuations, so there is little quality to *select for*. Rejection-sampling SFT assumes the
model can occasionally produce good output you can filter and amplify; **at this scale/budget that
assumption doesn't hold**, so SFT-ing on the filtered samples mostly reinforces mediocre text.

This mirrors the DPO lesson from the other direction: DPO had *too-easy* signal; RSFT has a *too-weak
base*. Both point to the same prerequisite.

## What would actually help (in order)

1. **A stronger base model first.** Pretrain longer (minutes→hours, or a bigger model) before any
   self-improvement step. This is the binding constraint — both fine-tuning attempts are bottlenecked
   by base capability, not by the fine-tuning method.
2. **A verifiable *task* instead of open-ended text.** Generate synthetic data for a task with a checkable
   answer (arithmetic, sorting, copying, bracket-matching). Difficulty is controllable, correctness is
   free, and "best sample" becomes objective rather than a fragile text-quality proxy.
3. **Then** layer DPO/KTO on top, using on-policy pairs (best vs. worst sample) rather than
   real-vs-corrupted, so the preference signal is non-trivial.

## Files

- `gen_rsft_data.py` — sampler + programmatic scorer → SFT dataset
- `rsft.py` — masked cross-entropy SFT on the kept samples
- `train.py` — unchanged from v2 except the `__main__` guard (so the scripts can `import GPT`)
