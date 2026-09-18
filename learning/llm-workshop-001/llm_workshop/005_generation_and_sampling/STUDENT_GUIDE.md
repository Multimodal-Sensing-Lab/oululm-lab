# 005: Generation and sampling — student guide

**Goal:** Append one selected token, recompute the next distribution, and repeat.

## Try it

```bash
python llm_workshop/005_generation_and_sampling/demo.py --step
python llm_workshop/005_generation_and_sampling/ui.py
```

1. Use a short prompt such as `the `. Inspect the vocabulary probabilities.
2. Change temperature and compare the concentration; no weights changed.
3. Click Next token. Identify its ID and the extended text. The chart now predicts the following token.

## Explain what changed

Why can two sampled continuations differ with identical weights? Random selection. Does temperature teach vocabulary or facts? No. Why keep prompt and decoding fixed when comparing checkpoints? Otherwise output changes have multiple causes.

## Remember

The tiny tokenizer has no natural end-of-sequence token, so the terminal uses an explicit token budget. The model recomputes its context rather than using a KV cache. It is a transparent teaching implementation.

[Visual topic page](index.html)
