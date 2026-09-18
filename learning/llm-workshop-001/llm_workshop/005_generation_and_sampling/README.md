# 005: Generation and sampling

Append one selected token, recompute the next distribution, and repeat.

[Topic page](index.html) · [Student guide](STUDENT_GUIDE.md)

```bash
python llm_workshop/005_generation_and_sampling/demo.py --step
python llm_workshop/005_generation_and_sampling/ui.py
```

The tiny tokenizer has no natural end-of-sequence token, so the terminal uses an explicit token budget. The model recomputes its context rather than using a KV cache. It is a transparent teaching implementation.
