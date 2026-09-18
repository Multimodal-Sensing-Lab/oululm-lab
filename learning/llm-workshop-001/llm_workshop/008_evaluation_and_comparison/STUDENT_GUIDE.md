# 008: Evaluation and comparison — student guide

**Goal:** Separate measured model outputs from metric examples and control comparisons.

## Try it

```bash
python llm_workshop/008_evaluation_and_comparison/demo.py --step
python llm_workshop/008_evaluation_and_comparison/ui.py
```

1. Inspect the six fixed reference labels. Change one predicted label and observe the metric changes.
2. Enter an unsupported response such as a sentence; it appears in the INVALID column.
3. Load before.pt and after.pt from the same training run and compare actual text.

## Explain what changed

Why can a cherry-picked prompt be misleading? It does not measure general behavior. Why use macro F1 with uneven classes? It weights class-level F1 equally. Does fluent prose imply accurate domain knowledge? No.

## Remember

The hand-written label fixture is never reported as measured model performance. Actual checkpoint generations are qualitative until scored against suitable references. Keep source, split, tokenizer and decoding settings with experiment results.

[Visual topic page](index.html)
