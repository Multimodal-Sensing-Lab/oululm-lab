# 003: Attention and transformer blocks — student guide

**Goal:** Inspect how a position reads information, then transforms its features.

## Try it

```bash
python llm_workshop/003_attention_and_transformer/demo.py --step
python llm_workshop/003_attention_and_transformer/ui.py
```

1. Enter `the cat` and select the attention view. Rows are query positions; columns are key positions.
2. Switch to mask and masked_scores; future scores are negative infinity, shown as blank cells.
3. Select Q/K/V, then switch head and block; the numbers belong to different projections or layers.

## Explain what changed

Why can the diagonal be nonzero? The current token is known when predicting the following token. Why does the first attention row have weight one? Only one location is allowed. Does this prove that the model understands the sentence? No.

## Remember

Attention coefficients change with the input; projection matrices are persistent parameters. Our blocks use normalization before each sublayer. Heatmaps of attention alone are not full explanations of model decisions.

[Visual topic page](index.html)
