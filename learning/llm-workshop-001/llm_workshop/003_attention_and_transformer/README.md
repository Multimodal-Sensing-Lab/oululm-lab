# 003: Attention and transformer blocks

Inspect how a position reads information, then transforms its features.

[Topic page](index.html) · [Student guide](STUDENT_GUIDE.md)

```bash
python llm_workshop/003_attention_and_transformer/demo.py --step
python llm_workshop/003_attention_and_transformer/ui.py
```

Attention coefficients change with the input; projection matrices are persistent parameters. Our blocks use normalization before each sublayer. Heatmaps of attention alone are not full explanations of model decisions.
