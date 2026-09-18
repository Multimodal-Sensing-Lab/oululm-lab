# 002: Embeddings and positions — student guide

**Goal:** Token IDs select vectors; position vectors distinguish locations.

## Try it

```bash
python llm_workshop/002_embeddings_and_positions/demo.py --step
python llm_workshop/002_embeddings_and_positions/ui.py
```

1. Type `aaa` and identify identical rows in the token panel.
2. Compare those rows in the position panel; each location has a different vector.
3. Inspect the combined panel and explain elementwise addition.

## Explain what changed

Why is an integer ID not a meaning vector? It only selects a row. Why do repeated tokens have different inputs to attention? Position information is added. Does a colorful random matrix demonstrate semantics? No.

## Remember

The implementation uses learned absolute positions. Other architectures may use sinusoidal or rotary schemes. Do not describe our table as universal.

[Visual topic page](index.html)
