# 007: Retrieval-augmented generation — student guide

**Goal:** Inspect the actual evidence and exact prompt before asking a model to answer.

## Try it

```bash
python llm_workshop/007_retrieval_augmented_generation/demo.py --step
python llm_workshop/007_retrieval_augmented_generation/ui.py
```

1. Ask where the fictional Neral instrument is stored; inspect its source passage and score.
2. Change K142 to M219 in that passage. The selected evidence and prompt update.
3. Ask a question with no matching terms. No evidence is fabricated.

## Explain what changed

Is a passage the same as a model token? No; a passage contains many tokens. Did replacing the room number update model weights? No. Can the right document still lead to a wrong answer? Yes; retrieval and generation can fail separately.

## Remember

TF-IDF is a lexical baseline, not a neural retrieval embedding model. The prompt preview alone proves only which evidence would be supplied. Actual generated answers require answer_with_model.py and a staged model.

[Visual topic page](index.html)
