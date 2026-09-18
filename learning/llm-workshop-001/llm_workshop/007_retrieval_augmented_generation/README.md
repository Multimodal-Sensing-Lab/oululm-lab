# 007: Retrieval-augmented generation

Inspect the actual evidence and exact prompt before asking a model to answer.

[Topic page](index.html) · [Student guide](STUDENT_GUIDE.md)

```bash
python llm_workshop/007_retrieval_augmented_generation/demo.py --step
python llm_workshop/007_retrieval_augmented_generation/ui.py
```

TF-IDF is a lexical baseline, not a neural retrieval embedding model. The prompt preview alone proves only which evidence would be supplied. Actual generated answers require answer_with_model.py and a staged model.
