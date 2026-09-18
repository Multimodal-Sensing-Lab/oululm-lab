# Text to predictions

Read the [student guide](STUDENT_GUIDE.md) and [detailed walkthrough](walkthrough.html).

Run from the workshop directory:

```bash
python llm_workshop/001_text_to_predictions/tokenizer_demo.py --step
python llm_workshop/001_text_to_predictions/bpe_demo.py --word lower --step
python llm_workshop/001_text_to_predictions/lesson.py --step
python llm_workshop/001_text_to_predictions/ui.py
```

The character tokenizer builds IDs from the fixed story corpus. The BPE example separately learns merge rules by counting pairs. The neural-network lesson then performs an untrained forward pass; tokenizer preparation is separate from model-weight training.
