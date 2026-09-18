# Text to predictions — student guide

Run `tokenizer_demo.py --step`: identify the character split, vocabulary construction, ID lookup and exact decoding. Spaces and punctuation are tokens too.

Run `bpe_demo.py --word lower --step`: count adjacent pairs, watch the four merges, then freeze the rules. Explain why `lowest` contains unsupported letters in this toy alphabet.

Run `lesson.py --step`: follow IDs through embeddings, attention and vocabulary probabilities. This model starts untrained; changing input does not train it.

[Detailed reading](walkthrough.html) · [Browser](../../index.html) · [Python setup](../../README.md#python-setup)
