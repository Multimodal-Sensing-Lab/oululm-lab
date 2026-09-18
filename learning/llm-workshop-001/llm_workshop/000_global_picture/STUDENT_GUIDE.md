# Student handout: from text to a prediction

Open [the short visual page](student.html), then [the architecture and pipeline map](index.html) when you want more detail.

## Five things to remember

1. **Tokenizer:** an algorithm that splits text into pieces and maps them to integer IDs. A token can be a character, word, subword or byte-based piece. Our first tokenizer uses characters.
2. **Embedding:** a vector (list of numbers) looked up using a token ID. The vector is a model parameter; the ID is just an address.
3. **Transformer blocks:** layers that combine information from allowed positions and transform the numerical features. Our model has two blocks.
4. **Prediction:** one score per possible next token. Softmax converts those scores into probabilities.
5. **Generation:** choose a token, append it, predict again, and convert the IDs back into text.

## What are we starting with?

A complete, **untrained** tiny GPT-style network. Its weights already exist, but they have not learned from the stories. It can make predictions, but random-looking output is expected. The tokenizer works before the neural network is trained.

The vocabulary is a stable table. For example, an invented table `['<UNK>', ' ', 'a', 'c', 't']` encodes `cat` as `[3, 2, 4]`. The real demo prints its own table. IDs are not drawn randomly for every sentence and do not carry meaning by their numerical size.

## Training and inference are different activities

**Training:** use known examples → predict → calculate a penalty (loss) → compute gradients → update weights. Check separate validation text to see whether learning generalizes.

**Inference:** use the current weights → predict → select a token → repeat. The weights stay fixed. Changing the input changes the answer without being a training update.

**Loss:** a penalty based on how much probability the model gave the observed next token. More probability on that token means a smaller penalty. We average penalties over positions. It is not an accuracy percentage.

**Temperature:** controls how concentrated the sampling distribution is. It changes how we choose tokens, not what the model has learned.

## First experiment

Run from learning/llm-workshop-001 after activating the workshop environment:

```bash
python llm_workshop/001_text_to_predictions/tokenizer_demo.py --step
```

Press Enter after each stage. Look for the spaces, repeated letters and repeated IDs. Compare the decoded result with the input. Edit `TEXT` or use `--text "the dog ran."` to try another example. An unsupported character produces an explicit error because this first vocabulary is deliberately small.

Only after that, trace the neural network:

```bash
python llm_workshop/001_text_to_predictions/lesson.py --step
```

Read the architecture, then watch tensor shapes through embeddings, blocks, scores and loss. This script does not train or create web pages.

## Questions to discuss

- Which is random at the start: the vocabulary lookup or the neural-network weights?
- What is the difference between an ID and an embedding vector?
- Why can the model use the current character but not future characters to predict the next one?
- If temperature changes while weights stay fixed, has learning occurred?

Later we will compare **fine-tuning**, which updates parameters from examples, with **RAG**, which retrieves document passages and supplies them in the input. A retrieved passage can contain many tokens; a document chunk is not a token.

## Desktop demos now available

All numbered topics 000–008 have terminal and desktop entry points. Start with `python -m llm_workshop.visual --topic 1`; follow [Participant guide](STUDENT_GUIDE.md) for training, adapters, retrieval and comparisons. Existing explanations above remain the conceptual reference.
