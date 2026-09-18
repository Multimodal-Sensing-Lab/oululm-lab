# Participant guide · Open the model

## Your route

1. **Global picture:** separate the tokenizer, model weights and application.
2. **Text → tokens:** enter `the cat sat.`; inspect spaces, repeated letters and IDs.
3. **Embeddings:** try `aaa`; compare token lookup rows with position lookup rows.
4. **Attention:** follow one reader through normalization, Q/K/V, scores, masking, softmax and value mixing. Attention percentages refer to input positions, not vocabulary entries.
5. **Generation:** compare untrained and trained weights. Append one token, then predict again.
6. **Training:** identify the data and update count before comparing checkpoints. Recorded CPU/GPU curves are not live browser training.
7. **Evidence and teaching:** inspect how supplied text changes a prompt. This activity assembles evidence; it does not fabricate an answer.
8. **Check understanding:** explain why an output changed and what remains uncertain.

## The word companion

Use the link in the main browser. Try `the small cat`, then `the cat cat`. The original word companion has 52 vocabulary entries and a 16-position context. Its trained checkpoint used 600 updates on 432 deliberately structured sentences. It is a different model from the 32-entry character demo and from the larger domain lab.

## Inspect the Python

Run commands from `learning/llm-workshop-001` after the [setup](../README.md#python-setup):

```bash
python llm_workshop/001_text_to_predictions/tokenizer_demo.py --step
python llm_workshop/001_text_to_predictions/bpe_demo.py --word lower --step
python llm_workshop/002_embeddings_and_positions/demo.py --text 'aaa' --step
python llm_workshop/003_attention_and_transformer/demo.py --step
python -m llm_workshop.words.demo --checkpoint runs/words_classroom_01/after.pt --text 'the small cat' --inspect --step
```

Tokenization makes IDs. Embeddings look up learned vectors. Q/K/V are projections of the normalized vectors. The mask hides future positions. Softmax supplies nonnegative mixing coefficients; their sum is one.

## Try domain adaptation

Run `python -m llm_workshop.domains.serve`, then open the displayed local URL. Begin with a branch's example. Read its starting checkpoint, source dataset, validation split and actual output. Keep the input unchanged when comparing base and fine-tuned weights.

The common base used 1,000 updates; each biology/email/finance/Q&A branch used 500 additional updates independently. Selecting an input format does not select a different model. The word vocabulary is fixed. Character mode can accept novel words made of known characters, but may produce incoherent text.

For the Q&A exercise compare `what is a cat?` with `where is the cat?`. One was practised as a definition; the other asks for information the exercise did not provide. A memorized answer is not evidence of reliable question answering.

## Make a controlled comparison

Change one thing at a time. Name the checkpoint and tokenization, keep the prompt and decoding fixed, record the raw output and inspect held-out loss. Compare loss only within the same tokenization/task. Include failures as well as successes.

## Take away

Write a prompt with a clear task, supplied evidence, constraints and an output format. For a bounded workflow: select permitted evidence → draft → check against evidence → human review. Use fictional or appropriate public data; state how AI was used and keep a person responsible for accepting the result.
