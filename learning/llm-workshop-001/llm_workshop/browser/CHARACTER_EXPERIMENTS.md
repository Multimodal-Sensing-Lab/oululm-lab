# Character generation, training and domain fine-tuning

## What were the original 200 updates?

The original checkpoint is unchanged. It was initialized with seed 1337, then trained on the text in `examples/data/tiny_train.jsonl` using CPU AdamW, learning rate 0.003 and dropout 0.1.

One update samples **four windows of 32 characters**. Each input character has a next-character target: 128 target predictions in the batch. The model computes loss, backpropagates and changes its 108,320 parameters once. Two hundred updates therefore sample **25,600 target positions**, with repeats. This is not 200 sentences, 200 epochs, or 25,600 unique examples.

| Corpus / split | Documents | Sentences | Whitespace words | Characters |
|---|---:|---:|---:|---:|
| Original stories · training | 3 | 12 | 109 | 521 |
| Original stories · validation | 2 | 8 | 75 | 367 |
| Controlled sentences · training | 351 | 351 | 2,815 | 14,848 |
| Controlled sentences · validation | 55 | 55 | 445 | 2,347 |

Characters include the two-newline separators between documents. Sentence counts count punctuation endings; word counts split on whitespace. All runs here tokenize **characters**, including those trained on the controlled sentences.

The story corpus concerns a robot, a ball, a boat, friends and a garden. The controlled corpus contains deliberately simple animal, classroom and garden sentences drawn from the existing word exercise. To preserve the original 32-entry character vocabulary, 81 of its 432 training documents and 17 of its 72 validation documents containing unsupported `q` or `j` were excluded. Nothing was silently replaced with UNK. The retained files, source paths, counts and hashes are recorded with the experiment.

Validation documents/sentences differ from training, but the controlled corpus shares templates and vocabulary across splits. Loss is measured on **four fixed 32-character windows per split**, not on every target in the corpus. The character experiment does not report a separate test set or a broad language benchmark.

## Recorded comparisons now available

All new full-training runs start from the exact same bundled random parameters. Snapshots at 200 and 1,000 updates are available for both corpora on both CPU and CUDA. The new CPU story run reproduces the original 200-step weights exactly. CPU/GPU differences are possible because of dropout and numerical kernels.

| Run | Updates | Final train loss | Final validation loss |
|---|---:|---:|---:|
| Original stories · CPU | 200 | 1.4313 | 2.2045 |
| Original stories · CPU | 1,000 | 0.2809 | 3.0507 |
| Original stories · GPU | 1,000 | 0.3652 | 3.0756 |
| Controlled sentences · CPU | 1,000 | 0.2551 | 0.3038 |
| Controlled sentences · GPU | 1,000 | 0.2798 | 0.2412 |

These are next-character cross-entropies in nats, not percentages. The longer story run illustrates overfitting: the training score improves while the held-out score deteriorates. The controlled task is easier and larger, but its lower loss is not evidence of general language mastery. Different datasets have different difficulty; do not treat this table as a general model ranking.

The GPU is the **NVIDIA GeForce RTX 2080 SUPER**; CUDA labels identify the NVIDIA GPU backend. The 1,000-update loops took approximately five seconds each in this recording, including periodic evaluation, excluding startup and checkpoint exports. Tiny batches can be dominated by overhead, so this does not establish a general GPU speed advantage. Browser inference always runs locally in JavaScript; it does not train on either device.

## Where to click

### 05 · One token at a time

Choose weights in the top **Model weights** menu. Start with a partial prompt such as `the small cat`. A generation click chooses one character, appends it and recomputes the next distribution. It does not train the model. Checkpoint changes reset the generated continuation to the typed prompt so comparisons start from the same text.

The notes distinguish attention shares over positions from next-character probabilities over vocabulary entries, explain greedy versus sampling, and show the selected checkpoint's training data/settings.

### 06 · Training & fine-tuning

Choose a **Recorded experiment**. CPU, GPU and LoRA runs are grouped. The shortcuts expose the original run, more steps, different data and the three domain adapters.

The **Inspect before / after** buttons change the live **CURRENTLY INSPECTING** preview: next-character probabilities, greedy continuation and a stored output weight. The historical loss chart and paired fixed samples deliberately stay fixed because they describe the complete run. The chosen weights also apply in generation, embeddings and attention.

For teaching, use a short sequence:

1. Original CPU 200: show the three-story dataset and explain one update.
2. CPU 1,000 on the same stories: show worsening validation loss.
3. Controlled GPU 1,000: show the different training data and a partial-prompt continuation.
4. One domain adapter: compare unchanged base versus adapter, then briefly mention the other two domains.

The full CPU/GPU matrix is optional depth, not twelve required demonstrations.

## Biology, email and finance: actual LoRA training

Each domain starts independently from **Controlled sentences · CUDA · 1,000 updates**. Only rank-4 A/B matrices on the vocabulary output layer are trained: `A: 4 × 64`, `B: 32 × 4`, **384 parameters**. The effective update is `W_base + 2 × B @ A`. The base parameters are checked unchanged; the merged output is checked against the unmerged adapter computation.

The training passages are the existing small fictional prose snippets in `006_domain_fine_tuning/demo.py`. Each has a separate short validation paragraph saved alongside it. Training uses 200 AdamW updates, learning rate 0.02, four 32-character windows per batch. The frozen model stays in evaluation mode, matching the mechanics lab; only A/B receive gradients. These few overlapping validation windows are a narrow diagnostic, not a domain benchmark.

The browser exposes the real low-rank matrix values and an actual update to one output weight. Since only `lm_head` changes, the adapter's embedding and attention views match the base exactly. The differences appear in vocabulary scores and generation. This is prose adaptation, not useful biology/financial advice, factual acquisition, instruction tuning or QLoRA. Biology validation does not improve in this recording; the UI shows that result rather than implying all adapters help.

## Reproduce or extend

The delivered recordings are in `runs/browser_character_20260917_01/`. They include inference checkpoints, exact filtered data, adapter A/B tensors, per-run reports, `browser_experiments.json` and the exact recording script matched to its source hash. These checkpoints do not contain optimizer-resume state.

From learning/llm-workshop-001, in the configured workshop environment:

```bash
python llm_workshop/browser/record_experiments.py \
  --output-dir runs/my_character_experiments \
  --steps 1000 --adapter-steps 200 --devices cpu cuda
```

Use a fresh output directory. This explicitly rebuilds `experiment-data.js`; the original `model-data.js` remains intact. Use `--devices cpu` on a CPU-only system, or `--steps 2000` for a longer recording (up to 10,000). Only recorded options appear in the browser. Data must use the original alphabet to keep the checkpoints compatible with the workshop; a new tokenizer/vocabulary would require a separate model route.

The browser stores the new runs in `experiment-data.js` (about 25 MB of readable numeric JSON). Keep it with the page. For a long resumable job, use the existing `llm_workshop.tiny` trainer rather than these inference-comparison snapshots.

## Checks

```bash
node llm_workshop/browser/test_experiments.cjs
node llm_workshop/browser/test_browser.cjs
node llm_workshop/browser/test_training_browser.cjs
node llm_workshop/browser/test_word_browser.cjs
```

The numerical check compares exported PyTorch logits, source hashes/counts, exact CPU reproduction and all frozen-base/merged LoRA weights. UI checks exercise run switching, visible checkpoint changes, generation reset, CPU/GPU labeling and mobile layouts.
