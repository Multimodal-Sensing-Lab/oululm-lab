# Word companion: a readable learning experiment

Open **[Predict whole words](../browser/words.html)**. Both before/after models are bundled; no training, server, GPU or download is needed to present the browser page. The [character workshop](../../index.html) links to this companion above its prompt box. Each route has its own tokenizer, checkpoints and browser session.

## What is different?

The input `the small cat` becomes `[START, the, small, cat]` instead of separate letters and spaces. Each token gets an ID. The model still does embedding lookup, adds positions, runs the same two TinyGPT transformer blocks, and predicts the next token. Words and punctuation are tokens; sentence-boundary markers are tokens too.

The word model has **52 vocabulary entries** (48 words, one period and three special markers), **64 features**, **2 blocks**, **2 heads per block**, **16 context positions**, **107,828 parameters**, and **dropout 0**. The character teaching model uses a different vocabulary and a longer positional table. The shared neural-network implementation is [tiny.py](../tiny.py); we did not substitute a lookup-table sentence generator.

Whole-word output makes the predictions easier to discuss. It can still be nonsense before training. The trained examples are readable because the corpus has deliberately simple grammar. This is a controlled pattern-learning demonstration, not a useful instruction assistant or a biology knowledge test.

## A 15–20 minute complementary route

Explore this companion after the character example, or use it as your main learning route.

| Stage | Action | Explain / ask | Python to open |
|---|---|---|---|
| 1. Words → IDs, 3 min | Start untrained with `the small cat`; then try `the cat cat` | Same word, same ID. START is an explicit marker. Spaces separate words. | [tokenizer.py](tokenizer.py): `pieces`, `encode`, `prompt` |
| 2. Inside the model, 5–7 min | Inspect token, position and sum views; then attention | Repeated words select identical token vectors. Position vectors differ. Future positions are masked. | [model.py](model.py): `WordSession`; [tiny.py](../tiny.py): `forward`; [003 inspection](../003_attention_and_transformer/demo.py) |
| 3. Choose a word, 4 min | Generate untrained; reset and choose after-training weights | The output ID is appended. Greedy and sampling are different selection policies. END stops generation. | [model.py](model.py): `generate` |
| 4. What training changed, 4–6 min | Read curves, held-out loss and all four fixed comparisons | Learning improved this controlled task. Held-out sentences still share templates and vocabulary. | [train.py](train.py): `train`, especially `loss.backward()` and `optimizer.step()` |

Each stage has **Open the Python guts** with a short code excerpt and a source link. Open that source in your editor for class; the browser does not run a Python kernel.

Good demonstration prompts are `the small cat`, `the green plant`, `the happy teacher`, and `the quiet robot carries`. `the cat cat` intentionally repeats a word for inspection; it is not normal corpus grammar. `the dragon` demonstrates unknown-word rejection.

### Tokenizer contract

- Accept English letters, ordinary whitespace and `. ! ?`; lowercase before splitting.
- Split into whole words and punctuation. Only entries in the saved training vocabulary can be encoded. The supplied corpus contains `.`, but not `!` or `?`, so those two are rejected as unknown vocabulary entries for this checkpoint.
- Normalize whitespace on decoding; this is not an exact byte-for-byte round trip.
- Add `<BOS>` (START, ID 0) before a prompt. Training sentences additionally end with `<EOS>` (END, ID 1) after their period.
- Reserve `<UNK>` at ID 2, but reject unknown input instead of silently replacing it. The untrained model can generate a reserved token.
- Keep IDs internally throughout generation, including special IDs. Never tokenize the display spelling `<UNK>`.
- Stop on END, the requested output budget, or the 16-position context limit. START consumes one position, so the prompt can contain at most 15 word/punctuation tokens. The corpus trains on short single sentences; arbitrary longer prompts are outside that simple distribution.

The next-token chart samples the full vocabulary; its bars show only the eight highest probabilities. No grammar rule is applied to repair generated sentences or restrict generation to corpus sentences.

## The actual corpus

Data lives in [examples/data/words](../../examples/data/words/). [corpus.py](corpus.py) authors all sentences, removes the possibility of duplicate sentences by construction, and splits each family reproducibly before fitting the tokenizer on training text only. No public dataset or model download is involved.

| Split | Sentences / records | Words | Text characters | Token IDs including START, period and END | JSONL bytes |
|---|---:|---:|---:|---:|---:|
| Train | 432 | 3,456 | 17,396 | 4,752 | 41,444 |
| Validation | 72 | 576 | 2,897 | 792 | 6,905 |
| Test | 72 | 576 | 2,875 | 792 | 6,883 |
| **Total** | **576** | **4,608** | **23,168** | **6,336** | **55,232** |

Characters include spaces and punctuation within each sentence, excluding JSON syntax and inter-record separators. Words count alphabetic word pieces. JSONL bytes include fields and syntax. File hashes and counting details are in [manifest.json](../../examples/data/words/manifest.json).

Families are **animals**, **classroom**, and **garden**, with 192 sentences each (144 train / 24 validation / 24 test). Examples:

```text
the small cat sleeps in the garden.
the happy teacher reads the book in the classroom.
the green plant needs water in the yard.
```

The splits have disjoint complete sentences, but familiar templates and vocabulary appear across splits. They test recombination of familiar patterns. They do not test new grammar, unknown vocabulary or general world knowledge. Garden examples are teaching prose, not a vetted biology reference.

## What was actually trained

The bundled base run is saved locally under `runs/words_classroom_01/`. It used **600 updates**, **batch size 32**, **AdamW learning rate 0.003**, seed **1337**, and **FP32** on the detected **NVIDIA GeForce RTX 2080 SUPER**.

| Loss, averaged over all non-padding target tokens | Before | After |
|---|---:|---:|
| Train | 4.0359 | 0.6427 |
| Validation | 4.0499 | 0.6787 |
| Test | 4.0202 | 0.6714 |

Validation is measured every 50 updates. The update budget was fixed before the run; test is measured for the initial and final models and does not select a checkpoint. Each sentence is an independent example; no training window crosses a sentence/split boundary. Targets for padded positions are `-100`, so cross-entropy ignores them. Loss includes punctuation and the END target, and cannot be directly compared with character-model loss.

Recorded greedy continuations include:

```text
the small cat sleeps in the yard.
the happy teacher carries the box in the room.
the green plant needs water in the yard.
the quiet robot carries the box in the library.
```

The measured training-loop interval was **2.98 seconds**, including periodic validation and the initial checkpoint save, excluding interpreter startup, final test/sample work and browser export. PyTorch reported about **23.7 MiB peak tensor memory allocated** for the run. This excludes CUDA context/driver/display memory and is not the total shown by `nvidia-smi`. These are measurements of this small run, not runtime or memory promises for larger models.

## Python commands for your presentation

Run from learning/llm-workshop-001 in the existing `llm-workshop` environment. No dependency changes are required.

### Use the already saved model

```bash
python -m llm_workshop.words.demo \
  --checkpoint runs/words_classroom_01/after.pt \
  --text 'the small cat' --inspect --step

python -m llm_workshop.words.compare \
  --left runs/words_classroom_01/before.pt \
  --right runs/words_classroom_01/after.pt --text 'the green plant'
```

`--inspect` prints token IDs, token/position/sum vectors, block-0/head-0 attention and next-token probabilities for the initial prompt. Add `--sample --temperature 0.8` for sampling; the word demo defaults to greedy selection for comparison. Inference/inspection uses CPU and does not need GPU access.

With **`--step`**, each turn shows the current sentence, eight most likely next tokens and the model's suggestion. You choose what to append:

- **Enter** accepts the suggestion.
- **Type a word**, such as `sleeps` or `plays`, to append your own choice instead. The next prediction is recomputed from that changed sentence.
- **`/words`** lists the checkpoint's vocabulary. Unknown words are rejected without changing the sentence.
- **`/quit`**, Ctrl+C or end-of-input stops cleanly.

For example, start with `the small cat`, type `plays`, and inspect the new probabilities for `the small cat plays`. Press Enter to accept the next suggestion, or enter another word yourself. This changes the input context; it does **not** train or change any weights. No retraining is needed to use this feature.

Enter one word at a time, with punctuation such as `.` entered separately. Manual and accepted tokens both count toward `--tokens`; the context limit still applies. You can override an `<EOS>` suggestion with a word, or accept it to finish. Without `--step`, the demo generates automatically as before.

### Train another independent base

```bash
python -m llm_workshop.words.train \
  --device auto --steps 600 --output-dir runs/words_my_run
```

`auto` selects CUDA when available and otherwise CPU; the report identifies the actual device. `--device cuda` requires CUDA and fails clearly if unavailable. `--device cpu` explicitly uses CPU. The GPU cannot be selected from the existing character trainer's CLI; this is the separate word command.

The data is already bundled. To regenerate it for inspection without replacing the supplied splits:

```bash
python -m llm_workshop.words.corpus --output-dir /tmp/word_corpus_check
```

Use a new output directory for each run. Files include `before.pt`, `after.pt` and `report.json`, with tokenizer metadata, configuration, losses, fixed samples, split hashes and runtime information. These are inference/comparison checkpoints, not exact optimizer-resume snapshots. Character checkpoints and word checkpoints are intentionally incompatible; use the matching commands.

### Export your new run into the browser

```bash
python -m llm_workshop.words.export --run-dir runs/words_my_run
node llm_workshop/browser/test_words.cjs
```

Export loads the saved CPU copies of both checkpoints and writes `browser/word-model-data.js` plus `word-reference.json`. It does **not** retrain. It deliberately replaces the bundled word experiment; keep the associated run directory and inspect its report first. It does not alter `browser/model-data.js`, the character experiment. Reload the word page to use the new export.

Exported models include checkpoint hashes and the current export-source hashes. Python reference tensors validate the browser's actual computations. Python and JavaScript are checked within explicit floating-point tolerances rather than for bit-identical arithmetic.

## Optional second experiment: a word adapter

This terminal extension uses real LoRA on the output layer, starting from the trained word model. It adapts toward one **existing family** of the corpus: `garden`, `classroom` or `animals`. It does not introduce new vocabulary or claim acquisition of new facts. Validation/test use the corresponding held-out family sentences.

```bash
python -m llm_workshop.words.adapt \
  --base-checkpoint runs/words_classroom_01/after.pt \
  --family garden --rank 4 --steps 100 --device auto \
  --output-dir runs/words_garden_my_run

python -m llm_workshop.words.compare \
  --left runs/words_classroom_01/after.pt \
  --right runs/words_garden_my_run/merged.pt --text 'the small'
```

Rank 4 trains **464 numbers**: 4 × 64 in A plus 52 × 4 in B. The base parameters are frozen and checked. The command also checks that merging preserves logits. The report saves family train/validation losses, held-out family loss, fixed comparisons and the original base hash. It writes `merged.pt` and `adapter_matrices.pt`. Start each family from the **same base** and use distinct directories.

Family loss can improve because the distribution is narrower. Check other families before claiming overall improvement. The word browser currently compares the untrained and fully trained base; adapter comparisons are a clearly separate Python extension, not a third hidden browser mode.

The prepared `runs/words_garden_01/` contains an actual 100-update CUDA adapter run: garden validation loss **0.6624 → 0.5523**, garden test loss **0.6407 → 0.5449**. You can use its `merged.pt` immediately for the optional terminal comparison.

## Before class

1. Open the character workshop and word companion in separate tabs. Both are ready offline.
2. Keep this README, the [Participant guide](../STUDENT_GUIDE.md) and the [glossary](../glossary.html) nearby.
3. Open `tokenizer.py`, `train.py`, `model.py` and shared `tiny.py` in the editor.
4. Rehearse one untrained continuation, one trained continuation and the learned END marker.
5. Test an unknown word and the repeated-word example. Retain an example of a limitation.
6. If changing data or training settings, train, export and test your new experiment. The bundled experiment needs no new training.

The agent sandbox initially hid CUDA; the same configured interpreter could access the GPU outside that sandbox. From a local terminal, check your environment with `torch.cuda.is_available()` if an explicit CUDA run fails. No new CUDA toolkit or environment was installed for this experiment.

## Verification

```bash
python -m unittest discover -s tests -p test_words.py -v
node llm_workshop/browser/test_words.cjs
node llm_workshop/browser/test_word_browser.cjs
node llm_workshop/browser/test_browser.cjs
```

The numerical word test compares both checkpoints' logits, embeddings and attention/block intermediates against Python; it also checks greedy output, repeated-word lookup, causality, masking, context limits, unknown input and special IDs. The Python checks cover reproducible corpus splits, padding-independent loss, learning, reloads, END handling and frozen/merged adapters. Browser tests use temporary Chrome profiles and local files only.

## Larger biology / finance word experiment

The [data lab](../browser/data-lab.html) records an additional 2,567-entry model trained on 9,500 biology records, then adapted to 3,800 finance records, with and without biology replay. It uses words plus separate digits and punctuation. It has its own checkpoints and tokenizer; the original 52-entry classroom experiment above stays separate. See [commands, input records, plots and measured results](../domains/README.md).

`words.train` and `words.adapt` now save an updated `progress.png` during training and an SVG when finished. Add `--live-plot` to show a desktop plot window. The input dataset path and split counts are printed before training.
