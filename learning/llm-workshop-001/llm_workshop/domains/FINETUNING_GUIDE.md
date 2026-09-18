# Which model, which data, which task?

## Start here

From learning/llm-workshop-001, with the `llm-workshop` environment active:

```bash
python -m llm_workshop.domains.serve
```

Open **http://127.0.0.1:8765/llm_workshop/browser/domain-lab.html**. Choose Words or Characters, then a branch. Compare **Common pretrained base** with **After this run**, keeping your input and format unchanged. Click Generate after changing weights.

Saved samples, reports and the data viewer also work by opening `domain-lab.html` directly. Generating a new answer requires the local Python server. It runs the real saved weights on CPU; it sends no requests to an external model. Training is a separate Python action.

### Why an input may fail

**Input format changes the prompt layout, not the weights.** Biology with “Question → answer” still uses a biology prose model. The page now names the selected model and its trained task beside the input, warns about mismatches, and offers a button to select Email or Q&A weights while keeping your text. Ordinary branch selection loads that branch's example; changing tokenization or base/after preserves your input.

- **Unknown words:** all word branches share the same 3,348-entry vocabulary. In `Write an answer` with incoming email `Hello, I am arriving late`, the pieces `am`, `arriving` and `late` are missing. Selecting Email does not add those words. The page checks the full formatted input before sending it. “Try character weights · keep input” can represent those words using its existing letters, but selects a separately trained model; accepting the input does not imply a useful answer.
- **Wrong task:** Words → Biology → Question → answer → `where is the cat?` produces just `.` followed by EOS with these saved weights. That is a real model failure, not an API failure. The browser keeps the raw output and explains punctuation-only or empty results.
- **Limited generalization:** Words → Authored Q&A → `what is a cat?` reproduces the practiced definition. `where is the cat?` produces `an a cat is a small mammal.`: it gives a definition rather than a location. The exercise has no information about where a particular cat is.
- **Email limitations:** Characters → Email accepts the late-arrival input, but the current 500-update checkpoint generates incoherent text. It demonstrates learned patterns and failure, not a working email assistant. Use the branch example for a reproducible before/after comparison, and keep failures visible.

These observations use the checkpoints listed in `browser/domain-catalog.json`, greedy decoding and a 96-token budget. The UI changes do not retrain checkpoints, expand their vocabulary or replace generated text with prepared answers.

## What was pretrained?

For each tokenization we trained a new TinyGPT **from random initialization for 1,000 updates**, on 432 controlled classroom sentences plus three stories (435 records). It is a workshop-trained base, not a downloaded general-purpose LLM. Validation uses 72 separate controlled sentences plus two stories (74 records).

We then copied that same base independently for:

- **Biology: 500 updates** on the biology CSV.
- **Email: 500 updates** on the email CSV.
- **Finance: 500 updates** on the finance CSV.
- **Q&A exercise: 500 updates** on explicitly authored examples, separate from those CSVs.

All weights are trainable. These runs use full fine-tuning; the earlier browser character adapters remain separate **LoRA** experiments. Email does not start from the biology branch. Finance does not start from the email branch. Each saved branch records the identical starting base hash.

### Actual input files and sizes

| Source | Records | Train / validation | Whitespace words | Prepared text characters | CSV file bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `data/external/animal_biology_sentences.csv` | 10,000 | 9,500 / 500 | 132,494 | 949,989 | 1,148,649 |
| `data/external/email_finetune.csv` | 1,686 | 1,602 / 84 | 214,354 | 1,310,855 | 2,663,428 |
| `data/external/finance_sentences.csv` | 4,000 | 3,800 / 200 | 78,703 | 459,933 | 541,196 |
| General base | 509 | 435 / 74 | 4,216 | 21,175 | JSONL inputs; see manifest |
| Authored Q&A | 40 | 30 / 10 | 724 | 3,828 | `prepare.py` definitions |

Counts sum the text given to preprocessing, before lowercasing, with no separators between records. Email counts include the reconstructed instruction/incoming-email/response headers, not a second copy of the redundant `text` column. File bytes also include metadata, quoting and redundant fields. A record is not necessarily one grammatical sentence. Words counted with whitespace are not model tokens.

Biology is mostly comparative animal prose (7,416 rows). Finance contains 2,496 calculation examples, 572 comparisons, 540 concepts, 252 instruments and 140 ratios. Email has 17 email types, including replies, invitations, reports and onboarding; its domain column says computer science. The viewer shows each category and actual record.

## Raw CSV → model input → learning targets

1. **Read and validate the schema.** Require unique IDs and nonempty training fields. Keep IDs/categories for auditing; do not turn them into prose.
2. **Remove exact normalized duplicates before splitting.** Lowercase/whitespace-normalized full prepared records are the duplicate key. These three files had none removed. Similar templates can still remain.
3. **Hold out about 5% by category**, using a fixed seed. Split before vocabulary fitting or training. Store the exact train/validation JSONL files.
4. **Build one shared vocabulary from all planned TRAIN splits.** This reserves domain words and question punctuation before base training; only the general base text supplies base-training gradients. Validation never adds vocabulary entries.
5. **Serialize the chosen fields and lowercase.** Biology/finance use `sentence` only. Email uses the structure below. Its `text` column repeats the content, so we do not append it. The textual `<|endoftext|>` is replaced by our native EOS token.
6. **Add BOS/EOS and form next-token targets.** Given `BOS a cat EOS`, inputs are `BOS a cat`, targets are `a cat EOS`. Keep records separate; pad short batch members with ignored targets. Oversized records raise an error; no silent truncation.
7. **Choose which targets count toward loss.** Prose scores every next token. Email and Q&A score response tokens plus EOS only. Prompt tokens remain visible to attention, but their target labels are `-100`, which the loss ignores.
8. **Train, then evaluate without updates.** Report target-weighted mean loss over complete validation splits. Save checkpoints, hashes, data, curves and raw samples.

```text
### instruction:
Write a friendly welcome email ...

### incoming email:
[Only included when supplied]

### email:
Subject: Welcome ...
...
```

Everything through `### email:` is the prompt. The email response is what we train the model to produce. In Q&A the corresponding structure is `question: ...\nanswer: ...`.

**Vocabulary means allowed pieces plus their integer IDs.** The designer chooses the splitting rule. Here we collect distinct lowercase characters, or whole words / individual digits / punctuation, then add BOS, EOS and UNK. The resulting size comes from the chosen training text: **61 character entries** or **3,348 word-mode entries**. There is no hand-picked claim that English has 3,348 words. This is not BPE. IDs and embedding row counts then stay fixed for these checkpoints. A new vocabulary requires aligned embeddings/output weights and a new compatible checkpoint; editing the prompt does not resize the model.

External-domain validation has zero unknown pieces. The word base-validation probe contains **10 UNK input tokens**; these are mapped to UNK, not silently added to the vocabulary. The character probe has zero. Context size is a separate choice.

## Settings and recordings

| Setting | Words + digits + punctuation | Characters |
| --- | ---: | ---: |
| Vocabulary | 3,348 | 61 |
| Context positions | 256 | 1,280 |
| Parameters | 548,372 | 189,885 |
| Batch records per update | 32 | 8 |
| Blocks / heads / width | 2 / 2 / 64 | 2 / 2 / 64 |
| Base updates / learning rate | 1,000 / 0.003 | 1,000 / 0.003 |
| Each branch updates / learning rate | 500 / 0.0006 | 500 / 0.0006 |

AdamW, dropout 0, seed 1337, gradient clipping 1.0. Each update samples a batch **with replacement**; updates are not epochs or unique records. Reports also give the actual scored target exposures. The context is large enough for these complete email records; this is a separate architecture from the original 64-position character and 16-position word demos.

Final recordings:

- [Word run](../../runs/domains500_words_20260917_02/report.json): base 8.8 s; biology 4.6 s; email 7.2 s; finance 4.3 s; Q&A 4.2 s.
- [Character run](../../runs/domains500_characters_20260917_01/report.json): base 7.4 s; biology 4.3 s; email 10.6 s; finance 3.9 s; Q&A 3.4 s.

Measured on this RTX 2080 SUPER, PyTorch 2.5.1/CUDA 12.1. Timings include validation and plot writes, but exclude startup and final checkpoint saving/generation. These are tiny-model measurements, not a large-model estimate. The `_words_01` run was a preparation run; the browser catalog points to `_words_02`.

### What actually improved?

| Branch | Word validation: base → after | Character validation: base → after |
| --- | ---: | ---: |
| Biology | 12.530 → 1.785 | 7.458 → 2.104 |
| Email (response loss) | 11.776 → 1.178 | 6.942 → 2.489 |
| Finance | 12.037 → 1.626 | 7.193 → 1.780 |
| Q&A (held-out phrasing) | 10.367 → 0.116 | 6.191 → 5.802 |

Compare before/after **within one cell**, not absolute loss across tokenizations or tasks. Lower validation loss means higher average probability for the observed targets; it is not an accuracy or factuality score. Base-corpus loss rises after adaptation; the second curve makes this cost visible.

The character Q&A run has final training-batch loss **0.013** but held-out loss **5.802**: it memorizes practiced forms without reliably handling new wording. The word Q&A result is better on these ten held-out phrasings, but the concepts and answers are shared with training. Neither result is an unseen-fact test.

## Asking “what is a cat?”

Choose **Authored Q&A exercise**, **Question → answer**, enter `what is a cat?`, and generate. Both saved models actually generate:

> a cat is a small mammal. it has fur and is often kept as a pet.

This question is explicitly in training. It is not an answer lookup in the browser; Python runs the checkpoint autoregressively. Then try `please explain a cat.`, and compare the common base using the same input/format. Expect failures on novel wording or concepts, especially in character mode. Unknown vocabulary is reported rather than secretly discarded. Character mode can express more novel word combinations, but accepting characters does not create knowledge.

The biology CSV does not supply this cat definition. Prose adaptation and question-answer training are different tasks. Raw biology/email/finance output can remain repetitive, malformed or false after 500 updates. For example, the saved word biology continuation calls a blue mussel a carnivore. Keep this failure visible; use it to motivate validation.

## Python training and progress plots

Use **new output directories**:

```bash
python -m llm_workshop.domains.finetune \
  --tokenization word --device cuda --base-steps 1000 --steps 500 \
  --data-dir data/external --output-dir runs/domains500_words_next --live-plot

python -m llm_workshop.domains.finetune \
  --tokenization character --device cuda --base-steps 1000 --steps 500 \
  --data-dir data/external --output-dir runs/domains500_characters_next --live-plot
```

`--live-plot` opens a desktop plot; omit it for headless operation. Both modes always write progress JSON and PNG during training, and SVG at completion. Each plot shows latest-batch training loss, branch validation and original-base validation. Latest-batch loss is not a full-corpus training average.

To publish your new recordings into the local browser viewer:

```bash
python -m llm_workshop.domains.export_finetune \
  --run-dirs runs/domains500_words_next runs/domains500_characters_next
```

Restart the local server after changing its checkpoint catalog. Checkpoints support inference, not exact optimizer resume. Inspect `prepare.py`, `encode_rows` in `finetune.py`, then its batch → loss → backward → optimizer step loop to show the Python guts.

## A short teaching route

1. Character chapter **06**: name the original run and its base. Compare weights/output with the prompt fixed. Its 200-update LoRA domain examples are separate from this new CSV experiment.
2. Open **500-update domain lab**: choose Words, Biology. Point to the lineage, 9,500 training / 500 validation records, before/after loss and one real input row. Switch base/after and regenerate the same prompt.
3. Choose Email: show instruction versus scored response. Choose Finance: explain that correct-looking numbers need independent checking.
4. Choose Q&A: ask the practiced cat question, then a held-out phrasing. Distinguish input acceptance, learned response format and reliable answering.
5. Chapter **07**: choose one teaching method, name a learner deliverable and validation criterion. The retrieval activity assembles evidence into a prompt; it does not generate an answer or update weights.
6. Chapter **08**, Q2: the row means **attention weights after masking and softmax, with dropout off**. Its percentages mix visible positions; they are neither vocabulary probabilities nor confidence that an answer is true. The quiz now shows the chosen answer, correct answer and reason.

The original 32-entry character and 52-entry word companions remain the clean matrix-inspection route. Their data/model labels now identify their own runs. Use the larger lab for the new datasets and question-answer experiment. Existing biology→finance replay results remain available as a separately labeled comparison experiment.
