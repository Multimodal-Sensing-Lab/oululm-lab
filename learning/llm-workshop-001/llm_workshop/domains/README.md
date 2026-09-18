# Biology → finance: train, adapt, measure forgetting

**New 500-update experiments:** [Method, data and commands](FINETUNING_GUIDE.md) · [browser lab](../browser/domain-lab.html). Each tokenization has one 1,000-update classroom base and independent 500-update biology, email and finance branches. A separate authored Q&A branch accepts `what is a cat?`. Start `python -m llm_workshop.domains.serve` for live local generation; saved previews and datasets work offline. The earlier experiments below remain separate.

Open [the offline data lab](../browser/data-lab.html) from either browser route. It contains the real input records, category/search/split filters, measured curves, exact settings and unedited generated examples. The new CSV checkpoints are separate from the original 32-entry character and 52-entry word models. The CSV page displays recordings; use Python for new input prompts.

## What is the input?

Only the CSV **sentence** field is model input. `id` and `category` are metadata. Counts below sum the original text fields, excluding CSV quoting, headers and record separators. A record can contain more than one grammatical sentence.

| Source | Unique records | Whitespace words | Text characters | Complete file bytes | Train / validation records |
|---|---:|---:|---:|---:|---:|
| [Biology](../../data/external/animal_biology_sentences.csv) | 10,000 | 132,494 | 949,989 | 1,148,649 | 9,500 / 500 |
| [Finance](../../data/external/finance_sentences.csv) | 4,000 | 78,703 | 459,933 | 541,196 | 3,800 / 200 |

Biology: comparative 7,416; concepts 610; traits 516; taxonomy 432; physiology 432; ecology 270; reproduction, summary and morphology 108 each. Comparative records dominate (74.16%).

Finance: calculations 2,496 (62.4%); comparative 572; concepts 540; instruments 252; ratios 140. Category metadata makes the imbalance visible. This recording uses every training record, without category downsampling. The supplied arithmetic has not been independently audited; generated arithmetic is certainly not guaranteed correct.

### Original classroom data

| Route / split | Documents / sentences | Whitespace words | Characters |
|---|---:|---:|---:|
| Original characters: training | 3 stories / 12 sentences | 109 | 521 joined stream |
| Original characters: validation | 2 stories / 8 sentences | 75 | 367 joined stream |
| Controlled words: training | 432 sentences | 3,456 | 17,396 text fields |
| Controlled words: validation | 72 sentences | 576 | 2,897 text fields |
| Controlled words: test | 72 sentences | 576 | 2,875 text fields |

Character stream counts include two newlines between documents. For the three stories, the actual text fields total 517 characters, plus four separators = 521. The data viewer uses **text-field counts** consistently, and explains this difference. The character route's filtered controlled corpus retains 351 training / 55 validation records to stay within its original alphabet; it is not the complete word corpus. Tiny LoRA paragraphs are also browsable separately.

## Actual recorded GPU experiments

Files: [word report](../../runs/domains_words_20260917_01/report.json), [character report](../../runs/domains_characters_20260917_01/report.json).

Both use 2 blocks, 2 heads, width 64, dropout 0, AdamW and seed 1337. Pretraining: 5 epochs over biology, batch 32, **1,485 updates**, learning rate **0.003**. Adaptation: 3 epochs over finance, **357 updates**, learning rate **0.0003**. All parameters train: these are **full fine-tuning**, not LoRA.

| CSV model | Vocabulary | Context | Trainable parameters | Biology pretrain | Finance only | Finance + replay |
|---|---:|---:|---:|---:|---:|---:|
| Words + digits + punctuation | 2,567 | 64 | 435,335 | 15.2 s | 4.0 s | 4.1 s |
| Characters + special markers | 51 | 384 | 131,251 | 16.8 s | 4.6 s | 5.1 s |

Measured on **NVIDIA GeForce RTX 2080 SUPER (8 GB)**, PyTorch 2.5.1/CUDA 12.1. Times include each training loop, full periodic validation and progressive plot writes; exclude interpreter startup and final export. These are tiny models and one recording, not a GPU benchmark or a promise for larger architectures. Browser inference remains separate from Python training.

### Final validation loss (nats per target token; lower is better)

| Model / checkpoint | Biology | Finance |
|---|---:|---:|
| Words: biology base | 0.814 | 11.000 |
| Words: finance only | 2.455 | 2.233 |
| Words: finance + replay | 1.193 | 2.298 |
| Characters: biology base | 0.497 | 3.429 |
| Characters: finance only | 2.057 | 0.948 |
| Characters: finance + replay | 0.761 | 1.026 |

In both recordings, finance training improves finance prediction and worsens biology prediction. Replay reduces the biology degradation, at a small cost in finance loss. This is measured forgetting on these held-out records, not a guarantee about every run. Do not compare character loss directly with word loss: their units differ. Whole-word outputs can still be false: after finance adaptation, the word model completes “the blue mussel” with “is a derivative.”

## Splits, fixed vocabulary, and fair comparisons

- Exact 5% category-stratified validation, using largest-remainder allocation and a fixed seed. Splits are saved as JSONL. Exact duplicate text/IDs are rejected; cross-domain duplicate text is rejected. Template-level overlap can remain.
- Vocabulary is fitted on the **union of both training splits**, before any model training. Finance words get IDs in advance, but no finance gradients occur in biology pretraining. Held-out text never fits vocabulary. Both recordings have zero validation unknown tokens. This is a planned shared vocabulary experiment, not an unseen-vocabulary challenge.
- CSV word tokenization lowercases whole words, keeps internal apostrophes/hyphens, separates punctuation, and splits numeric strings into individual digits. It is **not BPE**. This avoids one vocabulary entry per numeric literal. Character tokenization lowercases and retains characters. Both add BOS/EOS/UNK.
- Each record starts fresh; no concatenation across CSV records. Targets are inputs shifted one token; EOS is a target, BOS is input only; padding is ignored. Context is fixed at 64 word/digit positions or 384 character positions. Oversized records raise an error instead of being silently truncated.
- Both adaptation runs begin with identical biology weights, use the same finance order and exactly 11,400 finance-record exposures. Replay adds 2,850 biology-record exposures: 8 per full batch of 32 finance records (20% of records, not necessarily tokens). It has more compute/data than finance-only.
- Evaluation uses **all** targets in the held-out records. The plotted training-batch point is the most recent batch before its update, not a full training-set evaluation.
- Fixed budgets retain the final checkpoint. Validation is reported, not used for early stopping or checkpoint selection. These are development probes; there is no third untouched test set in the CSV experiment. The original controlled word experiment does have a test split.
- Generated numeric patterns are not arithmetic skills. Domain prose models are not instruction assistants or reliable subject experts.

## Train with plots

From learning/llm-workshop-001, in the configured `llm-workshop` environment:

```bash
python -m llm_workshop.domains.train \
  --biology data/external/animal_biology_sentences.csv \
  --finance data/external/finance_sentences.csv \
  --tokenization word --device cuda \
  --output-dir runs/my_domain_words --live-plot

python -m llm_workshop.domains.train \
  --tokenization character --device cuda \
  --output-dir runs/my_domain_characters --live-plot
```

Use `--device cpu` without CUDA. Omit `--live-plot` on a headless machine: PNG plots still update at every evaluation (default every 100 updates), and SVGs are saved on completion. Matplotlib is declared in `requirements-training.txt`. The desktop window closes when training finishes; saved plots remain available. Every run needs a new output directory.

Saved: starting and three final inference checkpoints; full reports; exact split records; source CSV hashes; tokenizer/configuration; source-code snapshots; progressive JSON/PNG/SVG plots; fixed greedy samples. Checkpoints support inference/comparison, not exact optimizer resume. Source CSVs stay in `data/external`; keep them alongside the run artifacts.

```bash
python -m llm_workshop.domains.demo \
  --checkpoint runs/domains_words_20260917_01/biology_base.pt \
  --text 'the blue mussel'

python -m llm_workshop.domains.demo \
  --checkpoint runs/domains_words_20260917_01/finance_only.pt \
  --text 'the blue mussel'

python -m llm_workshop.domains.export --run-dirs \
  runs/my_domain_words runs/my_domain_characters
```

Export updates only the data-lab recordings, not the original browser model weights. It checks that the two runs used identical split records.

Original demonstrations now save training plots too:

```bash
python -m llm_workshop.words.train --device cuda --steps 600 \
  --output-dir runs/my_classroom_words --live-plot
python -m llm_workshop.words.adapt \
  --base-checkpoint runs/words_classroom_01/after.pt --family garden \
  --output-dir runs/my_garden_adapter --live-plot
python llm_workshop/004_training_from_scratch/demo.py --steps 200 \
  --output-dir runs/my_character_stories --live-plot
python llm_workshop/006_domain_fine_tuning/demo.py \
  --base-checkpoint runs/classroom_tiny_01/after.pt --domain biology \
  --output-dir runs/my_biology_adapter --live-plot
```

For the numbered character demos, plots are saved beside the fresh output directory as `<run>_progress.png/.svg`; `--plot-file` overrides that location. Word trainers save `progress.png/.svg` inside the run directory. The original numerical fixtures and browser checkpoints remain unchanged.

## Teaching route

1. Character chapter 05: identify what the current checkpoint saw. Three stories explain why the original output is limited.
2. Chapter 06: switch before/after; read the selected data summary. Distinguish updates from unique data and from epochs.
3. Word route → What training changed: 432 controlled training sentences. Same math, easier output units.
4. Open the data lab. Read a biology row and a finance row. Select their validation splits to make the held-out idea concrete.
5. Words → biology base → finance only → replay. Compare the two validation columns and actual samples. Then switch to characters, but compare checkpoints only **within** one tokenization.
6. Show `domains/train.py`: `batch` shifts targets, `loss.backward()` computes gradients, `optimizer.step()` changes weights. PNG curves update while this runs.
7. Finish with a scale reference: public large-model budgets are trillions of tokens, whereas these are intentionally narrow classroom tasks.

## Large-model data references

- [Meta, Llama 3.1 405B (July 2024)](https://ai.meta.com/blog/meta-llama-3-1/): over 15 trillion training tokens.
- [DeepSeek-V3 (December 2024)](https://arxiv.org/abs/2412.19437): 14.8 trillion pretraining tokens.
- [Qwen3 (2025)](https://qwenlm.github.io/blog/qwen3/): about 36 trillion pretraining tokens.

Dated public examples, not a current leaderboard or a minimum data requirement. Token budgets do not equal unique words, documents or bytes. Data quality/diversity and compute/model capacity matter. Full amounts are not disclosed for every modern model. A smaller adaptation set is useful here because we start from trained weights; there is no universal rule requiring every fine-tuning dataset to be smaller than every pretraining dataset.
