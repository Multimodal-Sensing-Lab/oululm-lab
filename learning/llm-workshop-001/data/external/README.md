# Synthetic Text Datasets for Small Language Model Experiments

Three small synthetic English text corpora for training and fine-tuning tiny GPT-style
language models (nanoGPT / minGPT / TinyStories scale) on a local machine:

1. `animal_biology_sentences.csv` — technical sentences about animal biology (pretraining)
2. `finance_sentences.csv` — technical sentences about finance (domain-adaptation fine-tuning)
3. `email_finetune.csv` — instruction/response pairs for email writing (instruction fine-tuning)

They were designed as a progression: pretrain on domain A, fine-tune on domain B to study
domain adaptation and catastrophic forgetting, then instruction-tune on a task format.

---

## 1. `animal_biology_sentences.csv`

| Property | Value |
|---|---|
| Rows | 10,000 unique sentences |
| File size | 1.15 MB |
| Characters | ~950,000 |
| Mean / max sentence length | 95 / 191 characters |
| Columns | `id`, `category`, `sentence` |

**Content.** 54 animal species across 11 phyla (Porifera, Cnidaria, Platyhelminthes, Nematoda,
Annelida, Mollusca, Arthropoda, Tardigrada, Echinodermata, Chordata, and others). Each species is
described in terms of taxonomy, trophic ecology, habitat, thermoregulation, respiration,
circulation, nitrogenous waste, reproduction, and body symmetry, plus specific traits. The set also
contains ~120 general concepts of comparative animal physiology (Kleiber's law, the Bohr effect,
countercurrent exchange, protostome/deuterostome development, and so on).

**Category distribution.** `comparative` 7,416 · `concepts` 610 · `traits` 516 · `taxonomy` 432 ·
`physiology` 432 · `ecology` 270 · `reproduction` 108 · `summary` 108 · `morphology` 108.

**Note.** The `comparative` category dominates because it is generated from all species pairs. Use
the `category` column to downsample it if you want a more balanced corpus.

---

## 2. `finance_sentences.csv`

| Property | Value |
|---|---|
| Rows | 4,000 unique sentences |
| File size | 0.54 MB |
| Characters | ~460,000 |
| Mean / max sentence length | 115 / 243 characters |
| Columns | `id`, `category`, `sentence` |

**Content.** 18 financial instruments (stocks, bonds, T-bills, ETFs, futures, options, swaps, REITs,
MBS, and others), 20 financial ratios with their formulas and interpretation, ~110 core concepts
(time value of money, NPV, IRR, CAPM, beta, duration, yield curve, accounting statements, monetary
policy, Basel, KYC/AML), and ~2,500 worked numeric examples.

**Category distribution.** `calculations` 2,496 · `comparative` 572 · `concepts` 540 ·
`instruments` 252 · `ratios` 140.

**Numeric examples.** Compound and simple interest, present value, amortizing loan payments, bond
current yield, P/E, ROE, EPS, dividend yield, market capitalization, percentage change, and real
interest rates via the exact Fisher relation. Figures were computed in Python; the generated calculations have not all been independently audited. Amounts appear in euros, dollars, and pounds.

**Not financial advice.** The dataset is educational/technical text intended as training material.

---

## 3. `email_finetune.csv`

| Property | Value |
|---|---|
| Rows | 1,686 unique instruction/email pairs |
| File size | 2.66 MB |
| Characters (`text` field) | ~1,334,000 |
| Mean / max example length | 791 / 1,136 characters |
| Columns | `id`, `domain`, `email_type`, `instruction`, `incoming_email`, `email`, `text` |

**Content.** Professional emails in two domains: computer science / software engineering (930 rows)
and marketing (756 rows), across 17 email types. 410 rows are *reply* tasks that include an incoming
email in the `incoming_email` column; the rest are *write from instruction* tasks, where that column
is empty.

**Email types.** CS: bug report, code review request, incident report, meeting request, maintenance
announcement, security notice, onboarding, reply to a technical question, reply declining a request.
Marketing: product launch, newsletter, webinar invitation, cold outreach, follow-up, re-engagement,
reply to a complaint, reply to a partnership proposal.

**The `text` column** concatenates everything into a ready-to-train string:

```
### Instruction:
<what to write>

### Incoming email:        (only present for reply tasks)
<the email being answered>

### Email:
Subject: ...
<body>
<|endoftext|>
```

All persons, companies, and products are fictional. No real individuals or organizations are
referenced.

---

## How the data was created

**Generating model:** Claude Opus 5 (Anthropic), used through the Claude chat interface in
September 2026.

**Method — hybrid, not free-form generation.** The model did not write the rows one by one. For each
dataset it authored a Python generator script consisting of:

1. a hand-curated knowledge base written by the model (species attributes, financial instruments and
   ratios, concept statements, email scenarios and body texts);
2. a set of sentence or email templates with slots;
3. deterministic Python code that combines the two, computes the numeric examples, removes
   duplicates, shuffles the rows with a fixed random seed, and writes the CSV.

The generator scripts are included so the datasets are fully reproducible and extensible:

- `make_biology.py` (seed 42) → `animal_biology_sentences.csv`
- `make_finance.py` (seed 7) → `finance_sentences.csv`
- `make_emails.py` (seed 11) → `email_finetune.csv`

Run with `python <script>` (Python 3, standard library only). The public copies write next to the script. Copy a script into a new directory before running it if you want to preserve the bundled CSV.

**Encoding and format.** UTF-8, RFC 4180 CSV with a header row, written by Python's `csv` module;
fields containing commas, quotes, or newlines are quoted. The `email_finetune.csv` fields contain
embedded newlines, so read it with a real CSV parser (`pandas.read_csv` or `csv.reader`), not by
splitting on `\n`.

**Quality and provenance.** Factual content comes from the generating model's knowledge of general
textbook material; no text was copied from books, articles, or websites. The statements were written
to be factually accurate at the level of an undergraduate textbook, but they have not been reviewed
by a domain expert, and the biology and finance files should not be treated as an authoritative
reference.

## Known limitations

- **Low linguistic diversity.** Because of the template-based method, sentence structures repeat.
  A model trained on this data learns the templates as much as it learns the domain. For more
  natural language, mix in real text (for example, open-licensed encyclopaedia or abstract corpora).
- **Small scale.** ~950k, ~460k and ~1.3M characters respectively. This is adequate for
  character-level tiny models (comparable to the TinyShakespeare corpus), but small for BPE
  token-level training, where it amounts to only a few hundred thousand tokens.
- **Domain imbalance within files.** See the category distributions above.
- **English only**, and Western-European naming conventions in the email set.
- **No harmful, personal, or private data.** All content is synthetic; names are invented.

## Suggested use

```python
import pandas as pd
bio = pd.read_csv("animal_biology_sentences.csv")
corpus = "\n".join(bio["sentence"])          # pretraining text

emails = pd.read_csv("email_finetune.csv")
sft = emails["text"].tolist()                # instruction fine-tuning examples
```

For fine-tuning, use a learning rate well below the pretraining one, train for few epochs, mask the
loss on the prompt part of `text`, and hold out a validation split from *both* domains so you can
watch adaptation and forgetting at the same time.

## License and attribution

The supplied CSVs retain **CC0 1.0 Universal** (https://creativecommons.org/publicdomain/zero/1.0/). Generator scripts use the repository MIT license. Creation attribution: Dr. Constantino Álvarez Casado, University of Oulu, with Claude Opus 5-assisted deterministic generators, September 2026.

See [the public data inventory](../README.md) for exact counts, splits, limitations and the distinction between original workshop material and optional downloaded sources.
