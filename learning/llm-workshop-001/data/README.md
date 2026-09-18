# Data provenance and contents

## Included synthetic material

| Dataset | Records | Whitespace-separated words | Characters in text fields | Creation |
| --- | ---: | ---: | ---: | --- |
| Biology CSV | 10,000 | 132,494 | 949,989 | Claude Opus 5-assisted knowledge tables + deterministic templates |
| Finance CSV | 4,000 | 78,703 | 459,933 | Claude Opus 5-assisted templates; numeric examples computed by Python |
| Email CSV | 1,686 | 216,040 | 1,334,459 | Claude Opus 5-assisted scenarios/templates; counts use the serialized text column |
| Character stories, training | 3 stories / 12 sentences | 109 | 517 | Original fictional workshop examples; joined training stream adds four separator characters (521 total) |
| Controlled words, training | 432 sentences | 3,456 | 17,396 | Deterministic combinations of small subject/verb/location vocabularies |
| Authored Q&A | 30 train + 10 validation | 724 | 3,828 | Explicit questions and answers about 10 concepts; held-out wording |
| Fictional handbook | See prepared manifests | — | — | Seeded fictional instrument-to-room assignments; no actual university policy |

CSV files and scripts are in [external/](external/README.md). Character stories and controlled word data are in [examples/data/](../examples/data/). Exact domain training/validation records and counts are shipped beside each [checkpoint](../runs/README.md). The email CSV contains **930 computer-science and 756 marketing records**, with 17 email types and optional incoming messages. Do not count the repeated serialized text field twice when preparing training examples.

## How the data was made

The supplied CSV creation record credits **Claude Opus 5 (Anthropic), September 2026**, working with Dr. Constantino Álvarez Casado. The model helped author knowledge tables, template text and generator code. Python combines templates, computes finance examples, deduplicates and shuffles with fixed seeds. These are synthetic teaching corpora, not scraped textbook extracts or expert-verified benchmarks.

The original tiny stories, structured word corpus, fictional handbook and Q&A exercise were created for this workshop with the AI-assisted development described in the [workshop attribution](../README.md#attribution-and-supporting-material). The word generator is [corpus.py](../llm_workshop/words/corpus.py); the handbook generator is [data.py](../llm_workshop/data.py); the authored Q&A and split preparation are in [prepare.py](../llm_workshop/domains/prepare.py).

The three supplied scripts now write CSVs beside themselves, rather than to the original chat environment's private directory. To experiment without replacing the shipped data, copy a generator into a new directory and run it there. Regeneration should be checked against the included source hashes before reusing a recorded checkpoint report.

## Preparation and limits

Split records before fitting vocabulary; reserve validation examples; use one fixed vocabulary per model family. Training records become next-token targets, with prompt loss masked for email/Q&A. No new weights were trained for this public packaging step. The browser's data viewers expose raw sources, prepared rows, split counts and recorded losses.

Many examples share templates. Low loss may reflect memorization and does not establish broad factual knowledge. Generated finance numbers are not calculations performed by the model. Source calculations were generated programmatically but have not all been independently audited. Biology and finance statements are not authoritative references; email names and scenarios are synthetic.

## Optional downloaded datasets — not bundled

The public workshop does not redistribute the original workspace's Hugging Face cache or downloaded smoke-test subsets. The core browser and supplied tiny checkpoints do not depend on them. The [data loader](../llm_workshop/data.py) can prepare these optional experiments:

- [FinancialPhraseBank](https://huggingface.co/datasets/lmassaron/FinancialPhraseBank): financial sentences with sentiment labels; the referenced card lists CC BY-NC-SA 4.0.
- [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA): abstract-supported yes/no/maybe questions; the referenced card lists MIT.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): generated short stories; the referenced card lists CDLA-Sharing 1.0. The included three tiny stories are original workshop text, not a TinyStories extract.

Downloaded sources retain their own terms and attribution. Saved manifests record source revisions, splits and hashes. Original workshop-created data and weights are MIT; the three supplied synthetic CSVs retain CC0 and their scripts MIT. See [licensing notices](../THIRD_PARTY_NOTICES.md).
