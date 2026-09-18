# Open the model — visual browser workshop

**New 500-update experiments:** [Open the domain and question lab](domain-lab.html) · [method, data and commands](../domains/FINETUNING_GUIDE.md). Each tokenization has one 1,000-update classroom base and independent 500-update biology, email and finance branches. A separate authored Q&A branch accepts `what is a cat?`. Start `python -m llm_workshop.domains.serve` for live local generation; saved previews and datasets work offline. The earlier experiments below remain separate.

The **[whole-word companion](words.html)** is also ready offline. It uses the same transformer engine with its own 52-entry tokenizer, controlled 576-sentence corpus and measured GPU-trained weights. See [the word guide](../words/README.md) for the 15–20 minute route, Python source, commands, evaluation limits and tests. Character and word checkpoints/sessions are separate. The word route includes a separate optional Python LoRA experiment.

Open **[index.html](../../index.html)** directly in a browser. No build, server, network connection, API key or package installation is required. Keep all files in this directory together. The original Tk/Python lab is preserved and has a separate session.

The [participant guide](../student-guide.html) explains the learning route; [references](../references.html) document sources and methods.

## What runs where

**Character chapters 05/06 now include more recorded experiments:** original and longer CPU/GPU training on two corpora, plus biology/email/finance LoRA comparisons. See [data counts, commands and teaching route](CHARACTER_EXPERIMENTS.md). The original 200-update checkpoint is preserved.

- `engine.js`: real TinyGPT forward computation, independent of the interface; plain JavaScript arrays and numerical operations.
- `model-data.js`: two exported parameter sets (untrained seed 1337 and 200 CPU training updates), configuration, vocabulary, measured loss history and samples.
- `experiment-data.js`: additional measured CPU/GPU checkpoints at 200/1,000 updates and three merged domain adapters, with corpus counts, provenance and losses.
- `record_experiments.py`: records those experiments on the requested devices; keeps the original export intact. Saved files are inference comparisons, not optimizer-resume checkpoints.
- `app.js`, `style.css`, `index.html`: responsive visual presentation, SVG architecture map and attention connections, inspectable matrices, progressive operations, sampling, activities and exit questions.
- `export_model.py`: reproducible export using the existing Python Session and TrainingRun. It does not replace the raw numerical teaching scripts.
- `test_engine.cjs`: checks browser computations against exported PyTorch reference results and checks causality, sampling, inputs and retrieval.
- `test_browser.cjs`: launches local headless Chrome through its debugging pipe to exercise the real page without browser automation dependencies.

The browser trains no weights. The training chapter shows measured Python training; its before/after buttons select checkpoints and change the live checkpoint preview. The recorded curve and paired samples remain fixed for that run. Prompt edits recompute actual inference. Generating one token samples the full vocabulary, including the reserved ID, or uses greedy selection. Switching checkpoints resets the generated continuation to the typed prompt. The context is limited to 64 IDs and stops when full rather than silently cropping. The evidence activity performs live lexical retrieval and displays the assembled prompt, without claiming to generate an answer.

## Rebuild and verify

From learning/llm-workshop-001, using the existing workshop Python environment:

```bash
python llm_workshop/browser/export_model.py --references /tmp/llm-browser-reference.json
node llm_workshop/browser/test_engine.cjs /tmp/llm-browser-reference.json
node llm_workshop/browser/test_browser.cjs
```

The export deliberately regenerates the bundled files. It records data hashes and the PyTorch version for provenance. With the bundled run, training loss moves from 3.6224 to 1.4313; validation loss moves from 3.6513 to 2.2045. This is a small story-model demonstration, not a capability benchmark. Training and validation use separate source documents and fixed evaluation windows.

The numerical test checks more than 60,000 scalar values, including both blocks and heads. Tolerance is `3e-5 + 1e-5 * abs(reference)`. JavaScript uses double precision and an erf approximation in GELU; no bitwise equivalence claim is made. The browser test requires Node with built-in Web APIs and a local Chrome/Chromium executable (set `CHROME_BIN` to override `google-chrome`). It uses temporary profiles, never a personal browser profile. Screenshots are written to a temporary directory and its path is printed.

## Teaching and accessibility

Eight chapters can be explored at your own pace. Every chapter has “The idea”, “The maths” and “Python” panels. Participant explanations are available in the side panel. Matrix cells expose exact values through accessible labels and native titles; selected values also appear as text. Keyboard users can navigate buttons, focus cells and switch explanation tabs with arrow keys. Reduced-motion settings are respected, content reflows for narrow screens, and the guide can be printed. Input, evidence and quiz answers are not persisted or transmitted.

Signed colors differ from the Python lab's coolwarm palette; each interface labels its own scale. Values are the source of truth. The feature-column selector exposes all 64 embedding coordinates in groups of eight. Attention plots label query rows and key columns; full-width residual and FFN views combine every head.

## Scope and credits

This is an original interface inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) and [Brendan Bycroft's LLM Visualization](https://bbycroft.net/llm), with no copied third-party implementation or artwork. See the guide for further reading. The reference checkout under `external_resources/` is unchanged. The browser uses the repository's small character TinyGPT rather than downloading GPT-2, keeping every operation traceable to the workshop's raw Python code and making offline classroom use straightforward.

## Training data and larger CSV experiments

Open [Training data & domain experiments](data-lab.html) from the visible data links in character chapters 05/06 or word chapter 04. Browse the exact story, controlled-word, filtered-character, LoRA and CSV input records. The new CSV character/word runs include biology pretraining, finance full fine-tuning, replay comparison, settings, curves and actual samples. They are recorded Python runs; the original browser checkpoints remain the live-inference teaching fixtures.

See [the domain experiment guide](../domains/README.md) for input counts, measured RTX 2080 SUPER timings, checkpoint paths, progressive plots, limitations and export commands.
