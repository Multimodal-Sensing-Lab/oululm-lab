# Workshop 001 · LLM basics: Open the model

A participant toolkit for inspecting how text becomes tokens, how a tiny Transformer predicts a continuation, and how training changes the result. It includes runnable Python, the interactive browser, data and saved weights.

## Start in the browser

![Interactive learning interface for Workshop 001](../../assets/Screenshot_interactive_browser_workshop_001.png)

*Screenshot of the interactive interface for learning how language models work.*

Open **[the character workshop](index.html)**. Follow its eight chapters; use the **[word companion](llm_workshop/browser/words.html)** to see the same operations with whole words. Both calculate real predictions locally in JavaScript. Training curves and before/after checkpoints are recorded Python runs; editing a prompt never trains a model.

- [Participant guide](llm_workshop/student-guide.html) · [Markdown version](STUDENT_GUIDE.md)
- [Glossary with examples](llm_workshop/glossary.html)
- [Detailed reading](llm_workshop/001_text_to_predictions/walkthrough.html)
- [Python lab and scripts](llm_workshop/python-lab.html)
- [References and acknowledgements](llm_workshop/references.html) · [Full bibliography (Markdown)](REFERENCES.md)
- [Dataset provenance](data/README.md) · [Checkpoint inventory](runs/README.md)

## Python setup

Run commands **from this workshop directory**, not from the repository root:

```bash
cd learning/llm-workshop-001
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python llm_workshop/001_text_to_predictions/tokenizer_demo.py --step
python llm_workshop/001_text_to_predictions/bpe_demo.py --word lower --step
python llm_workshop/003_attention_and_transformer/demo.py --step
```

The tested development environment uses Python 3.11 and PyTorch 2.5.1. Core requirements allow compatible later versions; exact numerical comparisons use explicit tolerances. Desktop `ui.py` examples also require Tk support in your Python installation. They are optional; terminal and browser routes work without Tk.

### Live domain lab

```bash
python -m llm_workshop.domains.serve
```

Open **http://127.0.0.1:8765/llm_workshop/browser/domain-lab.html**. This server loads the included checkpoints and computes on your CPU, without an external API. Use `--port 8766` if needed. It serves this workshop directory; keep private files outside it. Stop with Ctrl+C.

Select a branch, identify its trained task and compare the same input before/after. Biology and finance learned prose; Email learned narrow instruction/response patterns; Q&A practised explicitly authored questions. These tiny models are not useful general assistants. Unknown words and failed outputs are part of the experiment. See [the method](llm_workshop/domains/FINETUNING_GUIDE.md).

## Train your own experiment

```bash
python -m llm_workshop.words.train --device cpu --steps 600 --output-dir runs/my_words --live-plot
python -m llm_workshop.words.demo --checkpoint runs/my_words/after.pt --text 'the small cat' --inspect --step
```

For CUDA, use a compatible CUDA-enabled PyTorch installation and `--device cuda`. CPU is sufficient for the small examples. Use a **new output directory** for every experiment. Omit `--live-plot` on a headless machine; progress images are still saved. New runs are ignored by Git; the supplied reference runs are included deliberately. Reference checkpoints support inference/comparison; see individual trainers for resume support.

## Optional pretrained-model extensions

The numbered Python topics also include optional Hugging Face training/evaluation examples. Install `requirements-training.txt` only for those extensions. They download models/data separately, require their upstream terms and may need substantially more memory. Quantized training additionally uses `requirements-qlora.txt`. The default browser and tiny-model labs require neither pretrained downloads nor access tokens.

## Reproduce checks

```bash
python -m unittest discover -s tests -p 'test_teaching_demos.py'
python -m unittest discover -s tests -p 'test_words.py'
node llm_workshop/browser/test_experiments.cjs
node llm_workshop/browser/test_words.cjs
node llm_workshop/browser/test_browser.cjs
node llm_workshop/browser/test_word_browser.cjs
# Keep the local Python server running for:
node llm_workshop/browser/test_domain_lab.cjs
```

The optional pretrained-model test suite (`test_workshop.py`) additionally requires `requirements-training.txt`. With those installed, run `python -m unittest discover -s tests` for all 22 tests.

Browser tests require Node.js and Chrome/Chromium; set `CHROME_BIN` for another executable. Numerical tests use the included recorded fixtures. No browser automation package is needed.

## Attribution and supporting material

To cite this workshop, use the [repository citation and BibTeX entry](../../README.md#how-to-cite) and identify **Workshop 001: LLM basics — Open the model**, together with the release or commit used.

Created and curated by **Dr. Constantino Álvarez Casado**, Postdoctoral Researcher, University of Oulu. The code was developed mostly with AI assistance from **ChatGPT 5.6 Sol, 6 Astra, and Claude Opus 5**, together with the author. These tool/model names are recorded as supplied by the author. The author selected the educational goals, iterated on the implementation and reviewed the examples. AI assistance does not guarantee correctness; inspect the code, tests and measured outputs.

Adapted teaching context from the supplied **OpinTori (Constantino Alvarez, 2026)** and scoping-review presentation. Additional sources include Stanford CS324, the original Transformer paper, Jay Alammar, 3Blue1Brown, Transformer Explainer, Brendan Bycroft and the Q/K/V article by Ebrahim Pichka. The [reference page](llm_workshop/references.html) preserves source links and distinguishes original implementation from inspiration.

## License

Original toolkit code, explanations and tiny model weights: [MIT](LICENSE). Synthetic CSV and external-source exceptions: [third-party notices](THIRD_PARTY_NOTICES.md). No instructor scripts, slides or facilitator guides are included.
