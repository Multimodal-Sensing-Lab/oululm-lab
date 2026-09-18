# Included reference runs

These are tiny models trained for the workshop, not downloaded foundation models. They support local inference and before/after comparisons. Current browser selection and task format determine which checkpoint is used.

| Run | Purpose |
| --- | --- |
| classroom_tiny_01 | Original 100-update terminal character experiment |
| classroom_biology_01 | Small biology adapter on that terminal base |
| words_classroom_01 | Controlled word model, 600 updates; 52-entry vocabulary |
| words_garden_01 | Word adapter mechanics example |
| browser_character_20260917_01 | Recorded 200/1,000-update CPU/GPU character experiments and LoRA branches |
| domains_words_20260917_01 | Earlier biology → finance/replay experiment, words |
| domains_characters_20260917_01 | Earlier biology → finance/replay experiment, characters |
| domains500_words_20260917_02 | Common 1,000-update base + independent 500-update biology/email/finance/Q&A branches; 3,348 entries |
| domains500_characters_20260917_01 | Matching branch design with characters; 61 entries |

The original character browser's 200-update weights are also bundled directly in `llm_workshop/browser/model-data.js`. Word and additional character weights are exported to adjacent JavaScript artifacts. The domain catalog selects the final two directories for live Python inference.

Reports, source snapshots, data manifests, prepared splits and plots accompany the runs. Checkpoint files are copied unchanged; hashes recorded in reports remain verifiable. Treat source snapshots as historical recording provenance; current runnable code is in `llm_workshop/`. Reports and labels may reflect the original recording dates/hardware. No optimizer resume is promised by these classroom exports.

Original tiny model weights use the repository MIT license. They are learning artifacts, not subject experts or reliable assistants. New run directories are ignored by Git; use a fresh directory for your own training.
