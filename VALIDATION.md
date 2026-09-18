# Public package validation

Checked on 18 September 2026 in the configured Python 3.11 / PyTorch 2.5.1 environment, with Node.js and headless Chrome. This records tests of the copied package; it is not a claim of a clean installation on every supported platform.

- All 22 Python tests passed, including training mechanics, tokenization, masks, adapter isolation and optional local Hugging Face tests.
- Character recordings: 4,224 independent PyTorch/JavaScript logit comparisons passed, plus corpus hashes and LoRA merge checks.
- Word engine: 44,332 scalar comparisons passed, including causality, positions, masking and generation.
- Browser interaction suites passed for character chapters, word stages, recorded training, the data explorer and live domain generation.
- Participant glossary, detailed reading, Python guide, references and student route passed desktop/mobile checks; selected screenshots were visually reviewed.
- The live domain suite used a separate server started from this workshop copy, not the original author workspace.
- All ten live-domain checkpoint hashes match their reports. All 42 distributed checkpoint files are byte-identical to their original recorded files.
- Biology, finance and email generators reproduce their included CSVs byte-for-byte after the portable output-path change.
- Every local HTML/Markdown link resolves. All 110 unique linked workshop resources returned HTTP 200 from the standalone local server.
- No slide decks, instructor guides, private workshop hub, environment files or download caches are included. Git ignores newly generated runs and caches while retaining the reference artifacts.
- The bundle is approximately 235 MiB, including datasets and weights. No individual file exceeds 100 MiB.

Run `node tools/check-release.cjs` from the repository root for a repeatable structural/checkpoint check. The workshop README lists the Python and browser tests. Browser fixtures and model outputs are measured artifacts; packaging does not improve the tiny models' capabilities.
