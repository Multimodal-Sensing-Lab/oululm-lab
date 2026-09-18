# 006: Domain adaptation and LoRA — student guide

**Goal:** Update a small parameter subset, then compare against the identical base.

## Try it

```bash
python llm_workshop/006_domain_fine_tuning/demo.py --step
python llm_workshop/006_domain_fine_tuning/ui.py
```

1. First train/load the tiny base from 004, or explicitly label a random-base run as mechanics only.
2. Select a domain and rank. Start a new adapter and watch its delta matrix and training loss.
3. Compare greedy base and adapter text for the same prompt.

## Explain what changed

Why does rank matter? It changes the capacity and number of parameters in the update. What is QLoRA? Training adapters over a quantized frozen base, supported in the real pretrained pipeline. Is the tiny UI pretending to run QLoRA? No.

## Remember

Tiny adapter_matrices.pt files are not PEFT format. merged.pt is a normal TinyGPT checkpoint. Prose adaptation is not supervised instruction fine-tuning. Neither teaches guaranteed factual recall.

[Visual topic page](index.html)
