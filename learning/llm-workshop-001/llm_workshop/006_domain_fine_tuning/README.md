# 006: Domain adaptation and LoRA

Update a small parameter subset, then compare against the identical base.

[Topic page](index.html) · [Student guide](STUDENT_GUIDE.md)

```bash
python llm_workshop/006_domain_fine_tuning/demo.py --step
python llm_workshop/006_domain_fine_tuning/ui.py
```

Tiny adapter_matrices.pt files are not PEFT format. merged.pt is a normal TinyGPT checkpoint. Prose adaptation is not supervised instruction fine-tuning. Neither teaches guaranteed factual recall.
