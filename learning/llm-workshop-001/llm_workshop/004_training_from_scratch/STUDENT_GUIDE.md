# 004: Training from scratch — student guide

**Goal:** Add loss.backward() and an optimizer to turn predictions into learning.

## Try it

```bash
python llm_workshop/004_training_from_scratch/demo.py --step
python llm_workshop/004_training_from_scratch/ui.py
```

1. Reset the shared model for a from-scratch comparison. Record the initial sample and loss.
2. Choose 100 updates and start. Explain one update before discussing an epoch.
3. Watch both curves. Pause with Stop; pending updates cease between steps.

## Explain what changed

Can lower training loss mean memorization? Yes. Which data should choose settings? Validation. Why does training need more memory than inference? It also needs activations for differentiation, gradients and optimizer state.

## Remember

These are tiny corpora and fixed diagnostic windows, not a broad language benchmark. Classroom save files are inference/comparison checkpoints; use llm_workshop.tiny for the longer resumable training workflow. Samples are collected at initialization, every 50 updates and final save.

[Visual topic page](index.html)
