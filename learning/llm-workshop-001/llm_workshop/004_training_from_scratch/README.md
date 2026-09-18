# 004: Training from scratch

Add loss.backward() and an optimizer to turn predictions into learning.

[Topic page](index.html) · [Student guide](STUDENT_GUIDE.md)

```bash
python llm_workshop/004_training_from_scratch/demo.py --step
python llm_workshop/004_training_from_scratch/ui.py
```

These are tiny corpora and fixed diagnostic windows, not a broad language benchmark. Classroom save files are inference/comparison checkpoints; use llm_workshop.tiny for the longer resumable training workflow. Samples are collected at initialization, every 50 updates and final save.
