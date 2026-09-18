"""Entry point to actual pretrained SFT/LoRA/QLoRA; see the topic guide first.

Example: python llm_workshop/006_domain_fine_tuning/train_pretrained.py --config configs/lora_finance.yml
This can download a model and use the configured GPU. It is not launched by the desktop toy lab.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_workshop.finetune import main

if __name__ == '__main__':
    main()
