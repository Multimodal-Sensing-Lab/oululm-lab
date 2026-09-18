"""Actual base/adapter evaluation entry point; --help lists data and retrieval options."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_workshop.evaluate import main

if __name__ == '__main__':
    main()
