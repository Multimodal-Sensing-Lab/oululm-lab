"""Short conventional entry point for the character-tokenizer terminal lesson."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module
main = import_module('llm_workshop.001_text_to_predictions.tokenizer_demo').main  # Use the detailed tokenizer lesson.

if __name__ == '__main__':
    main()
