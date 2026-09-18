"""Launch the native Python visual demo for topic 005."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_workshop.visual.app import launch

if __name__ == '__main__':
    launch(5)
