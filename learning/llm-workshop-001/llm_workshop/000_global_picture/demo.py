"""Walk through the model pipeline and distinguish training from inference."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_workshop.teaching import common_parser, show

STAGES = [  # Overview of the stages explored in topics 001–008.
    ('Tokenizer', 'Text -> token pieces -> integer IDs. The mapping is fixed, not random per prompt.'),
    ('Embeddings + positions', 'Look up a vector for each ID and add position information. These are model parameters.'),
    ('Transformer blocks', 'Normalize -> masked attention -> residual addition -> normalize -> feed-forward -> residual addition. Repeat.'),
    ('LM head', 'Normalize final features and project to one raw score (logit) per vocabulary token.'),
    ('Softmax', 'Convert vocabulary scores to probabilities. This is separate from the LM head.'),
    ('Training branch', 'Known targets -> loss -> backward gradients -> optimizer update. Repeat with new batches; validate separately.'),
    ('Inference branch', 'Select a next token -> append -> repeat -> decode to text. Model weights stay fixed.'),
]


def main():
    args = common_parser(__doc__).parse_args()  # Read terminal options, including --step.
    for title, detail in STAGES:
        show(title, detail, args.step)  # Print a stage; optionally pause.
    print('The UI lets you select each stage; topic 001 processes editable text.')


if __name__ == '__main__':
    main()
