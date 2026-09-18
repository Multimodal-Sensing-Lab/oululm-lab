"""Generate from a CSV checkpoint; --text is the actual input to the model."""
import argparse
from pathlib import Path
from .train import load, generate
from llm_workshop.words.model import device_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--text', default='the blue mussel')
    parser.add_argument('--tokens', type=int, default=48)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if not 1 <= args.tokens <= 384:
        parser.error('Choose 1–384 new tokens.')
    model, tokenizer, state = load(args.checkpoint, device_for(args.device))
    print('Stage:', state['stage'], '| Tokenization:', tokenizer.metadata()['rule'])
    print('Input:', repr(args.text))
    print('Input pieces:', tokenizer.pieces(args.text))
    print('Greedy output:', generate(model, tokenizer, args.text, args.tokens))


if __name__ == '__main__':
    main()
