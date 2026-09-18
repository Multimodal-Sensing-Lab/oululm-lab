"""Compare two word checkpoints with the same prompt and greedy decoding."""
import argparse
from pathlib import Path
import torch
from .model import load_checkpoint, generate


def compare(left, right, text='the small cat', tokens=12):
    torch.set_num_threads(2)
    loaded = [load_checkpoint(p) for p in [left, right]]
    if loaded[0][1].vocabulary != loaded[1][1].vocabulary:
        raise ValueError('Comparison requires the same tokenizer vocabulary.')
    return [{'checkpoint': str(path), 'step': state['step'],
             **generate(model, tokenizer, text, tokens=tokens)}
            for path, (model, tokenizer, state) in zip([left, right], loaded)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--text', default='the small cat')
    parser.add_argument('--tokens', type=int, default=12)
    args = parser.parse_args()
    for result in compare(args.left, args.right, args.text, args.tokens):
        print(result['checkpoint'], 'step', result['step'], '\n', result['text'], '\nStop:', result['stop'])
