"""Optional third demo: learn a few subword merge rules, then freeze and apply them.

A minimal character-based BPE illustration, NOT a production byte-level tokenizer.
It treats each supplied string as one word; no whitespace handling or special tokens.
Run: python llm_workshop/001_text_to_predictions/bpe_demo.py --word lower --step
"""
import argparse
from collections import Counter

# Repeated words make frequencies visible. Try changing their repetition counts.
TRAINING_WORDS = ['low'] * 5 + ['lower'] * 2 + ['newer'] * 3
NEW_WORD = 'lowest'
MERGE_COUNT = 4


def merge_pair(tokens, pair):
    """Scan left-to-right; replace non-overlapping occurrences of the chosen pair."""
    result = []
    index = 0
    while index < len(tokens):
        if tuple(tokens[index:index + 2]) == pair:
            result.append(''.join(pair))
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', action='store_true')
    parser.add_argument('--word', default=NEW_WORD)
    args = parser.parse_args()
    words = [list(word) for word in TRAINING_WORDS]
    vocabulary = sorted(set(''.join(TRAINING_WORDS)))
    rules = []
    print('Tokenizer training data:', Counter(TRAINING_WORDS))
    print('Initial alphabet:', vocabulary)
    for step in range(MERGE_COUNT):
        counts = Counter(pair for word in words for pair in zip(word, word[1:]))
        if not counts:
            break
        # Most frequent adjacent pair. Alphabetical tie-break makes this deterministic.
        pair = min(counts, key=lambda item: (-counts[item], item))
        rules.append(pair)
        merged_token = ''.join(pair)
        if merged_token not in vocabulary:
            vocabulary.append(merged_token)
        words = [merge_pair(word, pair) for word in words]
        print(f'\nMerge {step+1}: {pair} -> {merged_token!r}, observed {counts[pair]} times')
        print('Training words now:', words)
        if args.step:
            input('Press Enter for next merge... ')
    print('\nFREEZE rules and vocabulary. No neural network was trained.')
    print('Vocabulary IDs:', dict(enumerate(vocabulary)))
    tokens = list(args.word)
    print('New input:', repr(args.word), 'starts as', tokens)
    for pair in rules:
        tokens = merge_pair(tokens, pair)
        print('Apply', pair, '->', tokens)
    unknown = sorted(set(tokens) - set(vocabulary))
    if unknown:
        print('Cannot encode with this alphabet:', unknown)
        print('Try --word lower. Real byte-level BPE can represent arbitrary UTF-8 bytes.')
    else:
        token_to_id = {token: i for i, token in enumerate(vocabulary)}
        ids = [token_to_id[token] for token in tokens]
        print('IDs:', ids, '\nDecoded:', ''.join(vocabulary[i] for i in ids))
    print('A subword may be part of a word OR a complete word. It is not a document chunk.')


if __name__ == '__main__':
    main()
