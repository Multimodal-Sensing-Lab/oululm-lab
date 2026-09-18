"""First live demo: characters -> vocabulary -> IDs -> original text.

No neural network, PyTorch, HTML, plots, or downloads are involved.
Run: python llm_workshop/001_text_to_predictions/tokenizer_demo.py --step
"""
import argparse
import json
from pathlib import Path

# Change TEXT first. The alphabet comes from CORPUS, not from this input.
TEXT = 'the cat sat.'
CORPUS = Path(__file__).resolve().parents[2] / 'examples/data/tiny_train.jsonl'


def read_corpus(path):
    """Use the same local stories and separators as the later TinyGPT demo."""
    with path.open(encoding='utf-8') as source:
        return '\n\n'.join(json.loads(line)['text'] for line in source if line.strip())


def build_vocabulary(corpus_text):
    """Deterministic rule: sorted unique characters, with a reserved unknown ID."""
    characters = sorted(set(corpus_text))  # One entry per distinct corpus character.
    vocabulary = ['<UNK>'] + characters  # Reserve ID 0; regular characters follow.
    token_to_id = {token: index for index, token in enumerate(vocabulary)}  # Assign stable integer addresses.
    return vocabulary, token_to_id


def encode(text, token_to_id):
    """Look up each character. IDs are table positions, not random numbers."""
    unknown = sorted(set(text) - set(token_to_id))
    if unknown:
        raise ValueError(f'Characters absent from this fixed vocabulary: {unknown!r}')
    return [token_to_id[character] for character in text]


def decode(ids, vocabulary):
    """Reverse the lookup. Our character tokenizer preserves spaces/case exactly."""
    return ''.join(vocabulary[token_id] for token_id in ids)


def pause(step):
    if step:
        input('Press Enter to continue (Ctrl+C to stop)... ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default=TEXT)
    parser.add_argument('--step', action='store_true', help='Pause after each concept')
    args = parser.parse_args()

    print('1. INPUT TEXT:', repr(args.text))
    print('We preserve spaces, punctuation and case. No lowercasing or cleanup.')
    pause(args.step)

    # A tokenizer can use characters, words, or learned subwords.
    # This first example explicitly chooses one Python character per token.
    tokens = list(args.text)  # Keep spaces and punctuation as tokens too.
    print('\n2. SPLIT INTO CHARACTER TOKENS:', tokens)
    print('For comparison, text.split() gives words:', args.text.split())
    print('That word split loses exact whitespace; it is NOT our encoder.')
    pause(args.step)

    vocabulary, token_to_id = build_vocabulary(read_corpus(CORPUS))  # Build from the corpus, not the prompt.
    print('\n3. FIXED VOCABULARY FROM THE STORY CORPUS:')
    for token_id, token in enumerate(vocabulary):
        print(f'  {token_id:2d} -> {token!r}')
    print('ID 0 is reserved for <UNK>; this demo rejects unsupported characters.')
    print('The corpus chose the alphabet. We have NOT trained a language model.')
    pause(args.step)

    ids = encode(args.text, token_to_id)  # Text -> addresses for topic 002 embeddings.
    print('\n4. ENCODE BY LOOKUP:')
    for position, (token, token_id) in enumerate(zip(tokens, ids)):
        print(f'  position {position:2d}: {token!r:5s} -> ID {token_id}')
    print('IDs:', ids)
    pause(args.step)

    restored = decode(ids, vocabulary)  # IDs -> original text.
    print('\n5. DECODE:', repr(restored))
    assert restored == args.text
    print('Exact round trip. These IDs will select embedding rows in the model.')


if __name__ == '__main__':
    main()
