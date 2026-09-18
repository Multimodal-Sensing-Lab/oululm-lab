"""An intentionally small word tokenizer: lowercase words, punctuation, fixed IDs.

This is a teaching convention, not a production subword tokenizer. Whitespace
and case are normalized; decode reconstructs readable text, not original bytes.
"""
import re

SPECIAL = ['<BOS>', '<EOS>', '<UNK>']


def pieces(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Enter a few words, for example: the small cat')
    if re.search(r'[^a-zA-Z\s.!?]', text):
        raise ValueError('Use English letters, spaces and . ! ? in this teaching tokenizer.')
    return re.findall(r'[a-z]+|[.!?]', text.lower())


class WordTokenizer:
    def __init__(self, vocabulary):
        if vocabulary[:3] != SPECIAL or len(vocabulary) != len(set(vocabulary)):
            raise ValueError('Invalid word vocabulary or special-token order.')
        self.vocabulary = list(vocabulary)
        self.stoi = {token: i for i, token in enumerate(vocabulary)}

    @classmethod
    def fit(cls, texts):
        # Fit on training text only. Validation/test do not define the vocabulary.
        return cls(SPECIAL + sorted({p for text in texts for p in pieces(text)}))

    def encode(self, text, *, boundaries=False):
        tokens = pieces(text)
        missing = sorted(set(tokens) - self.stoi.keys())
        if missing:
            raise ValueError(f'Outside the fixed word vocabulary: {missing}. Try a corpus example.')
        ids = [self.stoi[token] for token in tokens]
        return [0] + ids + [1] if boundaries else ids

    def prompt(self, text, block_size):
        ids = [0] + self.encode(text)
        if len(ids) > block_size:
            raise ValueError(f'Use at most {block_size - 1} word/punctuation tokens; BOS takes one position.')
        return ids

    def decode(self, ids):
        # Only the initial BOS and final EOS are presentation markers. Never
        # re-tokenize generated '<UNK>' or other special-token display strings.
        tokens = [self.vocabulary[i] for i in ids]
        if tokens and tokens[0] == '<BOS>':
            tokens = tokens[1:]
        if tokens and tokens[-1] == '<EOS>':
            tokens = tokens[:-1]
        return re.sub(r'\s+([.!?])', r'\1', ' '.join(tokens))

    def metadata(self):
        return {'kind': 'word', 'version': 1, 'vocabulary': self.vocabulary,
                'normalization': 'lowercase; words and .!?; canonical spaces'}
