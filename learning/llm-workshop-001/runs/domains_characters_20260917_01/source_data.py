"""Audit CSV input and freeze a shared training-only vocabulary before training."""
import csv
import hashlib
import math
import random
import re
from collections import Counter
from pathlib import Path

SPECIAL = ['<BOS>', '<EOS>', '<UNK>']
WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d|[^\w\s]", re.UNICODE)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'category', 'sentence']:
            raise ValueError('Expected CSV columns: id,category,sentence')
        rows = list(reader)
    if not rows or any(not r['sentence'].strip() or not r['category'] for r in rows):
        raise ValueError('CSV must contain nonempty sentences and categories.')
    if len({r['id'] for r in rows}) != len(rows) or len({r['sentence'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs or exact sentences: deduplicate before splitting.')
    return rows


def split_rows(rows, seed=1337):
    """Exact 5% holdout, distributed across categories by largest remainder."""
    categories = sorted({r['category'] for r in rows})
    groups = {c: [r for r in rows if r['category'] == c] for c in categories}
    counts = {c: math.floor(len(groups[c]) * .05) for c in categories}
    remaining = round(len(rows) * .05) - sum(counts.values())
    for c in sorted(categories, key=lambda c: (-(len(groups[c]) * .05 % 1), c))[:remaining]:
        counts[c] += 1
    rng = random.Random(seed)
    train, validation = [], []
    for c in categories:
        values = groups[c][:]
        rng.shuffle(values)
        validation.extend(values[:counts[c]])
        train.extend(values[counts[c]:])
    return {'train': train, 'validation': validation}


class Tokenizer:
    def __init__(self, kind, vocabulary):
        self.kind, self.vocabulary = kind, vocabulary
        self.stoi = {v: i for i, v in enumerate(vocabulary)}

    def pieces(self, text):
        text = text.lower()
        return list(text) if self.kind == 'character' else WORD.findall(text)

    @classmethod
    def fit(cls, kind, texts):
        obj = cls(kind, SPECIAL)
        return cls(kind, SPECIAL + sorted({p for text in texts for p in obj.pieces(text)}))

    def encode(self, text, boundaries=True):
        ids = [self.stoi.get(p, 2) for p in self.pieces(text)]
        return [0] + ids + [1] if boundaries else ids

    def decode(self, ids):
        pieces = [self.vocabulary[i] for i in ids if i not in (0, 1)]
        if self.kind == 'character':
            return ''.join(pieces)
        text = ' '.join(pieces)
        # Digits remain separate model tokens; join them for human reading.
        text = re.sub(r'(?<=\d) (?=\d)', '', text)
        text = re.sub(r'\s+([.,!?;:%)])', r'\1', text)
        text = re.sub(r'(?<=\d)([.,]) (?=\d)', r'\1', text)
        return re.sub(r'([(£$€])\s+', r'\1', text)

    def metadata(self):
        return {'kind': self.kind, 'version': 'csv-v1', 'vocabulary': self.vocabulary,
                'rule': 'Lowercase characters' if self.kind == 'character' else 'Lowercase whole words (internal apostrophes/hyphens allowed), individual digits, punctuation/symbols; not BPE',
                'fit': 'Union of biology TRAIN and finance TRAIN only; fixed before any weight updates'}


def stats(rows, tokenizer=None):
    result = {'records': len(rows), 'unique_texts': len({r['sentence'] for r in rows}),
              'characters': sum(len(r['sentence']) for r in rows),
              'whitespace_words': sum(len(r['sentence'].split()) for r in rows),
              'categories': dict(sorted(Counter(r['category'] for r in rows).items()))}
    if tokenizer:
        tokens = [tokenizer.encode(r['sentence']) for r in rows]
        result.update(tokens_including_boundaries=sum(map(len, tokens)),
                      target_tokens=sum(len(t)-1 for t in tokens),
                      unknown_tokens=sum(t.count(2) for t in tokens),
                      max_input_positions=max(len(t)-1 for t in tokens))
    return result


def file_info(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
