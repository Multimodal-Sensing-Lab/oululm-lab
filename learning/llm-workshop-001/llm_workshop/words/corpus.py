"""Author a controlled 576-sentence corpus, then freeze disjoint sentence splits.

Run: python -m llm_workshop.words.corpus --output-dir examples/data/words
Shared templates make this a pattern/recombination lab, not a language benchmark.
"""
import argparse
import random
from pathlib import Path
from llm_workshop.io import assert_disjoint, fresh_directory, write_jsonl, write_json, sha256
from .tokenizer import WordTokenizer, pieces

DATA = Path(__file__).resolve().parents[2] / 'examples/data/words'


def sentences():
    rows = []
    def add(family, text):
        rows.append({'id': f'{family}-{len(rows):03d}', 'family': family, 'text': text})
    animals = {'cat': ['sleeps', 'rests'], 'dog': ['runs', 'plays'],
               'rabbit': ['jumps', 'rests'], 'bird': ['sings', 'rests'],
               'horse': ['runs', 'rests'], 'fox': ['runs', 'sleeps']}
    for animal, verbs in animals.items():
        for adjective in ['small', 'young', 'quiet', 'happy']:
            for verb in verbs:
                for place in ['garden', 'field', 'yard', 'park']:
                    add('animals', f'the {adjective} {animal} {verb} in the {place}.')
    for actor in ['robot', 'teacher', 'student', 'helper']:
        for adjective in ['quiet', 'happy', 'kind', 'young']:
            for predicate in ['carries the box', 'reads the book', 'moves the chair']:
                for place in ['classroom', 'library', 'room', 'hall']:
                    add('classroom', f'the {adjective} {actor} {predicate} in the {place}.')
    for subject in ['plant', 'tree', 'flower', 'leaf']:
        for adjective in ['small', 'green', 'young', 'new']:
            for predicate in ['needs water', 'needs light', 'grows slowly']:
                for place in ['garden', 'field', 'park', 'yard']:
                    add('garden', f'the {adjective} {subject} {predicate} in the {place}.')
    return rows


def prepare(output, seed=42):
    rows = sentences()
    assert len({r['text'] for r in rows}) == len(rows)
    splits = {name: [] for name in ['train', 'validation', 'test']}
    rng = random.Random(seed)
    for family in ['animals', 'classroom', 'garden']:
        group = [row for row in rows if row['family'] == family]
        rng.shuffle(group)
        for name, subset in zip(splits, [group[:144], group[144:168], group[168:]]):
            splits[name].extend(subset)
    assert_disjoint(splits)
    tokenizer = WordTokenizer.fit(r['text'] for r in splits['train'])
    for row in rows:
        tokenizer.encode(row['text'])  # Fail rather than silently expand vocabulary.
    output = fresh_directory(output)
    manifest = {'source': 'Original controlled sentence templates; fictional teaching examples',
                'seed': seed, 'split_method': '432/72/72 disjoint sentences, stratified by family; shared templates',
                'scope': 'Recombination of familiar words/templates; no claim of broad language or biology expertise',
                'tokenizer': tokenizer.metadata(), 'files': {}}
    for name, subset in splits.items():
        path = output / f'{name}.jsonl'
        write_jsonl(path, subset)
        texts = [r['text'] for r in subset]
        manifest['files'][name] = {'records': len(subset), 'sentences': len(subset),
            'words': sum(sum(p.isalpha() for p in pieces(t)) for t in texts),
            'characters': sum(len(t) for t in texts),
            'tokens_with_boundaries': sum(len(tokenizer.encode(t, boundaries=True)) for t in texts),
            'bytes': path.stat().st_size, 'sha256': sha256(path)}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DATA)
    args = parser.parse_args()
    print(prepare(args.output_dir))
