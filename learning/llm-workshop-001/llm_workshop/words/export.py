"""Export saved word checkpoints for real browser inference; does not train.

python -m llm_workshop.words.export --run-dir runs/words_classroom_01
"""
import argparse
import json
from pathlib import Path
import torch
from llm_workshop.io import sha256, write_json
from llm_workshop.browser.export_model import weights, reference
from .model import load_checkpoint, WordSession, generate

BROWSER = Path(__file__).resolve().parents[1] / 'browser'


def export(run_dir, output_dir=BROWSER):
    torch.set_num_threads(2)
    report = json.loads((run_dir / 'report.json').read_text())
    models, fixtures = {}, []
    vocabulary = None
    cfg = None
    for label, filename in [('untrained', 'before.pt'), ('trained', 'after.pt')]:
        model, tokenizer, state = load_checkpoint(run_dir / filename)
        if vocabulary is not None and (vocabulary != tokenizer.vocabulary or cfg != model.cfg.__dict__):
            raise ValueError('Before/after checkpoints must have identical tokenizer and architecture.')
        if state.get('data_hashes') != report['data_hashes']:
            raise ValueError('Checkpoint data hashes disagree with report.')
        vocabulary, cfg = tokenizer.vocabulary, model.cfg.__dict__
        models[label] = {'step': state['step'], 'weights': weights(model)}
        for prompt in ['the small cat', 'the green plant', 'the cat cat']:
            fixtures.append(reference(WordSession(model, tokenizer), label, prompt))
        for sample in report['samples']:
            # Browser fixture text is remeasured on CPU, matching the export.
            sample['before' if label == 'untrained' else 'after'] = generate(model, tokenizer, sample['prompt'])
    source_files = [Path(__file__).with_name(name) for name in ['tokenizer.py', 'corpus.py', 'model.py', 'train.py', 'export.py']]
    source_files.append(Path(__file__).resolve().parents[1] / 'tiny.py')
    payload = {'format': 'llm-workshop-word-browser-v1', 'config': cfg, 'vocabulary': vocabulary,
        'tokenizer': tokenizer.metadata(), 'models': models, 'training': report,
        'checkpoint_hashes': {name: sha256(run_dir / name) for name in ['before.pt', 'after.pt']},
        'export_source_hashes': {p.name: sha256(p) for p in source_files}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'word-model-data.js').write_text(
        '// Generated from real saved word checkpoints by python -m llm_workshop.words.export\n'
        'globalThis.WORKSHOP_WORD_MODEL = ' + json.dumps(payload, separators=(',', ':'), allow_nan=False) + ';\n', encoding='utf-8')
    write_json(output_dir / 'word-reference.json', fixtures)
    print('Exported saved checkpoints and Python reference tensors to', output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=BROWSER)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir)
