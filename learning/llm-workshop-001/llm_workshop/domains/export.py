"""Bundle measured runs and exact input records for the offline browser data lab."""
import argparse
import json
from pathlib import Path
from llm_workshop.io import read_jsonl, write_json
from .data import file_info, stats

ROOT = Path(__file__).resolve().parents[2]
BROWSER = ROOT/'llm_workshop/browser'


def export(run_dirs):
    runs, datasets = [], []
    for folder in run_dirs:
        report = json.loads((folder/'report.json').read_text())
        report['path'] = str(folder.relative_to(ROOT))
        runs.append(report)
        if not datasets:
            for domain, source in report['manifest']['sources'].items():
                rows = []
                for split in ['train', 'validation']:
                    rows += [{**r, 'split': split} for r in read_jsonl(folder/f'{domain}_{split}.jsonl')]
                datasets.append({'id': domain, 'label': f'{domain.title()} CSV', 'rows': rows,
                                 'stats': source, 'counting': report['manifest']['counting'],
                                 'note': 'Only the sentence field is model input; id/category are metadata. Both tokenizers use these same split records.'})
        else:
            for domain in ['biology', 'finance']:
                for split in ['train', 'validation']:
                    assert (folder/f'{domain}_{split}.jsonl').read_bytes() == (run_dirs[0]/f'{domain}_{split}.jsonl').read_bytes()
    for key, label, prefix in [('stories', 'Original character stories', ROOT/'examples/data'),
                              ('words', 'Original controlled word corpus', ROOT/'examples/data/words')]:
        rows = []
        for split in (['train','validation'] if key=='stories' else ['train','validation','test']):
            path = prefix/(f'tiny_{split}.jsonl' if key=='stories' else f'{split}.jsonl')
            rows += [{'id': r['id'], 'category': r.get('family', 'story'), 'sentence': r['text'], 'split': split} for r in read_jsonl(path)]
        datasets.append({'id': key, 'label': label, 'rows': rows, 'stats': stats(rows),
                         'counting': 'Counts sum text fields only, excluding JSON metadata and document separators.',
                         'note': 'Original story training: 3 documents, 12 sentences, 109 whitespace words, 521 joined characters (517 text-field characters + 4 separator newlines). Original word training: 432 sentences, 3,456 words, 17,396 text-field characters.'})
    # Include exact filtered character data and all tiny LoRA paragraphs, not just the new CSVs.
    experiments = json.loads((ROOT/'runs/browser_character_20260917_01/browser_experiments.json').read_text())
    seen = set()
    for run in experiments['runs']:
        key = run.get('domain') if run.get('kind') == 'adapter' else ('controlled' if 'controlled' in run['id'] else None)
        if not key or key in seen:
            continue
        seen.add(key)
        rows = []
        for split, source in run['corpus'].items():
            path = ROOT/source['path']
            if path.suffix == '.jsonl':
                records = read_jsonl(path)
                rows += [{'id': r.get('id',str(i)), 'category': r.get('family',key), 'sentence': r['text'], 'split': split} for i,r in enumerate(records)]
            else:
                rows.append({'id': split, 'category': key, 'sentence': path.read_text(), 'split': split})
        datasets.append({'id': 'character_'+key, 'label': 'Character '+('controlled corpus' if key=='controlled' else key+' LoRA paragraphs'),
                         'rows': rows, 'stats': stats(rows), 'counting': 'Counts sum text fields. The character stream adds two newlines between documents.',
                         'note': 'Exact input used by the original 32-entry character route; separate from the new CSV models.'})
    data = {'runs': runs, 'datasets': datasets}
    (BROWSER/'dataset-data.js').write_text('globalThis.WORKSHOP_DATA_LAB = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n', encoding='utf-8')
    print('Exported',len(runs),'runs and',len(datasets),'datasets to',BROWSER/'dataset-data.js')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dirs', nargs='+', type=Path, required=True)
    a=p.parse_args()
    export([x.resolve() for x in a.run_dirs])
