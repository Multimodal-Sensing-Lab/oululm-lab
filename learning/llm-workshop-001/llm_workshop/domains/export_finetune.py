"""Export settings, exact prepared records and measured previews (not weights) for the domain lab."""
import argparse,json
from pathlib import Path
from llm_workshop.io import read_jsonl,write_json
from .prepare import ROOT


def export(folders):
    runs=[]
    for folder in folders:
        report=json.loads((folder/'report.json').read_text())
        report['path']=str(folder.relative_to(ROOT));report['id']=report['manifest']['tokenizer']['kind']
        report['records']={name:{split:read_jsonl(folder/f'{name}_{split}.jsonl') for split in ['train','validation']} for name in report['manifest']['splits']}
        runs.append(report)
    payload={'runs':runs,'mode':'Saved Python recordings; live generation requires the local workshop server.'}
    out=ROOT/'llm_workshop/browser'
    (out/'domain-data.js').write_text('globalThis.WORKSHOP_DOMAIN_RUNS = '+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n')
    write_json(out/'domain-catalog.json',{'runs':[{'id':r['id'],'path':r['path']} for r in runs]})
    print('Exported',[(r['id'],len(r['stages'])) for r in runs])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dirs',type=Path,nargs='+',required=True);a=p.parse_args();export([x.resolve() for x in a.run_dirs])
