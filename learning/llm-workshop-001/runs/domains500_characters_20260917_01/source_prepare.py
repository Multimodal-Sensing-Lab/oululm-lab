"""Make auditable training records from prose CSVs and an instruction/email CSV."""
import csv
import json
from collections import Counter
from pathlib import Path
from .data import split_rows, file_info
from llm_workshop.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
# Explicitly authored classroom examples, not facts extracted from the biology CSV.
DEFINITIONS = {
 'a cat': 'a cat is a small mammal. it has fur and is often kept as a pet.',
 'a dog': 'a dog is a mammal. people often keep dogs as pets.',
 'a cell': 'a cell is a basic unit of living things.',
 'a plant': 'a plant is a living organism. most plants use light to make food.',
 'a bird': 'a bird is an animal with feathers.',
 'a fish': 'a fish is an animal that lives in water and usually breathes through gills.',
 'a bond': 'a bond is a loan made by an investor to a borrower.',
 'interest': 'interest is the cost of borrowing money or the return paid on savings.',
 'an email': 'an email is a message sent electronically. it often has a subject and a body.',
 'a token': 'a token is one piece of text represented by an integer id.'}


def formatted(row):
    if row['task'] == 'prose':return '', row['response']
    if row['task'] == 'email':
        prompt='### instruction:\n'+row['instruction']
        if row.get('incoming_email'):prompt+='\n\n### incoming email:\n'+row['incoming_email']
        return prompt+'\n\n### email:\n',row['response']
    return 'question: '+row['instruction']+'\nanswer: ',row['response']


def prepare(data_dir=ROOT/'data/external'):
    sources, data = {}, {}
    for domain,filename in [('biology','animal_biology_sentences.csv'),('finance','finance_sentences.csv'),('email','email_finetune.csv')]:
        path=data_dir/filename
        with path.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
        fields={'id','category','sentence'} if domain!='email' else {'id','domain','email_type','instruction','incoming_email','email','text'}
        if not raw or not fields <= raw[0].keys():raise ValueError('Unexpected schema: '+filename)
        rows=[];seen=set();removed=0
        for r in raw:
            row={'id':r['id'],'category':r.get('category',r.get('email_type')), 'task':'email' if domain=='email' else 'prose',
                 'instruction':r.get('instruction','').strip(),'incoming_email':r.get('incoming_email','').strip(),
                 'response':r['email' if domain=='email' else 'sentence'].strip()}
            if not row['response'] or (domain=='email' and not row['instruction']):raise ValueError('Empty model field: '+str(r['id']))
            prompt,response=formatted(row)
            row['sentence']=prompt+response
            key=' '.join(row['sentence'].lower().split())
            if key in seen:removed+=1;continue
            seen.add(key);rows.append(row)
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Repeated record IDs: '+filename)
        data[domain]=split_rows(rows)
        sources[domain]={**file_info(path),'raw_records':len(raw),'retained_records':len(rows),'duplicates_removed':removed,
                         'characters':sum(len(r['sentence']) for r in rows),'whitespace_words':sum(len(r['sentence'].split()) for r in rows),
                         'categories':dict(Counter(r['category'] for r in rows)),
                         'fields_used':['instruction','incoming_email','email'] if domain=='email' else ['sentence'],
                         'raw_example':raw[0]}
    data['base']={}
    for split in ['train','validation']:
        values=read_jsonl(ROOT/f'examples/data/words/{split}.jsonl')+read_jsonl(ROOT/f'examples/data/tiny_{split}.jsonl')
        data['base'][split]=[{'id':str(i),'category':'general classroom prose','task':'prose','instruction':'','response':r['text'],'sentence':r['text']} for i,r in enumerate(values)]
    sources['base']={'paths':['examples/data/words/train.jsonl','examples/data/tiny_train.jsonl'],
                     'description':'Original controlled classroom sentences plus the three stories; general TinyGPT pretraining, not a downloaded pretrained LLM.',
                     'files':[file_info(ROOT/p) for p in ['examples/data/words/train.jsonl','examples/data/words/validation.jsonl','examples/data/tiny_train.jsonl','examples/data/tiny_validation.jsonl']]}
    data['qa']={'train':[],'validation':[]}
    for concept,answer in DEFINITIONS.items():
        for i,question in enumerate([f'what is {concept}?',f'define {concept}.',f'tell me about {concept}.',f'please explain {concept}.']):
            row={'id':f'{concept}-{i}','category':concept,'task':'qa','instruction':question,'response':answer}
            row['sentence']=''.join(formatted(row))
            data['qa']['validation' if i==3 else 'train'].append(row)
    sources['qa']={'path':'llm_workshop/domains/prepare.py','description':'30 authored training Q&A examples / 10 held-out phrasings, 10 shared concepts. Cat question is explicitly in training; this illustrates memorization/format learning, not unseen-fact acquisition.'}
    for domain,parts in data.items():
        a={' '.join(r['sentence'].lower().split()) for r in parts['train']}
        b={' '.join(r['sentence'].lower().split()) for r in parts['validation']}
        if a & b:raise ValueError('Exact split leakage: '+domain)
    return data,sources
