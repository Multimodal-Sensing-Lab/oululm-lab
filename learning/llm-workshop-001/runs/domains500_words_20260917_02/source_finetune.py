"""A common pretrained base → three independent 500-update domain models, plus a Q&A exercise.

python -m llm_workshop.domains.finetune --tokenization word --device cuda --output-dir runs/domains500_words_01
"""
import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path
import torch
from llm_workshop.tiny import TinyGPT, TinyGPTConfig
from llm_workshop.io import fresh_directory, write_json, write_jsonl, sha256
from llm_workshop.words.model import device_for
from llm_workshop.progress import LossPlot
from .prepare import prepare, formatted, ROOT
from .data import Tokenizer
from .train import save, load


def encode_rows(rows, tok, context):
    encoded=[]
    for r in rows:
        prompt,response=formatted(r)
        prefix=tok.encode(prompt,False)
        target=tok.encode(response,False)+[1]
        ids=[0]+prefix+target
        if len(ids)-1>context:raise ValueError(f'Record {r["id"]} exceeds context {context}; no silent truncation.')
        labels=ids[1:]
        if r['task']!='prose':labels=[-100]*len(prefix)+labels[len(prefix):]
        encoded.append((ids[:-1],labels))
    return encoded


def batch(encoded, indices, device):
    values=[encoded[i] for i in indices];width=max(len(x) for x,y in values)
    x=torch.full((len(values),width),1,dtype=torch.long,device=device);y=torch.full_like(x,-100)
    for i,(xx,yy) in enumerate(values):
        x[i,:len(xx)]=torch.tensor(xx,device=device);y[i,:len(yy)]=torch.tensor(yy,device=device)
    return x,y


@torch.inference_mode()
def measure(model, encoded, device, bs=16):
    model.eval();total=count=0
    for start in range(0,len(encoded),bs):
        x,y=batch(encoded,range(start,min(start+bs,len(encoded))),device);n=int((y!=-100).sum())
        total+=float(model(x,y)[1])*n;count+=n
    return total/count


def prompt_for(task, text, incoming=''):
    if task=='email':return formatted({'task':'email','instruction':text,'incoming_email':incoming,'response':''})[0]
    if task=='qa':return 'question: '+text.strip()+'\nanswer: '
    return text


@torch.inference_mode()
def predict(model,tok,prompt,tokens=64):
    pieces=tok.pieces(prompt);unknown=sorted(set(pieces)-set(tok.vocabulary))
    if unknown:raise ValueError('Outside this checkpoint vocabulary: '+', '.join(unknown[:12])+'. Vocabulary is fixed; try one of its input examples.')
    ids=[0]+tok.encode(prompt,False);start=len(ids);device=next(model.parameters()).device
    if len(ids)>=model.cfg.block_size:raise ValueError('Input fills the context; shorten it to leave room for the response.')
    stop='token budget';model.eval()
    for _ in range(tokens):
        if len(ids)>=model.cfg.block_size:stop='context full';break
        chosen=int(model(torch.tensor([ids],device=device))[0][0,-1].argmax());ids.append(chosen)
        if chosen==1:stop='end token';break
    return {'text':tok.decode(ids[start:]),'ids':ids[start:],'stop':stop,'input_tokens':start,'decoding':'greedy'}


def train(args):
    if args.steps<1 or args.base_steps<1:raise ValueError('Updates must be positive.')
    torch.set_num_threads(2);torch.manual_seed(1337);device=device_for(args.device)
    if device=='cuda':torch.cuda.manual_seed_all(1337);torch.backends.cuda.matmul.allow_tf32=False
    data,sources=prepare(args.data_dir)
    tok=Tokenizer.fit(args.tokenization,(r['sentence'] for parts in data.values() for r in parts['train']))
    tok.fit_description='Base + biology + email + finance + authored Q&A TRAIN text; fixed before pretraining. No validation vocabulary fitting.'
    context=256 if args.tokenization=='word' else 1280
    cfg=TinyGPTConfig(len(tok.vocabulary),block_size=context,n_layer=2,n_head=2,n_embd=64,dropout=0.)
    encoded={d:{s:encode_rows(rows,tok,context) for s,rows in parts.items()} for d,parts in data.items()}
    out=fresh_directory(args.output_dir)
    for p in [Path(__file__),Path(__file__).with_name('prepare.py'),Path(__file__).with_name('data.py'),ROOT/'llm_workshop/tiny.py',ROOT/'llm_workshop/progress.py']:
        shutil.copy2(p,out/('source_'+p.name))
    specs={}
    for d,parts in data.items():
        specs[d]={}
        for split,rows in parts.items():
            write_jsonl(out/f'{d}_{split}.jsonl',rows)
            specs[d][split]={'records':len(rows),'characters':sum(len(r['sentence']) for r in rows),
                            'whitespace_words':sum(len(r['sentence'].split()) for r in rows),
                            'input_tokens':sum(len(x) for x,y in encoded[d][split]),
                            'target_tokens':sum(sum(t!=-100 for t in y) for x,y in encoded[d][split]),
                            'unknown_tokens':sum(x.count(2) for x,y in encoded[d][split])}
    manifest={'sources':sources,'splits':specs,'tokenizer':tok.metadata(),'config':cfg.__dict__,
              'pipeline':['Read CSV schema; choose model fields','Remove normalized exact duplicate records','Split by category: 5% validation before vocabulary fitting','Fit one shared vocabulary on TRAIN records only','Lowercase; add BOS/EOS; keep records separate','Shift inputs/targets; mask prompt targets for email and Q&A; ignore padding','Train on train split; evaluate complete validation sets'],
              'split_caution':'Exact text holdout; related templates can cross splits. Q&A holds out a phrasing, not the concepts. No separate final test set.',
              'vocabulary_scope':'Fit once on base + biology + email + finance + authored Q&A TRAIN text. Reserve all domain IDs before pretraining; only base text supplies base gradients.'}
    manifest['tokenizer']['fit']=manifest['vocabulary_scope']
    write_json(out/'data_manifest.json',manifest)
    model=TinyGPT(cfg).to(device);save(out/'untrained.pt',model,tok,'untrained',0)
    stages=[];bs=32 if args.tokenization=='word' else 8
    prompts={'base':'the small cat','biology':'the blue mussel','finance':'the future value of',
             'email':data['email']['validation'][0]['instruction'],'qa':'what is a cat?'}
    for name in ['base','biology','email','finance','qa']:
        if name!='base':model=copy.deepcopy(base)
        steps=args.base_steps if name=='base' else args.steps
        lr=.003 if name=='base' else .0006
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr);rng=random.Random(1337);history=[];seen=0
        plot=LossPlot(out/f'{name}_progress.png',f'{args.tokenization} · {name}',args.live_plot)
        started=time.perf_counter()
        for step in range(steps+1):
            if step%100==0 or step==steps:
                p={'step':step,'validation':measure(model,encoded[name]['validation'],device,bs),
                   'base_validation':measure(model,encoded['base']['validation'],device,bs)}
                if step:p['train_batch']=last_loss
                history.append(p);plot.update(history,('train_batch','validation','base_validation'))
                write_json(out/f'{name}_progress.json',history);print(name,json.dumps(p),flush=True)
            if step==steps:break
            indices=[rng.randrange(len(encoded[name]['train'])) for _ in range(bs)]
            x,y=batch(encoded[name]['train'],indices,device);seen+=int((y!=-100).sum())
            model.train();optimizer.zero_grad(set_to_none=True)
            _,loss=model(x,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            last_loss=float(loss.detach())
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started;plot.close();model.eval()
        save(out/f'{name}.pt',model,tok,name,steps)
        if name=='base':base=copy.deepcopy(model);base_hash=sha256(out/'base.pt')
        task='email' if name=='email' else 'qa' if name=='qa' else 'prose'
        stage={'name':name,'steps':steps,'learning_rate':lr,'batch_size':bs,'target_exposures':seen,'history':history,
               'seconds':elapsed,'checkpoint':f'{name}.pt','sha256':sha256(out/f'{name}.pt'),
               'base_checkpoint':None if name=='base' else 'base.pt','base_sha256':None if name=='base' else base_hash,
               'method':'Full-weight pretraining' if name=='base' else 'Full-weight fine-tuning (not LoRA)',
               'task':task,'example_input':prompts[name],
               'sample':predict(model,tok,prompt_for(task,prompts[name]),96 if args.tokenization=='word' else 160)}
        stages.append(stage);write_json(out/f'{name}_report.json',stage)
    report={'manifest':manifest,'stages':stages,'parameter_count':sum(p.numel() for p in base.parameters()),
            'device':device,'device_name':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU','seed':1337,
            'timing':'Loop + evaluation + plot writes, excluding startup and final save/generation.',
            'selection':'Fixed update budgets; final weights retained. No early stopping. Inference checkpoints only, no optimizer resume.'}
    write_json(out/'report.json',report);print('Saved',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenization',choices=['word','character'],default='word');p.add_argument('--steps',type=int,default=500)
    p.add_argument('--base-steps',type=int,default=1000);p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/external');p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--live-plot',action='store_true');train(p.parse_args())
