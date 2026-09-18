"""Record real CPU/CUDA character experiments and LoRA comparisons for the browser.

Keeps the original 200-step export intact. Browser inference still runs locally in JS.
Run from the repository root; use a new output directory for every recording.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from llm_workshop.teaching import Session, ROOT, topic, tokenizer
from llm_workshop.io import fresh_directory, read_jsonl, write_json, write_jsonl, assert_disjoint

HERE = Path(__file__).resolve().parent
DOMAIN_VALIDATION = {
    'biology': 'the plant grows in the garden. the cells need water and light. a leaf is green.',
    'email': 'dear friend. thank you for your report. please send the team a note today.',
    'finance': 'the firm reports a profit. the sales report shows cash and costs. costs are low.',
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def describe(path, rows):
    text = '\n\n'.join(row['text'] for row in rows)
    return text, {'path': str(path.relative_to(ROOT)), 'documents': len(rows),
                  'sentences': len(re.findall(r'[.!?]+', text)),
                  'words': len(text.split()), 'characters': len(text),
                  'bytes': len(text.encode()), 'sha256': digest(text),
                  'preview': rows[0]['text']}


def weights(model):
    return {key: value.detach().cpu().tolist() for key, value in model.named_parameters()}


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def measure(model, data, block):
    model.eval()
    with torch.inference_mode():
        result = {}
        for name, stream in data.items():
            starts = torch.linspace(0, len(stream)-block-1, steps=4).long().tolist()
            x = torch.stack([stream[i:i+block] for i in starts])
            y = torch.stack([stream[i+1:i+block+1] for i in starts])
            result[name] = model(x, y)[1].item()
        return result


def checkpoint(model, step, folder, session):
    torch.save({'model_state_dict': model.state_dict(), 'config': model.cfg.__dict__,
                'stoi': session.stoi, 'itos': dict(enumerate(session.vocabulary)),
                'step': step}, folder / f'step_{step}.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--browser-output', type=Path, default=HERE/'experiment-data.js')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--adapter-steps', type=int, default=200)
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    args = parser.parse_args()
    if not 200 <= args.steps <= 10000 or not 1 <= args.adapter_steps <= 1000:
        parser.error('Use 200–10000 full-training steps and 1–1000 adapter steps.')
    if 'cuda' in args.devices and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable. Use --devices cpu or run with GPU access.')
    folder = fresh_directory(args.output_dir.resolve())
    session = Session()
    original = json.loads((HERE/'model-data.js').read_text().split(' = ', 1)[1].rstrip(';\n'))
    assert session.vocabulary == original['vocabulary']
    session.model.load_state_dict({k: torch.tensor(v) for k, v in original['models']['untrained']['weights'].items()}, strict=False)
    # Keep the original fixed alphabet, allowing every checkpoint in every chapter.
    corpora = {}
    for name in ['stories', 'controlled']:
        texts, stats = {}, {}
        splits = {}
        for split in ['train', 'validation']:
            source = ROOT / ('examples/data/tiny_'+split+'.jsonl' if name=='stories' else 'examples/data/words/'+split+'.jsonl')
            rows = read_jsonl(source)
            omitted = 0
            path = source
            if name == 'controlled':
                filtered = [r for r in rows if set(r['text']) <= set(session.stoi)]
                omitted = len(rows)-len(filtered)
                rows = filtered
                path = folder/'corpora'/name/(split+'.jsonl')
                write_jsonl(path, rows)
            splits[split] = rows
            texts[split], stats[split] = describe(path, rows)
            stats[split].update({'source_path': str(source.relative_to(ROOT)), 'omitted_documents': omitted})
            tokenizer.encode(texts[split], session.stoi)  # Reject unknown characters; never silently remap.
        assert_disjoint(splits)
        corpora[name] = {'texts': texts, 'stats': stats}
    for split in ['train', 'validation']:
        assert digest(corpora['stories']['texts'][split]) == original['training']['data_hashes'][split], 'Original story data changed; inspect provenance before exporting.'
    label = {'stories': 'Three stories', 'controlled': 'Controlled sentences'}
    baseline = {'id': 'legacy', 'label': 'Original stories · CPU · 200 updates',
                'before': 'untrained', 'after': 'trained', 'device': 'cpu', 'device_name': 'CPU (original recording)',
                'steps': 200, 'history': original['training']['history'], 'samples': original['training']['samples'],
                'corpus': corpora['stories']['stats'], 'kind': 'full', 'learning_rate': .003,
                'batch_size': 4, 'context': 32, 'seed': 1337, 'training_seconds': None,
                'trainable_parameters': original['parameter_count'], 'initialization': 'untrained'}
    result = {'models': {}, 'runs': [baseline], 'config': original['config'],
              'vocabulary': original['vocabulary'], 'torch_version': torch.__version__,
              'source_hash': digest(Path(__file__).read_text()), 'reference_logits': []}
    trained_bases = {}

    def record(model, model_id, step, run, out):
        cpu = copy.deepcopy(model).cpu().eval()
        checkpoint(cpu, step, out, session)
        entry = {'step': step, 'label': run['label'], 'run': run['id'], 'weights': weights(cpu)}
        result['models'][model_id] = entry
        with torch.inference_mode():
            ids = session.ids('the cat sat.')
            result['reference_logits'].append({'model': model_id, 'text': 'the cat sat.', 'logits': cpu(ids)[0][0].tolist()})
            sample = session.decode(cpu.generate(session.ids('the '), 40, 0)[0].tolist())
        run['samples'] = {'prompt': 'the ', 'decoding': 'greedy', 'after': sample}
        result['runs'].append(run)
        write_json(out/(run['id']+'.json'), run)

    for corpus_name, corpus in corpora.items():
        for device in args.devices:
            run_id = f'{corpus_name}_{device}'
            out = fresh_directory(folder/run_id)
            model = copy.deepcopy(session.model).to(device)
            data = {k: torch.tensor(tokenizer.encode(t, session.stoi), device=device) for k, t in corpus['texts'].items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
            rng = torch.Generator().manual_seed(1337)
            history = [{'step': 0, **measure(model, data, 32)}]
            synchronize(device);started = time.perf_counter()
            for step in range(1, args.steps+1):
                starts = torch.randint(len(data['train'])-32, (4,), generator=rng).tolist()
                x = torch.stack([data['train'][i:i+32] for i in starts])
                y = torch.stack([data['train'][i+1:i+33] for i in starts])
                torch.manual_seed(1337+step-1)
                model.train();optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True);optimizer.step()
                if step % 50 == 0 or step == args.steps:
                    history.append({'step': step, **measure(model, data, 32)})
                if step in {200, args.steps}:
                    synchronize(device);elapsed = time.perf_counter()-started
                    key = f'{run_id}_{step}'
                    run = {'id': key, 'label': f'{label[corpus_name]} · {device.upper()} · {step} updates',
                           'before': 'untrained', 'after': key, 'device': device,
                           'device_name': torch.cuda.get_device_name(0) if device=='cuda' else platform.processor() or 'CPU',
                           'steps': step, 'history': copy.deepcopy(history), 'corpus': corpus['stats'], 'kind': 'full',
                           'learning_rate': .003, 'batch_size': 4, 'context': 32, 'seed': 1337,
                           'training_seconds': elapsed, 'trainable_parameters': original['parameter_count'],
                           'initialization': 'untrained', 'checkpoint_path': str((out/f'step_{step}.pt').relative_to(ROOT))}
                    # Timing includes optimizer steps and evaluation, excludes snapshot export.
                    record(model, key, step, run, out)
                    synchronize(device);started = time.perf_counter()-elapsed
                    print(key, history[-1], f'{elapsed:.2f}s', flush=True)
            trained_bases[(corpus_name, device)] = copy.deepcopy(model).cpu().eval()

    device = 'cuda' if 'cuda' in args.devices else 'cpu'
    base_id = f'controlled_{device}_{args.steps}'
    base = trained_bases[('controlled', device)]
    for domain, prose in topic(6).DOMAIN_TEXTS.items():
        out = fresh_directory(folder/('adapter_'+domain))
        texts = {'train': prose, 'validation': DOMAIN_VALIDATION[domain]}
        stats = {}
        for split, text in texts.items():
            path = out/(split+'.jsonl');rows = [{'id': domain+'-'+split, 'text': text}];write_jsonl(path, rows)
            _, stats[split] = describe(path, rows)
        model = copy.deepcopy(base).to(device).eval().requires_grad_(False)
        torch.manual_seed(42)
        model.lm_head = topic(6).LoRALinear(model.lm_head, rank=4).to(device)
        frozen = {k: v.detach().clone() for k, v in model.named_parameters() if not v.requires_grad}
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.02)
        data = {k: torch.tensor(tokenizer.encode(t, session.stoi), device=device) for k, t in texts.items()}
        history = [{'step': 0, **measure(model, data, 32)}];rng = torch.Generator().manual_seed(42)
        synchronize(device);started = time.perf_counter()
        for step in range(1, args.adapter_steps+1):
            starts = torch.randint(len(data['train'])-32, (4,), generator=rng).tolist()
            x = torch.stack([data['train'][i:i+32] for i in starts]);y = torch.stack([data['train'][i+1:i+33] for i in starts])
            optimizer.zero_grad(set_to_none=True);loss=model(x,y)[1];loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            if step % 10 == 0 or step == args.adapter_steps:history.append({'step': step, **measure(model,data,32)})
        synchronize(device);elapsed = time.perf_counter()-started
        for name, p in model.named_parameters():
            if name in frozen:torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
        layer = model.lm_head
        adapter = {'a': layer.a.detach().cpu(), 'b': layer.b.detach().cpu(), 'scale': layer.scale}
        torch.save(adapter,out/'adapter_matrices.pt')
        merged = copy.deepcopy(model);merged.lm_head = copy.deepcopy(layer.base)
        with torch.no_grad():
            merged.lm_head.weight.add_(layer.b@layer.a,alpha=layer.scale)
            probe=data['validation'][:32][None]
            torch.testing.assert_close(model(probe)[0],merged(probe)[0],atol=2e-5,rtol=2e-5)
        key='adapter_'+domain
        run={'id':key,'label':f'{domain.title()} LoRA · {device.upper()} · {args.adapter_steps} updates',
             'before':base_id,'after':key,'base_label':result['models'][base_id]['label'],
             'device':device,'device_name':torch.cuda.get_device_name(0) if device=='cuda' else platform.processor() or 'CPU',
             'steps':args.adapter_steps,'history':history,'corpus':stats,'kind':'adapter','domain':domain,
             'learning_rate':.02,'batch_size':4,'context':32,'seed':42,'training_seconds':elapsed,
             'rank':4,'trainable_parameters':sum(p.numel() for p in parameters),'initialization':base_id,
             'scope':'LoRA on lm_head only; base frozen; no attention projection changes',
             'a':adapter['a'].tolist(),'b':adapter['b'].tolist(),'scale':adapter['scale'],
             'checkpoint_path':str((out/f'step_{args.adapter_steps}.pt').relative_to(ROOT))}
        record(merged,key,args.adapter_steps,run,out)
        print(key,history[-1],flush=True)
    write_json(folder/'browser_experiments.json',result)
    args.browser_output.write_text('// Actual recorded character experiments; original model-data.js is unchanged.\n'
                                  +'globalThis.WORKSHOP_EXPERIMENTS = '+json.dumps(result,separators=(',',':'),allow_nan=False)+';\n')
    print('Exported',len(result['runs']),'runs to',args.browser_output,flush=True)


if __name__ == '__main__':
    main()
