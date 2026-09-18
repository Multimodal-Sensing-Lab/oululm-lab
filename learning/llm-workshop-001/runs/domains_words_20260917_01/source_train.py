"""Record biology pretraining, finance fine-tuning, and a 20%-biology replay comparison.

python -m llm_workshop.domains.train --tokenization word --device cuda --output-dir runs/domains_words_01
All plots update on disk during training. Add --live-plot for a desktop window.
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
import torch
from llm_workshop.io import fresh_directory, write_json
from llm_workshop.tiny import TinyGPT, TinyGPTConfig
from llm_workshop.words.model import device_for
from llm_workshop.progress import LossPlot
from .data import Tokenizer, read_csv, split_rows, stats, file_info

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'llm-workshop-csv-v1'
PROMPTS = ['the blue mussel', 'the animal', 'the future value of', 'a company with']


def batch(rows, tokenizer, device, context):
    sequences = [tokenizer.encode(r['sentence']) for r in rows]
    width = max(len(ids)-1 for ids in sequences)
    if width > context:
        raise ValueError('Record exceeds context: no silent truncation.')
    x = torch.full((len(rows), width), 1, dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, ids in enumerate(sequences):
        x[i, :len(ids)-1] = torch.tensor(ids[:-1], device=device)
        y[i, :len(ids)-1] = torch.tensor(ids[1:], device=device)
    return x, y


@torch.inference_mode()
def measure(model, rows, tokenizer, device):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(rows), 64):
        x, y = batch(rows[start:start+64], tokenizer, device, model.cfg.block_size)
        n = int((y != -100).sum())
        total += float(model(x, y)[1]) * n
        count += n
    return total / count


@torch.inference_mode()
def generate(model, tokenizer, prompt, tokens=48):
    ids = [0] + tokenizer.encode(prompt, boundaries=False)
    if 2 in ids:
        raise ValueError('Prompt contains a token outside the fixed vocabulary.')
    if len(ids) > model.cfg.block_size:
        raise ValueError('Prompt exceeds the context window.')
    model.eval()
    device = next(model.parameters()).device
    for _ in range(tokens):
        if len(ids) >= model.cfg.block_size:
            break
        logits = model(torch.tensor([ids], device=device))[0][0, -1]
        # Greedy decoding, including all special tokens (no hidden output filtering).
        chosen = int(logits.argmax())
        ids.append(chosen)
        if chosen == 1:
            break
    return tokenizer.decode(ids)


def save(path, model, tokenizer, stage, step):
    torch.save({'format': FORMAT, 'config': model.cfg.__dict__, 'tokenizer': tokenizer.metadata(),
                'stage': stage, 'step': step,
                'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, path)


def load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Use a checkpoint from llm_workshop.domains.train.')
    tokenizer = Tokenizer(state['tokenizer']['kind'], state['tokenizer']['vocabulary'])
    model = TinyGPT(TinyGPTConfig(**state['config'])).to(device).eval()
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer, state


def train(args):
    if args.base_epochs < 1 or args.adapt_epochs < 1 or args.eval_every < 1:
        raise ValueError('Epoch and evaluation counts must be positive.')
    for lr in (args.learning_rate, args.adapt_learning_rate):
        if not math.isfinite(lr) or not 0 < lr <= .1:
            raise ValueError('Learning rates must be in (0, .1].')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    paths = {'biology': args.biology, 'finance': args.finance}
    all_rows = {name: read_csv(path) for name, path in paths.items()}
    if {r['sentence'] for r in all_rows['biology']} & {r['sentence'] for r in all_rows['finance']}:
        raise ValueError('Cross-domain duplicate text could leak across held-out sets.')
    splits = {name: split_rows(rows, args.seed) for name, rows in all_rows.items()}
    tokenizer = Tokenizer.fit(args.tokenization, (r['sentence'] for parts in splits.values() for r in parts['train']))
    # Capacity is an explicit architecture choice, not fitted to validation text.
    context = 64 if args.tokenization == 'word' else 384
    for rows in all_rows.values():
        if stats(rows, tokenizer)['max_input_positions'] > context:
            raise ValueError('Dataset exceeds the chosen context; select a larger architecture explicitly.')
    cfg = TinyGPTConfig(len(tokenizer.vocabulary), block_size=context, n_layer=2, n_head=2, n_embd=64, dropout=0.)
    output = fresh_directory(args.output_dir)
    for source in [Path(__file__), Path(__file__).with_name('data.py'), ROOT/'llm_workshop/tiny.py', ROOT/'llm_workshop/progress.py']:
        shutil.copy2(source, output / ('source_' + source.name))
    manifest = {'sources': {name: {**file_info(path), **stats(all_rows[name], tokenizer)} for name, path in paths.items()},
                'splits': {name: {part: stats(rows, tokenizer) for part, rows in parts.items()} for name, parts in splits.items()},
                'split_method': 'Seeded category-stratified exact 5% validation; exact texts disjoint. Shared templates may remain.',
                'seed': args.seed, 'tokenizer': tokenizer.metadata(), 'config': cfg.__dict__,
                'counting': 'Characters/whitespace words sum the original sentence fields, excluding CSV metadata/separators. Records are not a linguistic sentence count.',
                'vocabulary_scope': 'Both TRAIN domains define the IDs in advance. Finance contributes no gradients during biology pretraining. Held-out text never fits vocabulary.',
                'validation_use': 'Entire validation sets, no gradient updates. Fixed epoch budgets; final checkpoint retained, no early stopping or best-checkpoint selection.'}
    write_json(output/'data_manifest.json', manifest)
    for name, parts in splits.items():
        for part, rows in parts.items():
            (output/f'{name}_{part}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    print(json.dumps({'input_data': manifest['sources'], 'splits': manifest['splits'], 'vocabulary': len(tokenizer.vocabulary), 'device': device}, indent=2), flush=True)
    model = TinyGPT(cfg).to(device)
    save(output/'before.pt', model, tokenizer, 'untrained', 0)
    reports = []

    def stage(name, model, domain, epochs, lr, replay=False):
        history = []
        plot = LossPlot(output/f'{name}_progress.png', f'{args.tokenization.title()} tokens · {name}', args.live_plot)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        order_rng = random.Random(args.seed + (0 if domain == 'biology' else 1))
        replay_rng = random.Random(args.seed + 2)
        rows = splits[domain]['train']
        steps = epochs * math.ceil(len(rows)/32)
        exposures, replay_exposures, targets_seen = 0, 0, 0
        started = time.perf_counter()
        step, epoch = 0, 0

        def record(train_loss=None):
            point = {'step': step, 'epoch': epoch, 'train_batch': train_loss,
                     **{d+'_validation': measure(model, splits[d]['validation'], tokenizer, device) for d in splits}}
            history.append(point)
            write_json(output/f'{name}_progress.json', history)
            plot.update(history, ('train_batch', 'biology_validation', 'finance_validation'))
            print(name, json.dumps(point), flush=True)
        record()
        for epoch_index in range(epochs):
            order = list(range(len(rows)))
            order_rng.shuffle(order)
            for offset in range(0, len(order), 32):
                selected = [rows[i] for i in order[offset:offset+32]]
                exposures += len(selected)
                if replay:
                    extra = max(1, round(len(selected)/4))  # 8 biology + 32 finance = 20% replay.
                    selected += replay_rng.choices(splits['biology']['train'], k=extra)
                    replay_exposures += extra
                x, y = batch(selected, tokenizer, device, context)
                targets_seen += int((y != -100).sum())
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = model(x, y)[1]  # Observed next tokens provide the supervision.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()      # Full fine-tuning: all parameters can change.
                step += 1
                epoch = epoch_index + min(offset+32, len(order))/len(order)
                if step % args.eval_every == 0 or step == steps:
                    record(float(loss.detach()))
        if device == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        plot.close()
        model.eval()
        save(output/f'{name}.pt', model, tokenizer, name, steps)
        report = {'name': name, 'domain': domain, 'epochs': epochs, 'steps': steps, 'learning_rate': lr,
                  'batch_size': 40 if replay else 32, 'main_domain_exposures': exposures,
                  'biology_replay_exposures': replay_exposures, 'target_exposures': targets_seen,
                  'training_seconds': seconds, 'history': history,
                  'samples': [{'prompt': p, 'text': generate(model, tokenizer, p)} for p in PROMPTS],
                  'method': 'Full weight training; not LoRA', 'checkpoint': f'{name}.pt'}
        reports.append(report)
        write_json(output/f'{name}_report.json', report)
        return model

    base = stage('biology_base', model, 'biology', args.base_epochs, args.learning_rate)
    # Identical starting weights and finance order/exposure in the two comparisons.
    stage('finance_only', copy.deepcopy(base), 'finance', args.adapt_epochs, args.adapt_learning_rate)
    stage('finance_replay', copy.deepcopy(base), 'finance', args.adapt_epochs, args.adapt_learning_rate, replay=True)
    report = {'format': FORMAT, 'manifest': manifest, 'stages': reports,
              'parameter_count': sum(p.numel() for p in base.parameters()),
              'device': device, 'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
              'torch_version': str(torch.__version__), 'seed': args.seed,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
              'timing': 'Training loop + full validation + progressive plot writes; excludes interpreter startup and final export.',
              'checkpoint_scope': 'Inference/comparison, no optimizer resume.',
              'limits': 'Synthetic/template-like rows; exact holdout is not a factual or arithmetic benchmark. Validation loss across different tokenizers is not directly comparable.'}
    write_json(output/'report.json', report)
    print('Saved:', output, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--biology', type=Path, default=ROOT/'data/external/animal_biology_sentences.csv')
    parser.add_argument('--finance', type=Path, default=ROOT/'data/external/finance_sentences.csv')
    parser.add_argument('--tokenization', choices=['word', 'character'], default='word')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--base-epochs', type=int, default=5)
    parser.add_argument('--adapt-epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--adapt-learning-rate', type=float, default=.0003)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--live-plot', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
