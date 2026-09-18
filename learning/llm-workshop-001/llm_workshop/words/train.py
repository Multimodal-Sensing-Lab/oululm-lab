"""Train the readable word companion on CPU or CUDA; no browser/UI code here.

python -m llm_workshop.words.train --device auto --steps 600 --output-dir runs/words_01
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import torch
from llm_workshop.io import read_jsonl, assert_disjoint, fresh_directory, sha256, write_json
from llm_workshop.tiny import TinyGPT, TinyGPTConfig
from llm_workshop.progress import LossPlot
from .corpus import DATA
from .tokenizer import WordTokenizer
from .model import batch, measure, generate, save_checkpoint, device_for

PROMPTS = ['the small cat', 'the happy teacher', 'the green plant', 'the quiet robot carries']


def train(output, data_dir=DATA, steps=600, device='auto', seed=1337, learning_rate=.003, live_plot=False):
    if not 1 <= steps <= 20000 or not math.isfinite(learning_rate) or not 0 < learning_rate <= .1:
        raise ValueError('Use 1–20000 updates and a learning rate in (0, 0.1].')
    device = device_for(device)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    rows = {name: read_jsonl(data_dir / f'{name}.jsonl') for name in ['train', 'validation', 'test']}
    assert_disjoint(rows)
    tokenizer = WordTokenizer.fit(r['text'] for r in rows['train'])
    cfg = TinyGPTConfig(len(tokenizer.vocabulary), block_size=16, n_layer=2, n_head=2, n_embd=64, dropout=0.)
    data = {name: batch(split, tokenizer, cfg.block_size, device) for name, split in rows.items()}
    output = fresh_directory(output)
    model = TinyGPT(cfg).to(device)
    before = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = []
    plot = LossPlot(output / "progress.png", "Whole-word classroom training", live_plot)
    print("Input data:", data_dir, "| splits:", {k: len(v) for k, v in rows.items()}, flush=True)
    started = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    hashes = {name: sha256(data_dir / f'{name}.jsonl') for name in rows}
    save_checkpoint(output / 'before.pt', before, tokenizer, 0, data_hashes=hashes)
    for step in range(steps + 1):
        if step % 50 == 0 or step == steps:
            point = {'step': step, **{name: measure(model, data[name]) for name in ['train', 'validation']}}
            history.append(point)
            print(json.dumps(point), flush=True)
            plot.update(history)
        if step == steps:
            break
        indices = torch.randint(len(rows['train']), (32,), generator=generator).to(device)
        x, y = (values[indices] for values in data['train'])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)       # Score each observed next word/punctuation/end token.
        loss.backward()            # Compute how each parameter affects the penalty.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()           # This line changes the stored model parameters.
    plot.close()
    model.eval()
    if device == 'cuda':
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    # Fixed update budget: test is reported once, never used to choose a checkpoint.
    test = {'before': measure(before, data['test']), 'after': measure(model, data['test'])}
    samples = [{'prompt': p, 'before': generate(before, tokenizer, p),
                'after': generate(model, tokenizer, p)} for p in PROMPTS]
    save_checkpoint(output / 'after.pt', model, tokenizer, steps, data_hashes=hashes)
    report = {'seed': seed, 'steps': steps, 'device': device,
        'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
        'torch_version': str(torch.__version__), 'learning_rate': learning_rate,
        'batch_size': 32, 'config': cfg.__dict__, 'parameter_count': sum(p.numel() for p in model.parameters()),
        'training_seconds': training_seconds,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'tokenizer': tokenizer.metadata(), 'history': history, 'test': test, 'samples': samples,
        'data_hashes': hashes, 'corpus': json.loads((data_dir / 'manifest.json').read_text()),
        'evaluation': 'All non-padding targets in each split, including punctuation and EOS; no gradients on validation/test',
        'checkpoint_type': 'Inference/comparison only; no optimizer-resume state',
        'scope': 'Controlled templates with held-out sentences; shared vocabulary and grammar, not general English'}
    write_json(output / 'report.json', report)
    print('Device:', report['device_name'], 'Training seconds:', round(training_seconds, 2))
    print('Test loss:', test)
    for sample in samples:
        print(sample['prompt'], '\n  before:', sample['before']['text'], '\n  after: ', sample['after']['text'])
    print('Saved:', output)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--live-plot', action='store_true', help='Show the plot window; PNG/SVG are always saved')
    args = parser.parse_args()
    train(args.output_dir, args.data_dir, args.steps, args.device, learning_rate=args.learning_rate, live_plot=args.live_plot)
