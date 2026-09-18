"""Optional word LoRA lab: bias a fixed base toward one existing corpus family.

python -m llm_workshop.words.adapt --base-checkpoint runs/words_classroom_01/after.pt \
    --family garden --device auto --output-dir runs/words_garden_01
"""
import argparse
import copy
from pathlib import Path
import torch
from llm_workshop.teaching import topic
from llm_workshop.io import read_jsonl, fresh_directory, sha256, write_json, assert_disjoint
from llm_workshop.progress import LossPlot
from .corpus import DATA
from .model import load_checkpoint, device_for, batch, measure, generate, save_checkpoint


def adapt(base_checkpoint, output, family='garden', steps=100, rank=4, device='auto', data_dir=DATA, live_plot=False):
    if family not in {'animals', 'classroom', 'garden'} or not 1 <= steps <= 2000:
        raise ValueError('Choose a corpus family and 1–2000 updates.')
    device = device_for(device)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base, tokenizer, state = load_checkpoint(base_checkpoint, device)
    all_rows = {name: read_jsonl(data_dir / f'{name}.jsonl') for name in ['train', 'validation', 'test']}
    assert_disjoint(all_rows)
    hashes = {name: sha256(data_dir / f'{name}.jsonl') for name in all_rows}
    if state.get('data_hashes') != hashes:
        raise ValueError('The adapter must use the same frozen corpus as its base checkpoint.')
    rows = {name: [r for r in values if r['family'] == family] for name, values in all_rows.items()}
    if any(not values for values in rows.values()):
        raise ValueError('Every split must contain examples of this family.')
    data = {name: batch(values, tokenizer, base.cfg.block_size, device) for name, values in rows.items()}
    model = copy.deepcopy(base).eval().requires_grad_(False)
    model.lm_head = topic(6).LoRALinear(model.lm_head, rank=rank).to(device)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.01)
    rng = torch.Generator().manual_seed(42)
    output = fresh_directory(output)
    history = []
    plot = LossPlot(output / "progress.png", f"Word LoRA · {family}", live_plot)
    print("Input:", data_dir, "| family:", family, "| records:", {k: len(v) for k, v in rows.items()})
    for step in range(steps+1):
        if step % 20 == 0 or step == steps:
            point = {'step': step, **{name: measure(model, data[name]) for name in ['train', 'validation']}}
            history.append(point)
            print(point, flush=True)
            plot.update(history)
        if step == steps:
            break
        indices = torch.randint(len(rows['train']), (32,), generator=rng).to(device)
        x, y = (v[indices] for v in data['train'])
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
    plot.close()
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    merged = copy.deepcopy(model)
    layer = merged.lm_head
    with torch.no_grad():
        layer.base.weight.add_(layer.b @ layer.a, alpha=layer.scale)
    merged.lm_head = layer.base
    torch.testing.assert_close(model(data['validation'][0])[0], merged(data['validation'][0])[0], rtol=1e-4, atol=1e-5)
    save_checkpoint(output / 'merged.pt', merged, tokenizer, steps, data_hashes=hashes,
                    adaptation_family=family, base_sha256=sha256(base_checkpoint))
    torch.save({'a': layer.a.detach().cpu(), 'b': layer.b.detach().cpu(), 'scale': layer.scale,
                'format': 'word-lm-head-lora-v1', 'base_sha256': sha256(base_checkpoint)}, output / 'adapter_matrices.pt')
    prompts = ['the', 'the small', 'the quiet']
    report = {'family': family, 'device': device, 'rank': rank, 'steps': steps,
        'base_checkpoint': str(base_checkpoint), 'base_sha256': sha256(base_checkpoint),
        'data_hashes': hashes, 'history': history,
        'trainable_parameters': sum(p.numel() for p in trainable),
        'test': {'base': measure(base, data['test']), 'adapter': measure(merged, data['test'])},
        'samples': [{'prompt': p, 'base': generate(base, tokenizer, p),
                     'adapter': generate(merged, tokenizer, p)} for p in prompts],
        'scope': 'Output-head LoRA on an existing corpus family; measures a distribution shift, not new facts'}
    write_json(output / 'report.json', report)
    print('Trainable parameters:', report['trainable_parameters'], 'Test loss:', report['test'])
    print('Saved:', output)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--family', choices=['animals', 'classroom', 'garden'], default='garden')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--live-plot', action='store_true')
    args = parser.parse_args()
    adapt(args.base_checkpoint, args.output_dir, args.family, args.steps, args.rank, args.device, args.data_dir, args.live_plot)
