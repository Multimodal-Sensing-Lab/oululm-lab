"""Real LoRA updates on a tiny model, followed by base/adapter output comparison.

This CPU mechanics lab is separate from pretrained instruction-model fine-tuning:
use train_pretrained.py for that experiment. Load a trained topic-004 checkpoint
for a meaningful adapted base; without one this starts from random weights.
"""
# Reuse the trained topic-004 base; update only small adapter matrices here.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import copy
import json
import math
import torch
from torch import nn
from llm_workshop.teaching import Session, common_parser, tokenizer
from llm_workshop.progress import LossPlot
from llm_workshop.io import fresh_directory

# Fictional didactic prose, not biological/financial advice. All use the fixed alphabet.
DOMAIN_TEXTS = {
    'biology': 'cells grow in a small leaf. a plant needs light and water. the green leaf grows. ',
    'email': 'dear team. please send the report. thank you for the help. have a good day. ',
    'finance': 'the report shows sales and profit. sales grow and costs fall. the firm has cash. ',
}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        if not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError('Rank must fit the input and output dimensions.')
        self.base = base
        self.base.requires_grad_(False)  # Freeze the original linear-layer parameters.
        self.a = nn.Parameter(torch.empty(rank, base.in_features))  # Learn a projection into the smaller rank.
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))  # Project back; zero starts with no update.
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        # W_eff = W_base + (alpha/r) B A. Only A and B are trainable.
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale


class AdapterRun:
    def __init__(self, session, domain='biology', rank=4, learning_rate=.02):
        if domain not in DOMAIN_TEXTS:
            raise ValueError('Select biology, email or finance.')
        self.session, self.domain, self.base_label = session, domain, session.label
        self.model = copy.deepcopy(session.model).eval().requires_grad_(False)  # Copy and freeze the base model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model.lm_head = LoRALinear(self.model.lm_head, rank=rank)  # Adapt only the vocabulary output layer.
        self.initial = {name: p.detach().clone() for name, p in self.model.named_parameters() if not p.requires_grad}
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]  # Collect only adapter parameters.
        self.optimizer = torch.optim.AdamW(self.trainable, lr=learning_rate)
        self.data = torch.tensor(tokenizer.encode(DOMAIN_TEXTS[domain] * 3, session.stoi))
        self.block = min(32, self.model.cfg.block_size)
        self.rng = torch.Generator().manual_seed(42)
        self.step = 0
        self.history = [self.measure()]

    def measure(self):
        with torch.inference_mode():
            x, y = self.data[:self.block][None], self.data[1:self.block+1][None]
            return {'step': self.step, 'adaptation_loss': self.model(x, y)[1].item()}

    def advance(self):
        starts = torch.randint(len(self.data)-self.block, (4,), generator=self.rng)
        x = torch.stack([self.data[i:i+self.block] for i in starts])
        y = torch.stack([self.data[i+1:i+self.block+1] for i in starts])
        self.optimizer.zero_grad(set_to_none=True)  # Clear previous adapter gradients.
        _, loss = self.model(x, y)  # Compare next-token predictions with targets.
        loss.backward()  # Compute gradients for trainable adapter weights.
        torch.nn.utils.clip_grad_norm_(self.trainable, 1., error_if_nonfinite=True)
        self.optimizer.step()  # Update A and B; keep the base frozen.
        self.step += 1
        if self.step % 10 == 0:
            self.history.append(self.measure())
        return loss.item()

    def compare(self, prompt='the '):
        # Reconstruct the unchanged base from the adapted copy, not a mutable UI session.
        base = copy.deepcopy(self.model)
        base.lm_head = base.lm_head.base
        ids = self.session.ids(prompt)
        return {'base': self.session.decode(base.generate(ids, 40, 0)[0].tolist()),
                'adapter': self.session.decode(self.model.generate(ids, 40, 0)[0].tolist())}

    def assert_base_unchanged(self):
        for name, p in self.model.named_parameters():
            if name in self.initial:
                torch.testing.assert_close(p, self.initial[name], rtol=0, atol=0)

    def save(self, output):
        self.assert_base_unchanged()
        output = fresh_directory(Path(output))
        # Save a normal TinyGPT with the adapter mathematically merged for topics 005/008.
        merged = copy.deepcopy(self.model)
        layer = merged.lm_head
        base = copy.deepcopy(layer.base)
        with torch.no_grad():
            base.weight.add_(layer.b @ layer.a, alpha=layer.scale)  # Fold the learned update into a normal weight matrix.
        merged.lm_head = base
        torch.save({'model_state_dict': merged.state_dict(), 'config': merged.cfg.__dict__,
                    'stoi': self.session.stoi, 'itos': dict(enumerate(self.session.vocabulary)),
                    'step': self.step}, output/'merged.pt')
        torch.save({'a': layer.a.detach(), 'b': layer.b.detach(), 'scale': layer.scale}, output/'adapter_matrices.pt')
        (output/'report.json').write_text(json.dumps({'domain': self.domain, 'base': self.base_label,
            'rank': layer.a.shape[0], 'history': self.history + [self.measure()], 'comparison': self.compare(),
            'scope': 'tiny character LM; lm_head only; prose adaptation; not HF/PEFT format',
            'trainable_parameters': sum(p.numel() for p in self.trainable)}, indent=2))
        return output


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--domain', choices=DOMAIN_TEXTS, default='biology')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--live-plot', action='store_true', help='Show a live loss window')
    parser.add_argument('--plot-file', type=Path, help='Continuously updated PNG (SVG saved on completion)')
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error('Choose 1–1000 updates.')
    session = Session()
    if args.base_checkpoint:
        session.load(args.base_checkpoint)
    print('Base:', session.label)
    print('This is a tiny LoRA mechanics lab. Instruction-model training uses train_pretrained.py.')
    run = AdapterRun(session, args.domain, args.rank)
    print('Training only A/B parameters:', sum(p.numel() for p in run.trainable))
    plot_path = args.plot_file or (args.output_dir.parent / (args.output_dir.name + '_progress.png') if args.output_dir else None)
    plot = LossPlot(plot_path or Path('/tmp/llm-classroom-progress.png'), 'Character training loss', args.live_plot) if plot_path or args.live_plot else None
    if plot:
        plot.update(run.history, ('adaptation_loss',))
    print('Input domain prose:', args.domain, repr(DOMAIN_TEXTS[args.domain]))
    print('Before:', run.history[0])
    for _ in range(args.steps):
        run.advance()
        if run.step % 10 == 0:
            print(run.history[-1])
            if plot:
                plot.update(run.history, ('adaptation_loss',))
            tokenizer.pause(args.step)
    run.assert_base_unchanged()
    if plot:
        plot.update(run.history + ([run.measure()] if run.history[-1]['step'] != run.step else []), ('adaptation_loss',))
        plot.close()
    print('After:', run.measure(), '\nSame prompt comparison:', run.compare())
    if args.output_dir:
        print('Saved:', run.save(args.output_dir))


if __name__ == '__main__':
    main()
