"""Short CPU training loop with measured loss curves and comparable samples.

This demonstrates full model training. The longer resumable trainer is tiny.py.
"""
# Topics 001–003 built predictions; now targets tell us how to change weights.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import copy
import hashlib
import json
import re
from datetime import datetime
import torch
from llm_workshop.teaching import Session, ROOT, tokenizer
from llm_workshop.progress import LossPlot
from llm_workshop.io import fresh_directory


class TrainingRun:
    def __init__(self, session, learning_rate=.003):
        if not 0 < learning_rate <= .1:
            raise ValueError('Choose a learning rate in (0, 0.1].')
        self.session = session
        self.model = session.model  # Train the same model held by the session.
        self.before = copy.deepcopy(self.model).eval()  # Keep an unchanged starting snapshot.
        # Distinct documents; vocabulary is fitted on train only.
        self.texts = {name: tokenizer.read_corpus(ROOT / f'examples/data/tiny_{name}.jsonl')
                      for name in ('train', 'validation')}
        self.data_specs = {name: {
            'path': f'examples/data/tiny_{name}.jsonl',
            'documents': sum(bool(line.strip()) for line in
                             (ROOT / f'examples/data/tiny_{name}.jsonl').read_text().splitlines()),
            'sentences_by_punctuation': len(re.findall(r'[.!?]+', text)),
            'whitespace_words': len(text.split()), 'joined_characters': len(text),
        } for name, text in self.texts.items()}
        self.data = {name: torch.tensor(tokenizer.encode(text, session.stoi)) for name, text in self.texts.items()}
        self.block = min(32, self.model.cfg.block_size)
        self.generator = torch.Generator().manual_seed(session.seed)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)  # Configure updates for all model weights.
        self.learning_rate = learning_rate
        self.step = 0
        self.history = [self.measure()]
        self.samples = [self.sample_pair()]

    def measure(self):
        self.model.eval()  # Disable training-time dropout for inspection.
        losses = {}
        with torch.inference_mode():
            for name, data in self.data.items():
                starts = torch.linspace(0, len(data)-self.block-1, steps=4).long()
                x = torch.stack([data[i:i+self.block] for i in starts])  # Batch the input token sequences.
                y = torch.stack([data[i+1:i+self.block+1] for i in starts])  # Shift by one token to form targets.
                losses[name] = self.model(x, y)[1].item()
        return {'step': self.step, **losses}

    def advance(self):
        # One optimizer update; all model parameters are trainable here.
        data = self.data['train']
        starts = torch.randint(len(data)-self.block, (4,), generator=self.generator)  # Pick four training windows.
        x = torch.stack([data[i:i+self.block] for i in starts])  # Batch the input token sequences.
        y = torch.stack([data[i+1:i+self.block+1] for i in starts])  # Shift by one token to form targets.
        self.model.train()  # Enable training-time behavior, including dropout.
        self.optimizer.zero_grad(set_to_none=True)  # Clear gradients from the previous update.
        # Reproducible dropout without consuming randomness used by sampling demos.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.session.seed + self.step)
            _, loss = self.model(x, y)  # Run embeddings + blocks + output head; measure error.
            loss.backward()  # Compute gradients of the prediction loss.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)  # Limit the total gradient norm.
        self.optimizer.step()  # Apply one update to the weights.
        self.model.eval()  # Disable training-time dropout for inspection.
        self.step += 1
        self.session.label = f'CPU classroom training: {self.step} updates this run'
        if self.step % 10 == 0:
            self.history.append(self.measure())
        if self.step % 50 == 0:
            self.samples.append(self.sample_pair())
        return loss.item()

    def sample_pair(self, prompt='the ', length=40):
        ids = self.session.ids(prompt)
        return {'step': self.step, 'prompt': prompt, 'decoding': 'greedy',
                'before': self.session.decode(self.before.generate(ids, length, 0)[0].tolist()),
                'after': self.session.decode(self.model.generate(ids, length, 0)[0].tolist())}

    def save(self, output):
        output = fresh_directory(Path(output))  # Save a new run without overwriting another.
        if self.history[-1]['step'] != self.step:
            self.history.append(self.measure())
        for name, model in [('before', self.before), ('after', self.model)]:
            torch.save({'model_state_dict': model.state_dict(), 'config': model.cfg.__dict__,
                        'stoi': self.session.stoi, 'itos': dict(enumerate(self.session.vocabulary)),
                        'step': 0 if name == 'before' else self.step}, output / f'{name}.pt')
        report = {'initial_model': 'snapshot at start of this classroom run', 'seed': self.session.seed,
                  'learning_rate': self.learning_rate, 'history': self.history,
                  'input_data': self.data_specs, 'batch_size': 4, 'training_context': self.block,
                  'samples': self.samples + [self.sample_pair()],
                  'data_hashes': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in self.texts.items()},
                  'checkpoint_type': 'inference comparison only; no optimizer-resume state'}
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--step', action='store_true', help='Pause after each group of 10 updates')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'experiments'/datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    parser.add_argument('--live-plot', action='store_true', help='Show a live loss window')
    parser.add_argument('--plot-file', type=Path, help='Continuously updated PNG (SVG saved on completion)')
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000:
        parser.error('Choose 1–2000 updates for this CPU demonstration.')
    run = TrainingRun(Session(), args.learning_rate)
    plot_path = args.plot_file or (args.output_dir.parent / (args.output_dir.name + '_progress.png') if args.output_dir else None)
    plot = LossPlot(plot_path or Path('/tmp/llm-classroom-progress.png'), 'Character training loss', args.live_plot) if plot_path or args.live_plot else None
    if plot:
        plot.update(run.history, ('train', 'validation'))
    print('Input data:', json.dumps(run.data_specs, indent=2))
    print('CPU | batch: 4 ×', run.block, 'positions | learning rate:', args.learning_rate)
    print('Before:', run.history[0], run.samples[0])
    for _ in range(args.steps):
        run.advance()  # Perform one complete training update.
        if run.step % 10 == 0:
            print(run.history[-1])
            if plot:
                plot.update(run.history, ('train', 'validation'))
            tokenizer.pause(args.step)
    if plot:
        plot.update(run.history + ([run.measure()] if run.history[-1]['step'] != run.step else []), ('train', 'validation'))
        plot.close()
    print('After:', run.sample_pair())
    print('Saved:', run.save(args.output_dir))


if __name__ == '__main__':
    main()
