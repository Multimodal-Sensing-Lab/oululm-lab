"""Small shared CPU setup for the numbered teaching demos (no UI code)."""
from importlib import import_module
from pathlib import Path
import torch
from .tiny import TinyGPT, TinyGPTConfig, load_checkpoint

tokenizer = import_module('llm_workshop.001_text_to_predictions.tokenizer_demo')
ROOT = Path(__file__).resolve().parents[1]
TOPICS = sorted(p for p in Path(__file__).parent.iterdir() if p.is_dir() and p.name[:3].isdigit())


def topic(number):
    folder = next(p for p in TOPICS if p.name.startswith(f'{int(number):03d}_'))
    return import_module(f'llm_workshop.{folder.name}.demo')


class Session:
    """One fixed vocabulary and model; editing a prompt never reinitializes weights."""
    def __init__(self, seed=1337):
        torch.set_num_threads(2)
        self.seed = seed
        self.vocabulary, self.stoi = tokenizer.build_vocabulary(tokenizer.read_corpus(tokenizer.CORPUS))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = TinyGPT(TinyGPTConfig(len(self.vocabulary), block_size=64,
                                n_layer=2, n_head=2, n_embd=64, dropout=.1)).cpu().eval()
        self.label = f'Untrained initialization, seed {seed}'

    def ids(self, text):
        if not text or len(text) > self.model.cfg.block_size:
            raise ValueError(f'Use 1–{self.model.cfg.block_size} characters.')
        return torch.tensor([tokenizer.encode(text, self.stoi)], dtype=torch.long)

    def decode(self, ids):
        return tokenizer.decode(ids, self.vocabulary)

    def load(self, path):
        model, checkpoint = load_checkpoint(Path(path), 'cpu')
        if model.cfg.n_embd > 512 or model.cfg.n_layer > 12:
            raise ValueError('Use a tiny teaching checkpoint in this desktop lab.')
        self.model = model.eval()
        self.stoi = checkpoint['stoi']
        self.vocabulary = [checkpoint['itos'][i] for i in range(len(checkpoint['itos']))]
        self.label = f'{Path(path).name}, step {checkpoint.get("step", "unknown")}'


def common_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--text', default='the cat sat.')
    parser.add_argument('--step', action='store_true', help='Pause between printed stages')
    return parser


def show(title, value, step=False):
    print(f'\n{title}\n{value}')
    tokenizer.pause(step)
