"""Word IDs use the same TinyGPT math as character IDs; only the vocabulary differs."""
from pathlib import Path
import math
import torch
from llm_workshop.tiny import TinyGPT, TinyGPTConfig
from .tokenizer import WordTokenizer

FORMAT = 'llm-workshop-word-v1'


def device_for(requested):
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in this interpreter. Use --device cpu or check your environment.')
    if requested not in {'cpu', 'cuda'}:
        raise ValueError('Choose cpu, cuda or auto.')
    return requested


def save_checkpoint(path, model, tokenizer, step, **metadata):
    torch.save({'format': FORMAT, 'config': model.cfg.__dict__,
                'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'tokenizer': tokenizer.metadata(), 'step': step, **metadata}, path)


def load_checkpoint(path, device='cpu'):
    state = torch.load(Path(path), map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT or state['tokenizer'].get('kind') != 'word':
        raise ValueError('Use a word-companion checkpoint, not a character or pretrained-model checkpoint.')
    tokenizer = WordTokenizer(state['tokenizer']['vocabulary'])
    if state['config']['vocab_size'] != len(tokenizer.vocabulary):
        raise ValueError('Checkpoint vocabulary and model dimensions disagree.')
    model = TinyGPT(TinyGPTConfig(**state['config'])).to(device).eval()
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer, state


class WordSession:
    """Small bridge to the existing explicit attention-inspection lesson (CPU)."""
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    def ids(self, text):
        return torch.tensor([self.tokenizer.prompt(text, self.model.cfg.block_size)])


@torch.inference_mode()
def generate(model, tokenizer, prompt, tokens=12, temperature=.8, greedy=True, seed=7,
             *, choose_token=None):
    """Generate incrementally; an optional chooser can override each suggested ID.

    The chooser receives (current IDs, probabilities, suggested ID). Return an ID
    to append, or None to stop. The next forward pass uses the chosen context.
    """
    if not math.isfinite(temperature) or temperature <= 0 or not 1 <= tokens <= 64:
        raise ValueError('Use positive finite temperature and 1–64 new tokens.')
    ids = tokenizer.prompt(prompt, model.cfg.block_size)
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    events = []
    stop = 'token budget'
    for _ in range(tokens):
        if len(ids) >= model.cfg.block_size:
            stop = 'context full'
            break
        logits = model(torch.tensor([ids], device=device))[0][0, -1]
        probabilities = (logits / temperature).softmax(-1)
        chosen = int(logits.argmax()) if greedy else int(torch.multinomial(probabilities, 1, generator=rng))
        if choose_token is not None:
            chosen = choose_token(tuple(ids), probabilities, chosen)
            if chosen is None:
                stop = 'user stopped'
                break
            if not isinstance(chosen, int) or not 0 <= chosen < len(tokenizer.vocabulary):
                raise ValueError('The token chooser must return a vocabulary ID or None.')
        ids.append(chosen)
        events.append({'id': chosen, 'token': tokenizer.vocabulary[chosen],
                       'text': tokenizer.decode(ids)})
        if chosen == 1:
            stop = 'end token'
            break
    return {'prompt': prompt, 'ids': ids, 'text': tokenizer.decode(ids),
            'events': events, 'stop': stop, 'decoding': 'greedy' if greedy else 'sampling'}


def batch(rows, tokenizer, block_size, device):
    # Each sentence starts fresh. Padding uses EOS in x and ignored -100 in y.
    # Causality ensures trailing padding cannot affect earlier real positions.
    x = torch.full((len(rows), block_size), 1, dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, row in enumerate(rows):
        ids = tokenizer.encode(row['text'], boundaries=True)
        if len(ids) - 1 > block_size:
            raise ValueError('A sentence exceeds the model context; no silent truncation.')
        x[i, :len(ids)-1] = torch.tensor(ids[:-1], device=device)
        y[i, :len(ids)-1] = torch.tensor(ids[1:], device=device)
    return x, y


@torch.inference_mode()
def measure(model, data, batch_size=64):
    model.eval()
    total, count = 0., 0
    x, y = data
    for offset in range(0, len(x), batch_size):
        targets = y[offset:offset+batch_size]
        n = int((targets != -100).sum())
        loss = model(x[offset:offset+batch_size], targets)[1]
        total += float(loss) * n
        count += n
    return total / count
