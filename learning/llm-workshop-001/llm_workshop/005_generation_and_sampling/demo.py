"""Generate one token at a time and inspect actual scores/probabilities."""
# After topic 004 training, reuse the learned embeddings and blocks to predict.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
import torch
from llm_workshop.teaching import Session, common_parser, show


def next_token(session, text, temperature=.8, greedy=False, generator=None):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive; use greedy mode separately.')
    # The simple TinyGPT uses a sliding context when generation grows past its limit.
    ids = session.ids(text[-session.model.cfg.block_size:])
    session.model.eval()  # Disable training-time dropout.
    with torch.inference_mode():
        logits = session.model(ids)[0][0, -1]  # Use vocabulary scores from the last position.
        probabilities = torch.softmax(logits / temperature, dim=-1)  # Temperature-adjusted next-token probabilities.
        chosen = int(logits.argmax()) if greedy else int(torch.multinomial(probabilities, 1, generator=generator))
    return {'logits': logits, 'probabilities': probabilities, 'chosen': chosen,
            'token': session.vocabulary[chosen], 'text': text + session.vocabulary[chosen]}


class Generation:
    """Keep IDs internally, including <UNK>; never tokenize its display spelling."""
    def __init__(self, session, prompt='the ', seed=7):
        self.session = session
        self.ids = session.ids(prompt)[0].tolist()  # Keep token IDs as the growing context.
        self.generator = torch.Generator().manual_seed(seed)

    def advance(self, temperature=.8, greedy=False):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Use a finite positive temperature.')
        with torch.inference_mode():
            ids = torch.tensor([self.ids[-self.session.model.cfg.block_size:]])  # Use the most recent context window.
            logits = self.session.model(ids)[0][0, -1]  # Take the last position's vocabulary scores.
            probs = (logits / temperature).softmax(-1)  # Convert scores into a sampling distribution.
            chosen = int(logits.argmax()) if greedy else int(torch.multinomial(probs, 1, generator=self.generator))
        self.ids.append(chosen)  # Append one token; predict again on the next call.
        return {'logits': logits, 'probabilities': probs, 'chosen': chosen,
                'token': self.session.vocabulary[chosen], 'text': self.session.decode(self.ids)}


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--tokens', type=int, default=20)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 256:
        parser.error('Choose 1–256 new tokens.')
    session = Session()
    if args.checkpoint:
        session.load(args.checkpoint)  # Load saved weights and their tokenizer.
    generation = Generation(session, args.text)
    print(session.label)
    for i in range(args.tokens):
        result = generation.advance(args.temperature, args.greedy)  # Choose one token without updating weights.
        show(f'Token {i+1}: {result["token"]!r}', result['text'], args.step)
    print('Generation changes the context, not the weights.')


if __name__ == '__main__':
    main()
