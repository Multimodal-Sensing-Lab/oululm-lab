"""Look inside a word checkpoint, then generate one whole token at a time.

python -m llm_workshop.words.demo --checkpoint runs/words_01/after.pt --text 'the small cat' --step
"""
import argparse
import math
from functools import partial
from pathlib import Path
import torch
from llm_workshop.teaching import topic
from .model import load_checkpoint, generate, WordSession


def choose_interactively(tokenizer, ids, probabilities, suggested):
    """Accept a prediction or append a user-supplied vocabulary token."""
    print('\nCurrent sentence:', tokenizer.decode(ids))
    top = probabilities.topk(min(8, len(tokenizer.vocabulary)))
    print('Top next tokens:', [(tokenizer.vocabulary[i], round(p, 4))
                              for i, p in zip(top.indices.tolist(), top.values.tolist())])
    print(f'Model suggestion: {tokenizer.vocabulary[suggested]!r}')
    while True:
        answer = input('Enter = accept; type one word (or .); /words = vocabulary; /quit = stop: ').strip()
        if answer == '/quit':
            return None
        if answer == '/words':
            print('Available input tokens:', ' '.join(tokenizer.vocabulary[3:]))
            continue
        chosen = suggested
        if answer:
            try:
                entered = tokenizer.encode(answer)
                if len(entered) != 1:
                    raise ValueError('Enter one word at a time; enter punctuation separately.')
                chosen = entered[0]
            except ValueError as error:
                print(f'{error} Use /words to see the vocabulary. Sentence unchanged.')
                continue
        source = 'Manual token' if answer else 'Model token'
        print(f'{source}: {tokenizer.vocabulary[chosen]!r}\n{tokenizer.decode([*ids, chosen])}')
        return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--text', default='the small cat')
    parser.add_argument('--tokens', type=int, default=12)
    parser.add_argument('--sample', action='store_true', help='Sample instead of the default greedy comparison')
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--inspect', action='store_true', help='Show embeddings, attention and output scores first')
    parser.add_argument('--step', action='store_true', help='Accept each prediction or type your own next word; this does not train')
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0 or not 1 <= args.tokens <= 64:
        parser.error('Use positive finite temperature and 1–64 new tokens.')
    torch.set_num_threads(2)
    model, tokenizer, state = load_checkpoint(args.checkpoint)
    session = WordSession(model, tokenizer)
    ids = session.ids(args.text)[0].tolist()
    print(f'Word checkpoint, {state["step"]} updates. CPU inference; fixed weights.')
    print('Tokens:', [tokenizer.vocabulary[i] for i in ids], '\nIDs:', ids)
    if args.inspect:
        with torch.inference_mode():
            x = session.ids(args.text)
            token_vectors = model.token_embedding(x)[0]
            positions = model.position_embedding(torch.arange(x.shape[1]))
            for name, values in [('Token lookup', token_vectors), ('Position lookup', positions),
                                 ('Sum', token_vectors + positions)]:
                print(name, tuple(values.shape), '\n', values[:, :4])
            result = topic(3).inspect(session, args.text)
            print('Block 0 / head 0 attention (reader rows, source columns):\n', result['attention'])
            logits = model(x)[0][0, -1]
            top = (logits / args.temperature).softmax(-1).topk(8)
            print('Top next tokens:', [(tokenizer.vocabulary[i], round(p, 4)) for i, p in zip(top.indices.tolist(), top.values.tolist())])
    try:
        if args.step:
            print('Your words change the context, not the weights. Predictions are recalculated after each choice.')
            print('Manual and model tokens both count toward --tokens. <EOS> ends the sentence when accepted.')
        result = generate(model, tokenizer, args.text, args.tokens, args.temperature, not args.sample,
                          choose_token=partial(choose_interactively, tokenizer) if args.step else None)
        if not args.step:
            for event in result['events']:
                print(f'Next token: {event["token"]!r}\n{event["text"]}')
        print('Stopped:', result['stop'])
    except (KeyboardInterrupt, EOFError):
        print('\nDisplay stopped. Model weights are unchanged.')


if __name__ == '__main__':
    main()
