"""Create fictional email request/reply data; no private mailbox is read.

Use two styles with identical requests/splits to compare response behavior.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import random
from llm_workshop.data import record, finish
from llm_workshop.io import fresh_directory


def prepare(output, style='formal', seed=42):
    if style not in {'formal', 'concise'}:
        raise ValueError('Choose formal or concise.')
    output = fresh_directory(Path(output))
    indices = list(range(80))
    random.Random(seed).shuffle(indices)
    splits = {'train': [], 'validation': [], 'test': []}
    for position, index in enumerate(indices):
        ticket = f'WORKSHOP-{index:03d}'
        subject = ['room booking', 'projector request', 'poster review', 'meeting notes'][index % 4]
        date = f'October {index % 28 + 1}'
        prompt = (f'Write a reply to this fictional email. Do not invent commitments beyond the supplied details.\n'
                  f'Request {ticket}: Please confirm receipt of my {subject} for {date}. '
                  'The only confirmed next step is that the teaching team will review it.')
        if style == 'formal':
            answer = (f'Dear colleague,\n\nThank you for your message regarding the {subject} for {date} '
                      f'({ticket}). We confirm receipt. The teaching team will review your request.\n\n'
                      'Kind regards,\nWorkshop support')
        else:
            answer = f'Received {ticket}: {subject} for {date}. The teaching team will review it.'
        split = 'train' if position < 60 else 'validation' if position < 70 else 'test'
        splits[split].append(record(ticket, ticket, prompt, answer, 'email'))
    finish(output, splits, {'source': 'Original fictional workshop templates; no personal email',
                           'style': style, 'seed': seed,
                           'split_method': '60/10/10 disjoint request IDs; shared templates, not unseen writing tasks',
                           'scope': 'Format/style adaptation demonstration, not a benchmark of email quality'})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--style', choices=['formal', 'concise'], default='formal')
    args = parser.parse_args()
    print(prepare(args.output_dir, args.style))
