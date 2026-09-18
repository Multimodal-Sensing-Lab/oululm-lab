"""Inspect real embedding lookups and position vectors. No plotting or HTML."""
# From topic 001: IDs address rows; their summed vectors enter topic 003.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from llm_workshop.teaching import Session, common_parser, show


def inspect(session, text):
    ids = session.ids(text)  # Encode one prompt: [batch=1, token count].
    with torch.inference_mode():  # Inspect values without tracking gradients.
        # IDs select rows. The position vector changes even when the token repeats.
        tokens = session.model.token_embedding(ids)[0]  # Select token rows; remove the batch axis.
        positions = session.model.position_embedding(torch.arange(ids.shape[1]))  # Look up positions 0 through T−1.
        combined = tokens + positions  # Add matching features, keeping the same width.
    return {'ids': ids[0], 'tokens': tokens, 'positions': positions, 'combined': combined}


def main():
    args = common_parser(__doc__).parse_args()
    values = inspect(Session(), args.text)  # Create the shared untrained teaching model.
    for name, matrix in values.items():
        show(f'{name}: shape {tuple(matrix.shape)}', matrix[..., :8], args.step)
    print('Change --text; repeated token vectors match, but their position vectors differ.')
    print('These are untrained features, not a map of learned semantic meaning.')


if __name__ == '__main__':
    main()
