"""Follow Q/K/V, masking, attention, residuals and the feed-forward network."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import math
import torch
from llm_workshop.teaching import Session, common_parser, show


def inspect(session, text, block_index=0, head=0):
    model = session.model  # Reuse the session's TinyGPT and its weights.
    ids = session.ids(text)  # Topic 001: text -> IDs, shape [1, token count].
    if not 0 <= block_index < len(model.blocks) or not 0 <= head < model.cfg.n_head:
        raise ValueError('Select an existing block and attention head.')
    with torch.inference_mode():  # Inspect calculations without tracking gradients.
        # Topic 002: token vectors + position vectors -> the first block's input.
        hidden = model.token_embedding(ids) + model.position_embedding(torch.arange(ids.shape[1]))
        for preceding in model.blocks[:block_index]:
            hidden = preceding(hidden)  # Reach the selected block through earlier blocks.
        block = model.blocks[block_index]  # Inspect one transformer block.
        normalized = block.ln1(hidden)  # Normalize features within each token's row.
        width, length = hidden.shape[-1], hidden.shape[1]  # Features and token positions.
        # Three learned projections: query (seek), key (match), value (contribute).
        q, k, v = block.attn.qkv(normalized).split(width, dim=-1)
        # [batch, positions, features] -> [batch, heads, positions, head features].
        q, k, v = [t.view(1, length, model.cfg.n_head, -1).transpose(1, 2) for t in (q, k, v)]
        # Each query scores every key; divide by the square root of head width.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        mask = block.attn.mask[:, :, :length, :length].bool()  # Allow current and earlier tokens.
        masked = scores.masked_fill(~mask, float('-inf'))  # Block future positions.
        weights = masked.softmax(-1)  # Each reader's mixing weights sum to one.
        mixture = weights @ v  # Mix source value vectors, separately per head.
        # Join the heads back into one feature vector per position.
        concatenated = mixture.transpose(1, 2).contiguous().view(1, length, width)
        attention_output = block.attn.proj(concatenated)  # Learn how to combine head features.
        residual = hidden + attention_output  # First bypass: keep input, add attention update.
        feed_forward = block.mlp(block.ln2(residual))  # Normalize, then transform each row.
        output = residual + feed_forward  # Second bypass: add the feed-forward update.
        # This explicit computation matches the shared implementation in eval mode.
        torch.testing.assert_close(output, block(hidden))
    # Remove the one-prompt batch axis; select a head for its intermediate values.
    return {'Q': q[0, head], 'K': k[0, head], 'V': v[0, head],
            'scores': scores[0, head], 'mask': mask[0, 0], 'masked_scores': masked[0, head],
            'attention': weights[0, head], 'weighted_values': mixture[0, head],
            'input': hidden[0], 'after_attention_residual': residual[0],
            'feed_forward': feed_forward[0], 'output': output[0]}


def main():
    parser = common_parser(__doc__)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--head', type=int, default=0)
    args = parser.parse_args()
    result = inspect(Session(), args.text, args.block, args.head)  # Start with untrained weights.
    for name, matrix in result.items():
        show(f'{name}: {tuple(matrix.shape)}', matrix[:, :8], args.step)  # Show only the first 8 columns.
    # Topic 005 uses the full model, then its vocabulary scores, to generate tokens.
    print('Each attention row sums to one:', result['attention'].sum(-1))
    print('Future tokens are blocked; the current token is allowed. Weights are untrained.')


if __name__ == '__main__':
    main()
