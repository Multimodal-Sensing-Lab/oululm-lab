"""Second live demo: trace one UNTRAINED TinyGPT forward pass.

This file computes and prints the forward pass.
Run: python llm_workshop/001_text_to_predictions/lesson.py --step
First run tokenizer_demo.py so students understand the integer input.
"""
import argparse
import math
from pathlib import Path
import sys

# Shared model implementation; every layer is readable in llm_workshop/tiny.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from llm_workshop.tiny import TinyGPT, TinyGPTConfig
from importlib import import_module

tokenizer = import_module('llm_workshop.001_text_to_predictions.tokenizer_demo')

# Change ONE at a time. Keep architecture fixed for the first demonstrations.
DEMO_TEXT = 'the cat sat.'
RANDOM_SEED = 1337
SAMPLE_TEMPERATURE = 0.8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default=DEMO_TEXT)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--temperature', type=float, default=SAMPLE_TEMPERATURE)
    parser.add_argument('--step', action='store_true')
    args = parser.parse_args()
    if not 2 <= len(args.text) <= 64:
        parser.error('Use 2–64 characters for this model demonstration.')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('Use a finite positive sampling temperature.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)

    # 1. PREPARE: the same tokenizer, now feeding a neural network.
    vocabulary, token_to_id = tokenizer.build_vocabulary(tokenizer.read_corpus(tokenizer.CORPUS))
    ids = tokenizer.encode(args.text, token_to_id)
    x = torch.tensor([ids[:-1]])  # [one sequence, input positions]
    y = torch.tensor([ids[1:]])   # observed NEXT token at each position
    print('1. TOKENIZE AND SHIFT')
    print('Text:', repr(args.text), '\nIDs:', ids)
    print('Input x:', x.tolist(), '\nTarget y:', y.tolist())
    tokenizer.pause(args.step)

    # 2. INITIALIZE: shape choices are human-designed; parameters start untrained.
    cfg = TinyGPTConfig(len(vocabulary), block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=.1)
    model = TinyGPT(cfg).cpu().eval()  # eval disables dropout; it does not erase weights.
    print('\n2. ARCHITECTURE (not yet trained)\n', model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters()))
    tokenizer.pause(args.step)

    # inference_mode avoids storing a backward graph. There is no optimizer here.
    with torch.inference_mode():
        # 3. EMBEDDINGS: ID lookup, plus a vector for each position.
        positions = torch.arange(x.shape[1])
        token_vectors = model.token_embedding(x)
        position_vectors = model.position_embedding(positions)
        hidden = token_vectors + position_vectors
        print('\n3. EMBEDDINGS')
        print('Token vectors shape:', tuple(token_vectors.shape))
        print('First token vector, first 8 values:', token_vectors[0, 0, :8].tolist())
        print('First position vector, first 8 values:', position_vectors[0, :8].tolist())
        print('Their sum, first 8 values:', hidden[0, 0, :8].tolist())
        tokenizer.pause(args.step)

        # 4. BLOCKS: attention mixes positions, then a feed-forward network changes features.
        for number, block in enumerate(model.blocks, start=1):
            hidden = block(hidden)
            print(f'\n4.{number}. TRANSFORMER BLOCK {number}: shape {tuple(hidden.shape)}')
            print('First position, first 8 features:', hidden[0, 0, :8].tolist())
            tokenizer.pause(args.step)

        # 5. OUTPUT: one vocabulary-sized score vector per position.
        logits = model.lm_head(model.ln_f(hidden))
        probabilities = torch.softmax(logits, dim=-1)
        print('\n5. NEXT-TOKEN DISTRIBUTION: shape', tuple(probabilities.shape))
        values, indices = probabilities[0, -1].topk(5)
        for probability, token_id in zip(values.tolist(), indices.tolist()):
            print(f'  {vocabulary[token_id]!r:8s}: {probability:.4f}')
        print('Distribution sum:', probabilities[0, -1].sum().item())
        tokenizer.pause(args.step)

        # 6. SCORE: use the probability assigned to each observed target.
        # log_softmax gives log probabilities without numerical underflow.
        log_probs = torch.log_softmax(logits, dim=-1)
        losses = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        print('\n6. LOSS = mean of -ln(probability of each observed target)')
        for i, penalty in enumerate(losses[0].tolist()):
            print(f'  prefix {args.text[:i+1]!r} -> target {args.text[i+1]!r}: penalty {penalty:.4f}')
        print('Mean loss:', losses.mean().item())
        print('Uniform reference ln(vocabulary size):', math.log(len(vocabulary)))
        # Verify our explicit path is the same computation as model.forward().
        reference_logits, reference_loss = model(x, y)
        torch.testing.assert_close(logits, reference_logits)
        torch.testing.assert_close(losses.mean(), reference_loss)
        tokenizer.pause(args.step)

        # 7. GENERATE: repeatedly append one sampled ID; there are no new weight updates.
        generated = model.generate(x, max_new_tokens=40, temperature=args.temperature)
        print('\n7. UNTRAINED SAMPLE:', repr(tokenizer.decode(generated[0].tolist(), vocabulary)))
        print('Zero training updates. Changing temperature changes sampling, not knowledge.')


if __name__ == '__main__':
    main()
