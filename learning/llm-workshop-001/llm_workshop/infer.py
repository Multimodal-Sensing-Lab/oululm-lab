"""Generate locally from a tiny checkpoint or a Hugging Face model and adapter."""
import argparse
import torch
from .hf import generate, load_base
from .tiny import encode, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tiny-checkpoint')
    source.add_argument('--base-model')
    parser.add_argument('--adapter')
    parser.add_argument('--revision')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0., help='Tiny model only; 0 is greedy')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.tiny_checkpoint:
        model, state = load_checkpoint(args.tiny_checkpoint, args.device)
        ids = torch.tensor([encode(args.prompt, state['stoi'], strict=True)], device=args.device)
        output = model.generate(ids, args.max_new_tokens, args.temperature)[0].tolist()
        print(''.join(state['itos'][i] for i in output))
    else:
        if args.temperature != 0:
            parser.error('HF inference is greedy; temperature applies only to the tiny model')
        model, tokenizer = load_base(args.base_model, args.device, args.fp16, args.bf16, args.load_in_4bit, args.revision)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        print(generate(model, tokenizer, [{'role': 'user', 'content': args.prompt}], args.max_new_tokens))


if __name__ == '__main__':
    main()
