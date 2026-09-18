"""Generate a real answer using a staged, local Hugging Face instruction model."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# This explicit local path stays offline. Stage the model with scripts/download_model.py first.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from importlib import import_module


def main():
    import argparse
    from llm_workshop.hf import load_base, generate
    demo = import_module('llm_workshop.007_retrieval_augmented_generation.demo')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--question', default=demo.QUESTION)
    parser.add_argument('--documents', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if not (args.model/'config.json').exists():
        parser.error('--model must be a staged model directory containing config.json')
    result = demo.retrieve(args.question, args.documents.read_text() if args.documents else demo.DOCUMENTS)
    print('Retrieved prompt:', result['prompt'])
    model, tokenizer = load_base(str(args.model), args.device, args.device == 'cuda', False, False)
    for name, prompt in [('without retrieval', args.question), ('with retrieval', result['prompt'])]:
        print(name, generate(model, tokenizer, [{'role':'user', 'content':prompt}], 80, 512))


if __name__ == '__main__':
    main()
