"""Greedy baseline/adapter evaluation, optionally using oracle or retrieved context."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

from .io import messages_for, normalized, read_jsonl, sha256, write_json, write_jsonl

LABELS = {'finance': ['positive', 'neutral', 'negative'], 'biology': ['yes', 'no', 'maybe']}


def score(rows):
    if not rows:
        raise ValueError('Cannot score zero predictions')
    correct = sum(normalized(r['prediction']) == normalized(r['reference']) for r in rows)
    result = {'n': len(rows), 'exact_match': correct / len(rows),
              'abstention_rate': sum(normalized(r['prediction']) == 'unknown' for r in rows) / len(rows),
              'wrong_non_abstaining_rate': sum(normalized(r['prediction']) not in
                  {normalized(r['reference']), 'unknown'} for r in rows) / len(rows)}
    tasks = {r['task'] for r in rows}
    if len(tasks) == 1 and next(iter(tasks)) in LABELS:
        labels = LABELS[next(iter(tasks))]
        confusion = {label: {p: 0 for p in labels + ['INVALID']} for label in labels}
        for row in rows:
            truth, predicted = normalized(row['reference']), normalized(row['prediction'])
            if truth not in labels:
                raise ValueError(f'Unknown reference label: {truth}')
            confusion[truth][predicted if predicted in labels else 'INVALID'] += 1
        f1s = []
        for label in labels:
            tp = confusion[label][label]
            fp = sum(confusion[other][label] for other in labels if other != label)
            fn = sum(confusion[label].values()) - tp
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        result.update(accuracy=result['exact_match'], macro_f1=sum(f1s) / len(f1s), confusion_matrix=confusion,
                      invalid_output_rate=sum(confusion[label]['INVALID'] for label in labels) / len(rows))
    if any('retrieved_ids' in row for row in rows):
        result['retrieval_recall'] = sum(row['source_id'] in row.get('retrieved_ids', []) for row in rows) / len(rows)
    return result


class Retriever:
    def __init__(self, documents):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([d['text'] for d in documents])

    def retrieve(self, question, k=1):
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        indices = sorted(range(len(scores)), key=lambda i: (-scores[i], self.documents[i]['id']))[:k]
        return [self.documents[i] for i in indices if scores[i] > 0]


def add_context(messages, passages):
    # Documents are evidence supplied to the user message, never a replacement system instruction.
    result = [dict(m) for m in messages]
    evidence = '\n\n'.join(f"[{p['id']}] {p['text']}" for p in passages)
    result[-1]['content'] += '\n\nUse the following current source passages. If unsupported, answer UNKNOWN.\n' + evidence
    return result


def run(args):
    import torch
    from .hf import generate, load_base
    torch.set_num_threads(args.num_threads)
    rows = read_jsonl(args.test_file) if args.test_file else [
        {'id': str(i), 'task': 'qualitative', 'messages': [{'role': 'user', 'content': line.strip()},
         {'role': 'assistant', 'content': '(unscored)'}]}
        for i, line in enumerate(Path(args.prompts).read_text().splitlines()) if line.strip()]
    if not rows:
        raise ValueError('No evaluation prompts')
    documents = read_jsonl(args.documents) if args.documents else []
    if args.context_mode != 'none' and not documents:
        raise ValueError('--documents is required for oracle/retrieve conditions')
    by_id = {d['id']: d for d in documents}
    retriever = Retriever(documents) if args.context_mode == 'retrieve' else None
    model, tokenizer = load_base(args.base_model, args.device, args.fp16, args.bf16, args.load_in_4bit, args.revision)
    model.eval()
    adapters = []
    for spec in args.adapter:
        name, separator, path = spec.partition('=')
        if not separator or not name or name == 'base' or name in [n for n, _ in adapters]:
            raise ValueError('Adapters need unique NAME=PATH specs; base is reserved')
        adapters.append((name, path))
    probe = tokenizer.apply_chat_template(messages_for(rows[0])[:-1], tokenize=True, return_dict=False, add_generation_prompt=True)
    probe_tensor = torch.tensor([probe[:32]], device=model.device)
    with torch.inference_mode():
        original = model(input_ids=probe_tensor).logits.detach().clone()
    if adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name)
        model.eval()
        with model.disable_adapter(), torch.inference_mode():
            restored = model(input_ids=probe_tensor).logits
        torch.testing.assert_close(original, restored)
    outputs = []
    for system in ['base'] + [name for name, _ in adapters]:
        if adapters and system != 'base':
            model.set_adapter(system)
        scope = model.disable_adapter() if adapters and system == 'base' else nullcontext()
        with scope:
            for row in rows:
                messages = messages_for(row)
                prompt = messages[:-1]
                passages = []
                if args.context_mode == 'oracle':
                    if row.get('source_id') not in by_id:
                        raise ValueError(f"Missing oracle document for {row['id']}")
                    passages = [by_id[row['source_id']]]
                elif retriever:
                    passages = retriever.retrieve(prompt[-1]['content'], args.top_k)
                if args.context_mode != 'none':
                    prompt = add_context(prompt, passages)
                prediction = generate(model, tokenizer, prompt, args.max_new_tokens, args.max_input_tokens)
                output = {'id': row['id'], 'system': system, 'task': row.get('task', 'custom'),
                          'test_group': row.get('test_group', 'held_out'), 'context_mode': args.context_mode,
                          'messages': prompt, 'reference': messages[-1]['content'], 'prediction': prediction}
                if args.context_mode != 'none':
                    output.update(source_id=row['source_id'], retrieved_ids=[p['id'] for p in passages])
                outputs.append(output)
                print(f"{system} | {row['id']} | {prediction}")
    write_jsonl(args.output, outputs)
    summary = {'base_model': args.base_model, 'revision': args.revision, 'device': args.device,
               'fp16': args.fp16, 'bf16': args.bf16, 'load_in_4bit': args.load_in_4bit,
               'do_sample': False, 'max_new_tokens': args.max_new_tokens, 'adapter_specs': args.adapter,
               'test_sha256': sha256(args.test_file or args.prompts),
               'documents_sha256': sha256(args.documents) if args.documents else None,
               'scores': {}}
    if args.test_file:
        groups = defaultdict(list)
        for row in outputs:
            groups[(row['system'], row['task'], row['test_group'])].append(row)
        summary['scores'] = {'/'.join(key): score(value) for key, value in groups.items()}
    write_json(str(args.output) + '.metrics.json', summary)
    return outputs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--revision')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--test-file', type=Path)
    source.add_argument('--prompts', type=Path, default=Path('examples/prompts/domain_comparison_prompts.txt'))
    parser.add_argument('--adapter', action='append', default=[], help='NAME=PATH, repeatable')
    parser.add_argument('--output', type=Path, default=Path('runs/comparison.jsonl'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--max-input-tokens', type=int, default=2048)
    parser.add_argument('--context-mode', choices=['none', 'oracle', 'retrieve'], default='none')
    parser.add_argument('--documents', type=Path)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--num-threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.top_k, args.max_new_tokens, args.max_input_tokens) < 1:
        parser.error('Token budgets and top-k must be positive')
    if args.output.exists() or Path(str(args.output) + '.metrics.json').exists():
        parser.error('Output exists; use a new output name')
    run(args)


if __name__ == '__main__':
    main()
