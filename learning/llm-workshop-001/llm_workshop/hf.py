"""Shared model loading and explicit last-answer tokenization."""
from __future__ import annotations

import torch
from .io import messages_for


def resolve_dtype(device, fp16=False, bf16=False):
    if fp16 and bf16:
        raise ValueError('Choose at most one of fp16 and bf16')
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        if bf16 and torch.cuda.get_device_capability()[0] < 8:
            raise ValueError('This GPU has no native BF16 support; use FP16')
    elif fp16 or bf16:
        raise ValueError('Use float32 for the CPU teaching path')
    return torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32


def load_base(name, device='cpu', fp16=False, bf16=False, quantized=False, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dtype = resolve_dtype(device, fp16, bf16)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer needs an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    kwargs = dict(revision=revision, dtype=dtype, trust_remote_code=False, attn_implementation='eager')
    if quantized:
        if not str(device).startswith('cuda'):
            raise ValueError('This workshop QLoRA path requires CUDA')
        kwargs.update(device_map={'': device}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not quantized:
        model.to(device)
    return model, tokenizer


def tokenize_rows(rows, tokenizer, max_length, objective='sft'):
    if max_length < 2 or objective not in {'sft', 'continued_pretraining'}:
        raise ValueError('Invalid token budget or training objective')
    encoded, dropped = [], []
    for index, row in enumerate(rows):
        if objective == 'continued_pretraining':
            text = row['text']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Expected nonempty text for continued pretraining')
            ids = tokenizer(text, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            labels = ids.copy()
        else:
            messages = messages_for(row)
            prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=False)
            if ids[:len(prefix)] != prefix:
                raise ValueError('Chat template is not prefix-aligned; inspect tokens before training')
            labels = [-100] * len(prefix) + ids[len(prefix):]
        # Drop the entire example instead of quietly truncating its answer.
        if len(ids) > max_length:
            dropped.append(row.get('id', str(index)))
            continue
        if len(ids) < 2 or not any(label != -100 for label in labels[1:]):
            raise ValueError('Example contains no next-token supervision')
        encoded.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels})
    if not encoded:
        raise ValueError(f'All {len(rows)} records exceed max_seq_length or are invalid')
    return encoded, {'input_records': len(rows), 'kept_records': len(encoded), 'dropped_overlength_ids': dropped,
                     'supervised_tokens': sum(sum(x != -100 for x in r['labels'][1:]) for r in encoded)}


class AnswerCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        length = max(len(row['input_ids']) for row in rows)
        # Mask padding by length, NOT by token ID (EOS can also be the pad token).
        return {key: torch.tensor([row[key] + [padding] * (length - len(row[key])) for row in rows])
                for key, padding in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]}


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens=32, max_input_tokens=2048):
    model.eval()
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True)
    if not ids or len(ids) > max_input_tokens:
        raise ValueError(f'Prompt has {len(ids)} tokens; budget is {max_input_tokens}. No silent truncation.')
    model_limit = getattr(model.config, 'max_position_embeddings', None)
    if model_limit and len(ids) + max_new_tokens > model_limit:
        raise ValueError('Prompt plus completion exceeds model context')
    inputs = torch.tensor([ids], device=model.device)
    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, len(ids):], skip_special_tokens=True).strip()
