"""Single-device SFT or continued pretraining with full weights, LoRA, or QLoRA.

Token masking is explicit in hf.py. Transformers Trainer handles optimization,
accumulation and resumable checkpoints; TRL is not needed for this first path.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
from .hf import AnswerCollator, load_base, tokenize_rows
from .io import assert_disjoint, fresh_directory, load_config, read_jsonl, sha256, write_json

CONFIG_KEYS = '''domain model_name_or_path model_revision train_file validation_file output_dir device
seed max_seq_length num_train_epochs max_steps learning_rate per_device_train_batch_size
per_device_eval_batch_size gradient_accumulation_steps load_in_4bit fp16 bf16 lora_r lora_alpha
lora_dropout target_modules logging_steps eval_steps save_steps gradient_checkpointing report_to
method objective num_threads warmup_ratio save_total_limit'''.split()


class FiniteLoss(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        import math
        for key in ('loss', 'eval_loss', 'grad_norm'):
            if key in (logs or {}) and not math.isfinite(logs[key]):
                raise FloatingPointError(f'Nonfinite {key} at step {state.global_step}')


def train(cfg, resume=None):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This teaching trainer supports one process/device; DDP is a later exercise')
    device = cfg.get('device', 'cpu')
    torch.set_num_threads(int(cfg.get('num_threads', 2)))
    seed = int(cfg.get('seed', 42))
    set_seed(seed)
    method = cfg.get('method', 'qlora' if cfg.get('load_in_4bit', False) else 'lora')
    if method not in {'lora', 'qlora', 'full'}:
        raise ValueError('method must be lora, qlora, or full')
    quantized = method == 'qlora'
    if 'load_in_4bit' in cfg and bool(cfg['load_in_4bit']) != quantized:
        raise ValueError('method and load_in_4bit disagree')
    fp16, bf16 = bool(cfg.get('fp16', False)), bool(cfg.get('bf16', False))
    if method == 'full' and fp16:
        raise ValueError('Use FP32 full tuning in this teaching path; FP16 base parameters need a different master-weight setup')
    rows = {name: read_jsonl(cfg[key]) for name, key in
            [('train', 'train_file'), ('validation', 'validation_file')]}
    assert_disjoint(rows)
    hashes = {key: sha256(cfg[key]) for key in ('train_file', 'validation_file')}
    output = Path(cfg['output_dir'])
    if resume:
        if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2, 6):
            raise ValueError('Transformers optimizer resume requires PyTorch >= 2.6; upgrade the isolated environment or resume on Roihu')
        manifest_path = Path(resume).parent / 'run_manifest.json'
        if not manifest_path.exists():
            raise ValueError('Resume checkpoint must belong to a workshop run with run_manifest.json')
        previous = json.loads(manifest_path.read_text())
        if previous['data_hashes'] != hashes or previous['config'] != cfg:
            raise ValueError('Resume requires identical configuration and data; use the original planned max_steps')
    else:
        fresh_directory(output)
    model, tokenizer = load_base(cfg['model_name_or_path'], device, fp16, bf16, quantized, cfg.get('model_revision'))
    checkpointing = bool(cfg.get('gradient_checkpointing', True))
    model.config.use_cache = False
    if method != 'full':
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if quantized:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing,
                                                    gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(
            r=int(cfg.get('lora_r', 8)), lora_alpha=int(cfg.get('lora_alpha', 16)),
            lora_dropout=float(cfg.get('lora_dropout', .05)), bias='none', task_type='CAUSAL_LM',
            target_modules=cfg.get('target_modules', ['q_proj', 'v_proj'])))
        model.print_trainable_parameters()
    tokenized, stats = {}, {}
    for split in rows:
        tokenized[split], stats[split] = tokenize_rows(rows[split], tokenizer,
            int(cfg.get('max_seq_length', 512)), cfg.get('objective', 'sft'))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if method != 'full' and any('lora_' not in name for name in trainable):
        raise ValueError('Unexpected trainable base weights')
    versions = {package: importlib.metadata.version(package) for package in ('torch', 'transformers', 'peft', 'accelerate')}
    manifest = {'config': cfg, 'data_hashes': hashes, 'tokenization': stats, 'versions': versions,
                'platform': platform.platform(), 'device': device,
                'model_revision_resolved': getattr(model.config, '_commit_hash', None),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'trainable_modules': trainable, 'source_hashes': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    write_json(output / 'run_manifest.json', manifest)
    # Inspectable tokens, labels, and mask for the first supervised example.
    first = tokenized['train'][0]
    write_json(output / 'first_example_tokens.json', [
        {'token': tokenizer.convert_ids_to_tokens(i), 'id': i, 'label': label}
        for i, label in zip(first['input_ids'], first['labels'])])
    steps = int(cfg.get('max_steps', -1))
    effective_batch = int(cfg.get('per_device_train_batch_size', 1)) * int(cfg.get('gradient_accumulation_steps', 8))
    if effective_batch < 1:
        raise ValueError('Batch size and gradient accumulation must be positive')
    total_updates = steps if steps > 0 else math.ceil(
        math.ceil(len(tokenized['train']) / effective_batch) * float(cfg.get('num_train_epochs', 1)))
    args = TrainingArguments(
        output_dir=str(output), use_cpu=not str(device).startswith('cuda'), seed=seed, data_seed=seed,
        max_steps=steps, num_train_epochs=float(cfg.get('num_train_epochs', 1)),
        learning_rate=float(cfg.get('learning_rate', 2e-4)),
        per_device_train_batch_size=int(cfg.get('per_device_train_batch_size', 1)),
        per_device_eval_batch_size=int(cfg.get('per_device_eval_batch_size', 1)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 8)),
        fp16=fp16, bf16=bf16, gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=int(cfg.get('logging_steps', 5)), logging_nan_inf_filter=False,
        eval_strategy='steps', eval_steps=int(cfg.get('eval_steps', 25)),
        save_strategy='steps', save_steps=int(cfg.get('save_steps', 25)),
        save_total_limit=int(cfg.get('save_total_limit', 2)),
        warmup_steps=math.ceil(total_updates * float(cfg.get('warmup_ratio', .05))), report_to=cfg.get('report_to', 'none'),
        dataloader_num_workers=0, dataloader_pin_memory=str(device).startswith('cuda'),
        optim='adamw_torch', remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'],
                      eval_dataset=tokenized['validation'], data_collator=AnswerCollator(tokenizer.pad_token_id),
                      processing_class=tokenizer, callbacks=[FiniteLoss()])
    before = trainer.evaluate()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    after = trainer.evaluate()
    # Separate reloadable adapter/full model from Trainer's resumable checkpoint directories.
    trainer.save_model(str(output / 'final'))
    tokenizer.save_pretrained(output / 'final')
    trainer.save_state()
    summary = {'validation_before': before, 'validation_after': after, 'training': result.metrics,
               'global_step': trainer.state.global_step,
               'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
               'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else None}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path, help='Trainer checkpoint directory, not final adapter')
    parser.add_argument('--model', help='Local model snapshot or Hub ID override')
    parser.add_argument('--output-dir')
    parser.add_argument('--train-file')
    parser.add_argument('--validation-file')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    cfg = load_config(args.config, CONFIG_KEYS)
    for argument, key in [('model','model_name_or_path'), ('output_dir','output_dir'),
                          ('train_file','train_file'), ('validation_file','validation_file'),
                          ('max_steps','max_steps'), ('device','device')]:
        value = getattr(args, argument)
        if value is not None:
            cfg[key] = value
    if args.device == 'cpu':
        cfg.update(fp16=False, bf16=False)
    train(cfg, args.resume)


if __name__ == '__main__':
    main()
