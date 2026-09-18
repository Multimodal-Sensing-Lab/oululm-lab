"""Offline CPU checks for learning mechanics, leakage, masks and adapter isolation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from llm_workshop.data import handbook, record
from llm_workshop.io import assert_disjoint, read_jsonl, write_jsonl
from llm_workshop.tiny import TinyGPT, TinyGPTConfig, get_batch, train as train_tiny, load_checkpoint
from llm_workshop.hf import AnswerCollator, tokenize_rows
from llm_workshop.evaluate import score, Retriever


def local_hf_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    vocab = ['<unk>', '<pad>', '<|im_start|>', '<|im_end|>', 'user', 'assistant',
             'Question', 'Answer', 'positive', 'neutral', 'negative'] + [str(i) for i in range(30)]
    backend = Tokenizer(models.WordLevel(dict(zip(vocab, range(len(vocab)))), unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>',
                                       eos_token='<|im_end|>', additional_special_tokens=['<|im_start|>'])
    tokenizer.chat_template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                        bos_token_id=None, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                        tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config)
    if path:
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    return model, tokenizer


class DataTests(unittest.TestCase):
    def test_handbook_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, repeat = [Path(tmp)/name for name in ('a', 'b', 'repeat')]
            handbook(a, assignment='a')
            handbook(b, assignment='b')
            handbook(repeat, assignment='a')
            self.assertEqual((a/'train.jsonl').read_bytes(), (repeat/'train.jsonl').read_bytes())
            self.assertNotEqual((a/'train.jsonl').read_bytes(), (b/'train.jsonl').read_bytes())
            splits = {s: read_jsonl(a/f'{s}.jsonl') for s in ('train','validation','test')}
            assert_disjoint(splits)
            self.assertEqual(len(splits['train']), 240)
            self.assertTrue(set(r['group_id'] for r in read_jsonl(a/'test_paraphrase.jsonl')) <=
                            set(r['group_id'] for r in splits['train']))
            with self.assertRaises(ValueError):
                assert_disjoint({'train': splits['train'], 'test': read_jsonl(a/'test_paraphrase.jsonl')})
            with self.assertRaises(ValueError):
                handbook(a)

    def test_invalid_labels_stay_in_denominator(self):
        rows = [{'task':'finance', 'reference':r, 'prediction':p} for r,p in
                [('positive','positive'),('neutral','neutral'),('negative','It is negative')]]
        result = score(rows)
        self.assertEqual(result['accuracy'], 2/3)
        self.assertEqual(result['invalid_output_rate'], 1/3)
        self.assertEqual(result['macro_f1'], 2/3)

    def test_retrieval_finds_entity(self):
        docs = [{'id':'a','text':'Neral instrument is in K123'}, {'id':'b','text':'Talum instrument is in K125'}]
        self.assertEqual(Retriever(docs).retrieve('Where is Talum?')[0]['id'], 'b')


class TinyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_causality_and_overfit(self):
        model = TinyGPT(TinyGPTConfig(12, 8, 1, 2, 16, 0.))
        tokens = torch.randint(12, (2, 9))
        x, y = tokens[:, :-1], tokens[:, 1:]
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 12
        model.eval()
        torch.testing.assert_close(model(x)[0][:,:4], model(changed)[0][:,:4])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.03)
        initial = model(x, y)[1].item()
        for _ in range(50):
            optimizer.zero_grad()
            loss = model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * .15)
        with self.assertRaises(ValueError):
            get_batch(torch.arange(8), 1, 8, 'cpu')

    def test_exact_steps_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, text in [('train', 'Once upon a time a robot found a box. '*10),
                                ('validation', 'Once a robot ran to a friend. '*10)]:
                write_jsonl(root/f'{split}.jsonl', [{'id':split, 'text':text}])
            cfg = dict(train_file=str(root/'train.jsonl'), validation_file=str(root/'validation.jsonl'),
                       output_dir=str(root/'full'), device='cpu', seed=13, batch_size=2, block_size=16,
                       max_steps=6, eval_interval=2, learning_rate=.002, n_layer=1,n_head=2,n_embd=16,
                       dropout=.1, generate_prompt='Once', max_new_tokens=1)
            train_tiny(cfg)
            full, full_state = load_checkpoint(root/'full/checkpoint.pt')
            partial = {**cfg, 'max_steps':3, 'output_dir':str(root/'partial')}
            train_tiny(partial)
            partial['max_steps'] = 6
            train_tiny(partial, root/'partial/checkpoint.pt')
            resumed, state = load_checkpoint(root/'partial/checkpoint.pt')
            self.assertEqual(state['step'], 6)
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            ids = torch.tensor([[1,2,3]])
            torch.testing.assert_close(full(ids)[0], resumed(ids)[0])


class HFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)

    def test_answer_mask_padding_and_overlength(self):
        model, tokenizer = local_hf_model(None)
        rows = [record('1','1','Question 1','positive','finance'),
                record('2','2','Question 2 3 4','negative','finance')]
        encoded, stats = tokenize_rows(rows, tokenizer, 64)
        labels = encoded[0]['labels']
        self.assertTrue(any(i == -100 for i in labels))
        self.assertEqual(labels[-1], tokenizer.eos_token_id)
        batch = AnswerCollator(tokenizer.eos_token_id)(encoded)
        self.assertEqual(batch['labels'][0,-1], -100)
        self.assertIn(tokenizer.eos_token_id, batch['labels'][0].tolist())
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            tokenize_rows(rows, tokenizer, 2)

    def test_lora_only_updates_and_disable_restores(self):
        from peft import get_peft_model, LoraConfig
        model, tokenizer = local_hf_model(None)
        ids = torch.tensor([[4, 6, 11, 5, 8, 3]])
        model.eval()
        with torch.no_grad():
            baseline = model(ids).logits.clone()
        originals = {name: p.clone() for name,p in model.named_parameters()}
        model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4,
                                               target_modules=['q_proj','v_proj']))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=.1)
        for _ in range(3):
            optimizer.zero_grad()
            model(ids, labels=ids).loss.backward()
            optimizer.step()
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            restored = model(ids).logits
        torch.testing.assert_close(baseline, restored, atol=0, rtol=0)
        with torch.no_grad():
            self.assertGreater((model(ids).logits-restored).abs().max().item(), 1e-5)

    def test_offline_trainer_and_adapter_reload(self):
        from llm_workshop.finetune import train
        from peft import PeftModel
        from llm_workshop.hf import load_base
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_hf_model(root/'base')
            for split, indices in [('train', range(8)), ('validation', range(8,10))]:
                write_jsonl(root/f'{split}.jsonl', [record(str(i),str(i),f'Question {i}',
                              ['positive','neutral','negative'][i%3],'finance') for i in indices])
            cfg = dict(model_name_or_path=str(root/'base'), train_file=str(root/'train.jsonl'),
                       validation_file=str(root/'validation.jsonl'), output_dir=str(root/'run'),device='cpu',
                       method='lora',max_steps=2,max_seq_length=64,gradient_accumulation_steps=1,
                       per_device_train_batch_size=2,gradient_checkpointing=False,eval_steps=1,save_steps=1,
                       logging_steps=1,lora_r=2,lora_alpha=4,num_threads=2)
            trainer = train(cfg)
            self.assertEqual(trainer.state.global_step, 2)
            self.assertTrue((root/'run/final/adapter_model.safetensors').exists())
            model, tokenizer = load_base(root/'base')
            model = PeftModel.from_pretrained(model, root/'run/final')
            self.assertTrue(torch.isfinite(model(torch.tensor([[4,6,11]])).logits).all())
            from argparse import Namespace
            from llm_workshop.evaluate import run
            args = Namespace(base_model=str(root/'base'), revision=None, device='cpu', fp16=False,
                             bf16=False, load_in_4bit=False, test_file=root/'validation.jsonl',
                             prompts=None, adapter=[f"trained={root/'run/final'}"],
                             output=root/'predictions.jsonl', context_mode='none', documents=None,
                             top_k=1, num_threads=2, max_new_tokens=2, max_input_tokens=64)
            predictions, summary = run(args)
            self.assertEqual(len(predictions), 4)
            self.assertEqual(set(p['system'] for p in predictions), {'base', 'trained'})
            self.assertIn('trained/finance/held_out', summary['scores'])
            # New Transformers protects optimizer deserialization on torch < 2.6.
            if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2,6):
                resumed = train(cfg, root/'run/checkpoint-1')
                self.assertEqual(resumed.state.global_step, 2)
            else:
                with self.assertRaisesRegex(ValueError, 'PyTorch >= 2.6'):
                    train(cfg, root/'run/checkpoint-1')


if __name__ == '__main__':
    unittest.main()
