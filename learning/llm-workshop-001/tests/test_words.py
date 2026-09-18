"""Behavior and artifact-contract checks for the independent word companion."""
import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from unittest.mock import patch
import torch
from llm_workshop.io import read_jsonl, assert_disjoint
from llm_workshop.tiny import TinyGPT, TinyGPTConfig, load_checkpoint as character_load
from llm_workshop.words.tokenizer import WordTokenizer
from llm_workshop.words.corpus import DATA, prepare
from llm_workshop.words.model import batch, measure, generate, save_checkpoint, load_checkpoint
from llm_workshop.words.train import train
from llm_workshop.words.adapt import adapt
from llm_workshop.words.demo import choose_interactively


class WordChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rows = {name: read_jsonl(DATA / f'{name}.jsonl') for name in ['train','validation','test']}
        self.tokenizer = WordTokenizer.fit(r['text'] for r in self.rows['train'])

    def test_corpus_splits_and_reproducible_generation(self):
        assert_disjoint(self.rows)
        self.assertEqual([len(r) for r in self.rows.values()], [432,72,72])
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'corpus'
            prepare(out)
            for split in self.rows:
                self.assertEqual((out/f'{split}.jsonl').read_bytes(),(DATA/f'{split}.jsonl').read_bytes())
                for row in self.rows[split]:
                    self.tokenizer.encode(row['text'])

    def test_word_normalization_ids_and_unknowns(self):
        t=self.tokenizer
        self.assertEqual(t.encode(' THE\tcat. '),t.encode('the cat.'))
        ids=t.encode('the cat cat.',boundaries=True)
        self.assertEqual(ids[2],ids[3])
        self.assertEqual(t.decode(ids),'the cat cat.')
        self.assertEqual(t.decode([0,2,1]),'<UNK>')
        for text in ['', 'the dragon', 'café', 'the 123', 'the <UNK>']:
            with self.assertRaises(ValueError):t.encode(text)
        with self.assertRaises(ValueError):t.prompt('cat '*16,16)

    def test_padding_does_not_change_real_token_loss(self):
        model=TinyGPT(TinyGPTConfig(len(self.tokenizer.vocabulary),block_size=16,n_layer=2,n_head=2,n_embd=32,dropout=0)).eval()
        row=self.rows['train'][0]
        x,y=batch([row],self.tokenizer,16,'cpu')
        ids=self.tokenizer.encode(row['text'],boundaries=True)
        with torch.inference_mode():
            expected=model(torch.tensor([ids[:-1]]),torch.tensor([ids[1:]]))[1]
            actual=model(x,y)[1]
        torch.testing.assert_close(actual,expected)
        self.assertEqual(int((y!=-100).sum()),len(ids)-1)

    def test_manual_word_changes_next_forward_context_and_prediction(self):
        t = self.tokenizer
        model = TinyGPT(TinyGPTConfig(len(t.vocabulary), block_size=16, n_layer=1,
                                     n_head=2, n_embd=16, dropout=0)).eval()
        original = copy.deepcopy(model.state_dict())
        seen = []

        def forward(x):
            seen.append(x[0].tolist())
            # A controlled predictor makes the effect of the chosen context explicit.
            next_word = 'in' if int(x[0, -1]) == t.stoi['plays'] else 'sleeps'
            logits = torch.full((1, x.shape[1], len(t.vocabulary)), -10.)
            logits[0, -1, t.stoi[next_word]] = 10.
            return logits, None

        output = io.StringIO()
        with patch.object(model, 'forward', side_effect=forward), \
                patch('builtins.input', side_effect=['plays', '', '/quit']), redirect_stdout(output):
            result = generate(model, t, 'the small cat', choose_token=partial(choose_interactively, t))
        self.assertEqual(result['text'], 'the small cat plays in')
        self.assertEqual(result['stop'], 'user stopped')
        self.assertEqual(seen[1], t.prompt('the small cat plays', 16))
        self.assertEqual(seen[2], t.prompt('the small cat plays in', 16))
        self.assertIn("Model suggestion: 'sleeps'", output.getvalue())
        self.assertIn("Manual token: 'plays'", output.getvalue())
        self.assertIn("Model token: 'in'", output.getvalue())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def test_interactive_retries_and_end_override(self):
        t = self.tokenizer
        model = TinyGPT(TinyGPTConfig(len(t.vocabulary), block_size=16, n_layer=1,
                                     n_head=2, n_embd=16, dropout=0)).eval()
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias.fill_(-100)
            model.lm_head.bias[1] = 100
        output = io.StringIO()
        with patch('builtins.input', side_effect=['dragon', 'plays in', '/words', 'PLAYS', '']), \
                redirect_stdout(output):
            result = generate(model, t, 'the small cat', choose_token=partial(choose_interactively, t))
        self.assertEqual(result['ids'], t.prompt('the small cat plays', 16) + [1])
        self.assertEqual(result['stop'], 'end token')
        self.assertEqual(len(result['events']), 2)
        self.assertEqual(output.getvalue().count('Sentence unchanged.'), 2)
        self.assertIn('Available input tokens:', output.getvalue())

    def test_accepted_suggestions_match_automatic_generation_and_limits(self):
        t = self.tokenizer
        model = TinyGPT(TinyGPTConfig(len(t.vocabulary), block_size=16, n_layer=1,
                                     n_head=2, n_embd=16, dropout=0)).eval()
        for greedy in [True, False]:
            automatic = generate(model, t, 'the small cat', greedy=greedy)
            accepted = generate(model, t, 'the small cat', greedy=greedy,
                                choose_token=lambda ids, probabilities, suggested: suggested)
            self.assertEqual(automatic, accepted)
        force_word = lambda ids, probabilities, suggested: t.stoi['cat']
        limited = generate(model, t, 'the small cat', tokens=2, choose_token=force_word)
        self.assertEqual(limited['stop'], 'token budget')
        self.assertEqual(len(limited['events']), 2)
        full = generate(model, t, 'cat ' * 14, choose_token=force_word)
        self.assertEqual(full['stop'], 'context full')
        self.assertEqual(len(full['events']), 1)
        stopped = generate(model, t, 'the small cat', choose_token=lambda *args: None)
        self.assertEqual(stopped['text'], 'the small cat')
        self.assertEqual(stopped['events'], [])

    def test_training_reload_generation_and_frozen_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'base'
            report=train(out,steps=50,device='cpu')
            self.assertLess(report['history'][-1]['validation'],report['history'][0]['validation'])
            model,t,state=load_checkpoint(out/'after.pt')
            original=copy.deepcopy(model.state_dict())
            self.assertEqual(generate(model,t,'the small cat'),generate(model,t,'the small cat'))
            # Force END as the next choice: exactly one new token and clean display.
            with torch.no_grad():model.lm_head.weight.zero_();model.lm_head.bias.fill_(-100);model.lm_head.bias[1]=100
            result=generate(model,t,'the small cat')
            self.assertEqual(result['stop'],'end token');self.assertEqual(result['text'],'the small cat')
            self.assertEqual(len(result['events']),1)
            model.load_state_dict(original)
            with self.assertRaisesRegex(ValueError,'word checkpoint'):character_load(out/'after.pt')
            report=adapt(out/'after.pt',Path(folder)/'adapter',steps=20,device='cpu')
            self.assertEqual(report['trainable_parameters'],4*(64+len(t.vocabulary)))
            restored,_,_=load_checkpoint(out/'after.pt')
            for key in original:torch.testing.assert_close(restored.state_dict()[key],original[key],rtol=0,atol=0)
            merged,_,_=load_checkpoint(Path(folder)/'adapter/merged.pt')
            for key in original:
                if key!='lm_head.weight':torch.testing.assert_close(merged.state_dict()[key],original[key],rtol=0,atol=0)
            self.assertFalse(torch.equal(merged.lm_head.weight,original['lm_head.weight']))


if __name__=='__main__':unittest.main()
