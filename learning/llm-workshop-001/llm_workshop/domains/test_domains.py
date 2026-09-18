"""Validate saved experiments against their data and reloaded model predictions."""
import json
import unittest
from pathlib import Path
import torch
from llm_workshop.io import read_jsonl, sha256
from .data import Tokenizer, read_csv, split_rows
from .train import ROOT, batch, load, measure, generate


class DomainRecordsTest(unittest.TestCase):
    def test_split_and_frozen_vocabulary(self):
        splits = {d: split_rows(read_csv(ROOT/'data/external'/f'{name}_sentences.csv'))
                  for d, name in [('biology','animal_biology'),('finance','finance')]}
        for d, parts in splits.items():
            self.assertEqual(len(parts['validation']), 500 if d=='biology' else 200)
            a={r['sentence'].lower() for r in parts['train']}
            b={r['sentence'].lower() for r in parts['validation']}
            self.assertFalse(a & b)
            self.assertEqual(parts, split_rows(read_csv(ROOT/'data/external'/('animal_biology_sentences.csv' if d=='biology' else 'finance_sentences.csv'))))
        tok=Tokenizer.fit('word', (r['sentence'] for parts in splits.values() for r in parts['train']))
        self.assertEqual(tok.pieces('EUR 12.50!'), ['eur','1','2','.','5','0','!'])
        self.assertEqual(tok.encode('newunknownword',False), [2])
        self.assertEqual(len(tok.vocabulary),2567)
        x,y=batch(splits['biology']['train'][:2],tok,'cpu',64)
        self.assertTrue(torch.all(x[:,0]==0))
        for i,row in enumerate(splits['biology']['train'][:2]):
            ids=tok.encode(row['sentence'])
            self.assertEqual(x[i,:len(ids)-1].tolist(),ids[:-1])
            self.assertEqual(y[i,:len(ids)-1].tolist(),ids[1:])
            self.assertTrue(torch.all(y[i,len(ids)-1:]==-100))
        with self.assertRaises(ValueError):batch(splits['biology']['train'][:2],tok,'cpu',2)

    def test_recorded_checkpoints_and_metrics(self):
        torch.set_num_threads(2)
        for kind,folder in [('word','domains_words_20260917_01'),('character','domains_characters_20260917_01')]:
            root=ROOT/'runs'/folder
            report=json.loads((root/'report.json').read_text())
            for domain,source in report['manifest']['sources'].items():
                self.assertEqual(sha256(Path(source['path'])),source['sha256'])
            base,tokenizer,_=load(root/'biology_base.pt')
            self.assertEqual(tokenizer.kind,kind)
            self.assertEqual(sum(p.numel() for p in base.parameters()),report['parameter_count'])
            for stage in report['stages']:
                model,tok,_=load(root/stage['checkpoint'])
                self.assertEqual(tok.vocabulary,tokenizer.vocabulary)
                for domain in ['biology','finance']:
                    values=read_jsonl(root/f'{domain}_validation.jsonl')
                    loss=measure(model,values,tok,'cpu')
                    self.assertAlmostEqual(loss,stage['history'][-1][domain+'_validation'],places=4)
                # Compare a non-tied trained continuation after checkpoint reload.
                self.assertEqual(generate(model,tok,stage['samples'][0]['prompt']),stage['samples'][0]['text'])
                for suffix in ['png','svg']:
                    self.assertGreater((root/f"{stage['name']}_progress.{suffix}").stat().st_size,1000)
            only,replay=report['stages'][1:]
            self.assertEqual(only['main_domain_exposures'],11400)
            self.assertEqual(replay['main_domain_exposures'],11400)
            self.assertEqual(replay['biology_replay_exposures'],2850)
            for key in ['biology_validation','finance_validation']:
                self.assertEqual(only['history'][0][key],replay['history'][0][key])
                self.assertEqual(only['history'][0][key],report['stages'][0]['history'][-1][key])


if __name__=='__main__':unittest.main()
