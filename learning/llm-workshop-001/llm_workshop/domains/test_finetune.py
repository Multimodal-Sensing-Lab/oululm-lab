"""Check preparation boundaries and the provenance/numerics of the actual recordings."""
import json
import unittest
import torch
from .prepare import prepare, formatted, ROOT
from .data import Tokenizer
from .finetune import encode_rows, measure, predict, prompt_for
from .train import load
from llm_workshop.io import sha256


class FineTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data, cls.sources = prepare()

    def test_splits_vocabulary_and_response_targets(self):
        for kind,context,size in [('word',256,3348),('character',1280,61)]:
            tok=Tokenizer.fit(kind,(r['sentence'] for p in self.data.values() for r in p['train']))
            self.assertEqual(len(tok.vocabulary),size)
            for domain,parts in self.data.items():
                normalize=lambda r:' '.join(r['sentence'].lower().split())
                self.assertFalse(set(map(normalize,parts['train'])) & set(map(normalize,parts['validation'])))
                for split,rows in parts.items():
                    for r,(x,y) in zip(rows,encode_rows(rows,tok,context)):
                        prefix,response=formatted(r)
                        expected=tok.encode(response,False)+[1]
                        self.assertEqual([i for i in y if i!=-100],expected)
                        self.assertEqual(len(x),len(y))
                        self.assertLessEqual(len(x),context)
                        if prefix:
                            n=len(tok.encode(prefix,False))
                            self.assertEqual(y[:n],[-100]*n)
                            self.assertEqual(x[0],0)
                            self.assertEqual(x[1:n+1],tok.encode(prefix,False))
            # A novel piece maps to UNK; fitting never silently expands the IDs.
            self.assertIn(2,tok.encode('🦄neveraworkshopword'))
        email=self.data['email']['train'][0]
        self.assertEqual(email['sentence'].count('### email:'),1)
        self.assertNotIn('<|endoftext|>',email['sentence'])

    def test_recorded_lineage_losses_and_generated_cat_answer(self):
        catalog=json.loads((ROOT/'llm_workshop/browser/domain-catalog.json').read_text())
        for item in catalog['runs']:
            folder=ROOT/item['path'];report=json.loads((folder/'report.json').read_text())
            base=report['stages'][0]
            self.assertEqual(base['steps'],1000)
            for stage in report['stages']:
                path=folder/stage['checkpoint']
                self.assertEqual(sha256(path),stage['sha256'])
                if stage['name']!='base':
                    self.assertEqual(stage['steps'],500)
                    self.assertEqual(stage['base_sha256'],base['sha256'])
                    self.assertAlmostEqual(stage['history'][0]['base_validation'],base['history'][-1]['base_validation'],places=6)
                model,tok,state=load(path)
                self.assertIn('email',state['tokenizer']['fit'])
                # Recompute the complete held-out result from saved weights on CPU.
                rows=encode_rows(self.data[stage['name']]['validation'],tok,model.cfg.block_size)
                actual=measure(model,rows,'cpu',stage['batch_size'])
                self.assertAlmostEqual(actual,stage['history'][-1]['validation'],delta=2e-4)
                if stage['name']=='qa':
                    result=predict(model,tok,prompt_for('qa','what is a cat?'),120)
                    self.assertEqual(result['text'],'a cat is a small mammal. it has fur and is often kept as a pet.')
                    self.assertEqual(result['stop'],'end token')
                    with self.assertRaisesRegex(ValueError,'Outside this checkpoint vocabulary'):
                        predict(model,tok,'🦄',1)


if __name__=='__main__':unittest.main()
