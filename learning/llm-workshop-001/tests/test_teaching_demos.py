"""Behavior checks for the CPU teaching computations; no desktop display required."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from llm_workshop.teaching import Session, topic


class TeachingChecks(unittest.TestCase):
    def test_embeddings_repeat_but_positions_differ(self):
        result = topic(2).inspect(Session(), 'aaa')
        torch.testing.assert_close(result['tokens'][0], result['tokens'][1])
        self.assertFalse(torch.equal(result['combined'][0], result['combined'][1]))

    def test_causality_and_explicit_block_agreement(self):
        session = Session()
        for block in (0, 1):
            first = topic(3).inspect(session, 'the cat', block, 1)
            altered = topic(3).inspect(session, 'the dog', block, 1)
            torch.testing.assert_close(first['output'][:4], altered['output'][:4])
            self.assertEqual(int(torch.triu(first['attention'], 1).count_nonzero()), 0)
            torch.testing.assert_close(first['attention'].sum(-1), torch.ones(7))

    def test_training_checkpoint_comparison(self):
        session = Session()
        run = topic(4).TrainingRun(session)
        initial = copy.deepcopy(run.before.state_dict())
        for _ in range(30):
            run.advance()
        self.assertLess(run.measure()['train'], run.history[0]['train'])
        for key in initial:
            torch.testing.assert_close(run.before.state_dict()[key], initial[key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            output = run.save(Path(folder)/'run')
            restored = Session()
            restored.load(output/'after.pt')
            results = topic(8).compare(session, restored)
            self.assertEqual(results[0]['output'], results[1]['output'])

    def test_lora_preserves_base_and_merged_output(self):
        session = Session()
        run = topic(6).AdapterRun(session, 'finance', rank=2)
        before = run.measure()['adaptation_loss']
        for _ in range(30):
            run.advance()
        run.assert_base_unchanged()
        self.assertLess(run.measure()['adaptation_loss'], before)
        self.assertGreater(float(run.model.lm_head.b.abs().sum()), 0)
        self.assertEqual([n for n, p in run.model.named_parameters() if p.requires_grad], ['lm_head.a', 'lm_head.b'])
        with tempfile.TemporaryDirectory() as folder:
            output = run.save(Path(folder)/'adapter')
            merged = Session()
            merged.load(output/'merged.pt')
            ids = session.ids('the ')
            with torch.inference_mode():
                torch.testing.assert_close(merged.model(ids)[0], run.model(ids)[0], rtol=1e-4, atol=1e-5)

    def test_generation_rng_and_reserved_token_roundtrip(self):
        session = Session()
        first = topic(5).Generation(session, 'the ', 42)
        second = topic(5).Generation(session, 'the ', 42)
        for _ in range(10):
            self.assertEqual(first.advance()['chosen'], second.advance()['chosen'])
        # Force a reserved ID into context; it must stay an ID, never be re-tokenized as '<UNK>'.
        first.ids.append(0)
        self.assertIn('<UNK>', first.advance()['text'])

    def test_retrieval_updates_evidence_not_model(self):
        module = topic(7)
        old = module.retrieve(module.QUESTION)
        new = module.retrieve(module.QUESTION, module.DOCUMENTS.replace('K142', 'M219'))
        self.assertIn(0, old['selected'])
        self.assertIn('M219', new['prompt'])
        self.assertNotIn('K142', new['prompt'])
        empty = module.retrieve('xyzzy')
        self.assertEqual(empty['selected'], [])
        self.assertIn('No matching passage', empty['prompt'])

    def test_invalid_outputs_count_as_errors(self):
        metrics = topic(8).metrics()
        self.assertAlmostEqual(metrics['accuracy'], 4/6)
        self.assertAlmostEqual(metrics['invalid_output_rate'], 1/6)
        with self.assertRaises(ValueError):
            topic(8).metrics(['positive'])


if __name__ == '__main__':
    unittest.main()
