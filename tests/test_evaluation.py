"""
Unit tests for evaluation metrics and error analysis.
"""

import unittest
from src.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_classification_metrics,
    compute_qa_metrics,
)
from src.evaluation.error_analysis import analyse_errors


class TestEvaluation(unittest.TestCase):

    def test_bleu_and_rouge(self):
        preds = ["yeh ek accha din hai"]
        refs = ["yeh ek accha din hai"]
        bleu = compute_bleu(preds, refs)
        self.assertGreater(bleu["bleu"], 90.0)

        rouge = compute_rouge(preds, refs)
        self.assertEqual(rouge["rougeL"], 100.0)

    def test_classification_metrics(self):
        preds = [0, 1, 2, 1]
        refs = [0, 1, 2, 0]
        res = compute_classification_metrics(preds, refs)
        self.assertEqual(res["accuracy"], 0.75)
        self.assertIn("f1", res)

    def test_qa_metrics(self):
        preds = ["Mumbai"]
        refs = ["Mumbai"]
        qa_res = compute_qa_metrics(preds, refs)
        self.assertEqual(qa_res["exact_match"], 100.0)
        self.assertEqual(qa_res["f1"], 100.0)

    def test_error_analysis(self):
        preds = ["positive", "negative"]
        refs = ["positive", "neutral"]
        texts = ["Acha hai", "Thik thak tha"]
        rep = analyse_errors("sentiment", preds, refs, texts)
        self.assertEqual(rep.total_errors, 1)


if __name__ == "__main__":
    unittest.main()
