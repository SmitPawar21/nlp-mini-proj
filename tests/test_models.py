"""
Unit tests for baseline models and predictor interface.
"""

import unittest
from src.models.baselines import TFIDFClassifier
from src.inference.predictor import UnifiedPredictor


class TestModels(unittest.TestCase):

    def test_tfidf_baseline(self):
        train_texts = ["Acha hai", "Bohot bura hai", "Neutral review"]
        train_labels = ["positive", "negative", "neutral"]
        
        clf = TFIDFClassifier(classifier_type="lr")
        clf.fit(train_texts, train_labels)
        
        preds = clf.predict(["Acha movie"])
        self.assertEqual(len(preds), 1)

    def test_unified_predictor_fallback(self):
        predictor = UnifiedPredictor()
        res = predictor.predict("Yeh bohot acchi film hai", task="sentiment")
        self.assertIn("sentiment", res)
        self.assertIn("confidence", res)

        res_trans = predictor.predict("Party ka plan banao", task="translation")
        self.assertIn("translation", res_trans)

        res_intent = predictor.predict("Flight ticket book karni hai", task="intent")
        self.assertIn("intent", res_intent)


if __name__ == "__main__":
    unittest.main()
