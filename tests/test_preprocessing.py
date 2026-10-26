"""
Unit tests for preprocessing and language utilities.
"""

import unittest
from src.preprocessing.normalizer import CodeMixedNormalizer
from src.preprocessing.augmentation import SyntheticCodeMixer
from src.utils.lang_utils import detect_script, detect_language_mix, is_devanagari, is_romanized


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.normalizer = CodeMixedNormalizer(lang_pair="hi-en")

    def test_unicode_normalization(self):
        text = "क\u093e" # Devanagari character combination
        norm = self.normalizer.normalize_unicode(text)
        self.assertIsNotNone(norm)

    def test_url_and_username_cleaning(self):
        text = "Check this link https://t.co/abc1234 @username for details"
        clean = self.normalizer.normalize(text)
        self.assertNotIn("https://", clean)
        self.assertNotIn("@username", clean)

    def test_elongation_reduction(self):
        text = "Yeh bohot acchaaaaa movie hai"
        clean = self.normalizer.reduce_elongation(text)
        self.assertIn("acchaaa", clean)  # max elongation is 3 by default

    def test_romanized_hindi_normalization(self):
        text = "bohot accha hai"
        clean = self.normalizer.normalize_romanized(text)
        self.assertEqual(clean, "bahut acha hai")

    def test_script_detection(self):
        info_dev = detect_script("यह एक हिंदी वाक्य है")
        self.assertEqual(info_dev.dominant_script, "devanagari")

        info_rom = detect_script("Yeh ek English sentence hai")
        self.assertEqual(info_rom.dominant_script, "latin")

        info_mix = detect_script("Yeh एक बहुत सुंदर code-mixed sentence hai")
        self.assertTrue(info_mix.is_mixed_script)

    def test_synthetic_code_mixer(self):
        mixer = SyntheticCodeMixer(replacement_prob=1.0, seed=42)
        text = "What is good today"
        mixed = mixer.code_mix(text)
        self.assertNotEqual(text, mixed)


if __name__ == "__main__":
    unittest.main()
