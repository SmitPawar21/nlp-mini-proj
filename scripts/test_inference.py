import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import UnifiedPredictor

def run_tests():
    predictor = UnifiedPredictor()
    
    print("--- 1. Translation: Code-Mixed -> English ---")
    res = predictor.predict("Aaj shaam ko hum sab friends milke party karenge.", task="translation", translation_direction="cm_to_en")
    print("TGT:", res["translation"])
    print("Model:", res["model_type"])

    print("\n--- 2. Translation: English -> Code-Mixed Hindi ---")
    res = predictor.predict("The weather in Delhi is very pleasant today.", task="translation", translation_direction="en_to_cm")
    print("TGT:", res["translation"])
    print("Model:", res["model_type"])

    print("\n--- 3. Sentiment Analysis ---")
    res = predictor.predict("Yeh movie bohot acchi thi, casting aur direction mind blowing tha!", task="sentiment")
    print("Sentiment:", res["sentiment"])
    print("Confidence:", res["confidence"])
    print("Model:", res["model_type"])

    print("\n--- 4. NER ---")
    res = predictor.predict("Shah Rukh Khan Mumbai me Mannat me rehte hain.", task="ner")
    print("Entities:", res["entities"])
    print("Model:", res["model_type"])

    print("\n--- 5. QA ---")
    ctx = "Shah Rukh Khan Indian film actor hain. Woh Mumbai me Mannat house me rehte hain."
    res = predictor.predict("Shah Rukh Khan kahan rehte hain?", task="qa", context=ctx)
    print("Answer:", res["answer"])
    print("Confidence:", res["confidence"])
    print("Model:", res["model_type"])

    print("\n--- 6. Intent ---")
    res = predictor.predict("Mujhe Delhi se Mumbai ki flight ticket book karni hai.", task="intent")
    print("Intent:", res["intent"])
    print("Confidence:", res["confidence"])
    print("Model:", res["model_type"])

if __name__ == "__main__":
    run_tests()
