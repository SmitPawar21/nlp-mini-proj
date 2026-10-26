# Hinglish NLP Toolkit: Code-Mixed Language Understanding

A unified deep learning and multilingual NLP framework designed specifically for **Hindi-English Code-Mixed (Hinglish)** text. 

Built on **MuRIL** (`google/muril-base-cased`) and **MarianMT** (`Helsinki-NLP`), the toolkit provides a single programmatic interface (`UnifiedPredictor`) and an interactive, clean **Streamlit web application** covering 5 essential NLP tasks:

1. **Machine Translation**: Bidirectional translation (Hinglish $\leftrightarrow$ English, Devanagari Hindi $\rightarrow$ English).
2. **Sentiment Analysis**: 3-class classification (*Positive*, *Negative*, *Neutral*).
3. **Named Entity Recognition (NER)**: Subword BIO token tagging for *Person (PER)*, *Location (LOC)*, *Organization (ORG)*, and *Miscellaneous (MISC)*.
4. **Question Answering (Extractive QA)**: Reading comprehension and exact span extraction from reference contexts.
5. **Intent Detection**: Conversational intent recognition across 9 practical domains (*flight booking*, *weather*, *balance check*, *food orders*, *alarms*, etc.).

---

## ⚡ Quickstart: Running on Your PC

### 1. Prerequisites
* **Python**: 3.8+ (Python 3.10 or 3.11 recommended).
* **Hardware**: Runs on standard **CPU**; accelerated automatically if an **NVIDIA GPU** with CUDA is available.
* **Git**: Installed on your system.

### 2. Clone Repository & Setup Virtual Environment

#### Windows (PowerShell)
```powershell
# Clone the repository
git clone https://github.com/SameedMulla/hinglish-nlp-toolkit.git
cd hinglish-nlp-toolkit

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

*(If PowerShell execution policy prevents script activation, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

#### Linux / macOS / Bash
```bash
# Clone the repository
git clone https://github.com/SameedMulla/hinglish-nlp-toolkit.git
cd hinglish-nlp-toolkit

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU Acceleration (Optional)**: If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🖥️ Launching the Web Application

The interactive web demo provides an interface with real-time script dominance detection, benchmark presets, custom input testing, and span highlighting:

```bash
streamlit run app/streamlit_app.py
```

Once started, open your browser at **`http://localhost:8501`**.

### Application Features:
* **Clean White Aesthetic**: Minimalist layout with neutral slate typography.
* **Synchronized QA Benchmarks**: Pre-configured question and context passage pairs with span highlighting.
* **Full Custom Input Support**: Test any custom Hinglish text or reading passage without preset overwrites.
* **Linguistic Telemetry**: Real-time display of dominant script (Latin vs. Devanagari), script mixing status, and code-mix percentage.

---

## 🐍 Python API Usage (`UnifiedPredictor`)

You can directly import and run the `UnifiedPredictor` in your own Python scripts:

```python
import sys, os
sys.path.insert(0, os.path.abspath("."))

from src.inference.predictor import UnifiedPredictor

# Initialize predictor (automatically detects GPU/CUDA if available)
predictor = UnifiedPredictor(lang_pair="hi-en")
```

### 1. Question Answering (QA)
```python
context = "IPL ka grand final match Mumbai ke Wankhede stadium me khela gaya tha."
question = "Match kahan khela gaya tha?"

res = predictor.predict(text=question, task="qa", context=context)
print(f"Answer: {res['answer']}")        # Output: Mumbai
print(f"Confidence: {res['confidence']}")  # Output: ~0.95
```

### 2. Sentiment Analysis
```python
text = "Yeh movie bohot acchi thi mujhe direction bohot pasand aaya!"
res = predictor.predict(text=text, task="sentiment")

print(f"Sentiment: {res['sentiment']}")    # Output: positive
print(f"Confidence: {res['confidence']}")  # Output: 0.85+
```

### 3. Intent Detection
```python
text = "Mujhe kal Delhi ke liye flight ticket book karni hai."
res = predictor.predict(text=text, task="intent")

print(f"Intent: {res['intent']}")          # Output: book_flight
print(f"Candidates: {res['top_k']}")      # Top-3 intent distribution
```

### 4. Named Entity Recognition (NER)
```python
text = "Virat Kohli ne Mumbai me BCCI officials se mulaqat ki."
res = predictor.predict(text=text, task="ner")

for ent in res['entities']:
    print(f"{ent['word']} -> {ent['entity']} ({ent['score']:.1%})")
# Output:
# Virat Kohli -> PER
# Mumbai -> LOC
# BCCI -> ORG
```

### 5. Bidirectional Translation
```python
# Code-Mixed (Hinglish) -> English
res_en = predictor.predict(
    text="Aaj shaam ko hum sab friends milke party karenge.",
    task="translation",
    translation_direction="cm_to_en"
)
print(res_en['translation'])
# Output: This evening all of us friends will party together.

# English -> Code-Mixed Hindi
res_hi = predictor.predict(
    text="The weather in Delhi is very pleasant today.",
    task="translation",
    translation_direction="en_to_cm"
)
print(res_hi['translation'])
```

---

## 📊 Dataset: COMI-LINGUA Benchmark

The toolkit integrates with the official **COMI-LINGUA** benchmark dataset located in `data/COMI-LINGUA/`:
* **`NER_train.csv` / `NER_test.csv`**: Annotated code-mixed entity tokens across `PERSON`, `LOCATION`, `ORGANISATION`, and `DATE`.
* **`MT_train.csv` / `MT_test.csv`**: Parallel code-mixed Hinglish, standard English, and Devanagari Hindi text.
* **`MLI_train.csv` / `POS_train.csv`**: Multi-lingual inference and part-of-speech annotations.

---

## 🏋️ Training & Fine-Tuning

### Fine-Tuning MuRIL on Tasks
To train or fine-tune models on specific tasks:

```bash
# Sentiment Analysis
python scripts/train.py --task sentiment --backbone google/muril-base-cased

# Intent Detection
python scripts/train.py --task intent --backbone google/muril-base-cased

# Named Entity Recognition
python scripts/train.py --task ner --backbone google/muril-base-cased

# Extractive Question Answering
python scripts/train.py --task qa --backbone google/muril-base-cased
```

Checkpoints are serialized to `models/{task}/` (`model.safetensors`, `config.json`, `label_map.json`, `tokenizer.json`).

### Training Classical Baselines
To train or update the lightweight fallback models (TF-IDF & CRF):
```bash
python scripts/train_baselines.py
```

---

## 🛡️ Three-Tier Reliability Architecture

In production and demo environments, the `UnifiedPredictor` dynamically chooses the best available engine:
1. **Tier 1 (Deep Learning)**: Fine-tuned MuRIL transformer model with GPU/CPU acceleration.
2. **Tier 2 (Machine Learning Baselines)**: Serialized TF-IDF Classifiers (`baseline_{task}.joblib`) or CRF Sequence Taggers (`baseline_crf_{task}.model`).
3. **Tier 3 (Domain-Adapted Heuristics)**: Question-type aware extractive heuristics and lexical mapping rules ensuring robust answers even on edge cases or offline environments.

---

## 📁 Repository Structure

```
hinglish-nlp-toolkit/
├── app/
│   └── streamlit_app.py        # Streamlit interactive web application
├── config/
│   ├── base.yaml               # Global model and path configurations
│   └── tasks.yaml              # Per-task hyperparameter definitions
├── data/
│   ├── COMI-LINGUA/            # Official COMI-LINGUA benchmark CSVs
│   ├── processed/              # Preprocessed JSONL training/validation splits
│   └── scripts/                # Data downloading and preprocessing scripts
├── models/
│   ├── intent/                 # Trained Intent model checkpoints & label maps
│   ├── ner/                    # Trained NER model checkpoints
│   ├── qa/                     # Trained Extractive QA checkpoints
│   └── sentiment/              # Trained Sentiment model checkpoints
├── scripts/
│   ├── train.py                # Transformer training CLI
│   ├── train_baselines.py      # TF-IDF / CRF baseline training CLI
│   └── evaluate.py             # Model evaluation CLI
├── src/
│   ├── inference/
│   │   └── predictor.py        # UnifiedPredictor core engine
│   ├── models/                 # Model wrappers (SentimentModel, QAModel, etc.)
│   ├── preprocessing/          # Normalizers, tokenizers, augmentations
│   └── utils/                  # Script detection, logging, configuration loader
├── requirements.txt            # Python dependencies
└── technical_summary.md        # In-depth technical architecture document
```

---

## ❓ Frequently Asked Questions (FAQ)

**Q: `ModuleNotFoundError: No module named 'src'` when running scripts?**  
**A:** Run commands from the project root directory. On Windows, you can set `$env:PYTHONPATH="."` (or in Linux/macOS `export PYTHONPATH="."`).

**Q: How do I force CPU inference instead of GPU?**  
**A:** Initialize the predictor with `UnifiedPredictor(device="cpu")`.

**Q: What languages are supported?**  
**A:** The primary focus is `hi-en` (Hindi-English Hinglish). Code-base architecture is modular and extendable to other Indic pairs like `ta-en` (Tamil-English).
