# Hinglish NLP Toolkit — Architecture & Technical Guide

> **Written in plain English.** This document explains how the entire project is built, how each piece connects, and why things work the way they do.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Bird's-Eye View of the Codebase](#2-birds-eye-view-of-the-codebase)
3. [The Five NLP Tasks](#3-the-five-nlp-tasks)
4. [How a Request Flows Through the System](#4-how-a-request-flows-through-the-system)
5. [The Three-Tier Reliability System](#5-the-three-tier-reliability-system)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Model Types & How They Work](#7-model-types--how-they-work)
8. [Training Workflow](#8-training-workflow)
9. [Evaluation System](#9-evaluation-system)
10. [Configuration System](#10-configuration-system)
11. [The Web Application](#11-the-web-application)
12. [Data & Datasets](#12-data--datasets)
13. [Dependency Map](#13-dependency-map)

---

## 1. What This Project Does

**Hinglish** (Hindi + English mixed together) is how hundreds of millions of Indians actually communicate — in WhatsApp messages, tweets, movie reviews, and chatbots. Standard NLP tools built for pure English or pure Hindi struggle with this mixed language.

This toolkit solves that. It provides **five NLP capabilities for Hinglish text**, all through one unified interface:

| Task | What It Does | Example |
|------|-------------|---------|
| **Translation** | Translates Hinglish ↔ English or Devanagari Hindi ↔ English | "Aaj mausam bahut acha hai" → "The weather is very good today" |
| **Sentiment Analysis** | Decides if text is Positive, Negative, or Neutral | "Yeh movie bohot bekar thi" → Negative |
| **Named Entity Recognition (NER)** | Finds names of people, places, and organizations | "Virat Kohli ne Mumbai me match khela" → PER: Virat Kohli, LOC: Mumbai |
| **Question Answering (QA)** | Reads a passage and answers a question from it | Given a cricket news excerpt, answers "Match kahan khela gaya?" |
| **Intent Detection** | Understands what the user wants to do | "Mujhe flight book karni hai" → book_flight |

---

## 2. Bird's-Eye View of the Codebase

```
hinglish-nlp-toolkit/
│
├── app/
│   └── streamlit_app.py        ← The interactive web demo (frontend)
│
├── config/
│   ├── base.yaml               ← Shared settings for everything
│   ├── sentiment.yaml          ← Overrides just for sentiment task
│   ├── ner.yaml                ← Overrides just for NER task
│   ├── qa.yaml                 ← Overrides just for QA task
│   ├── intent.yaml             ← Overrides just for intent task
│   └── translation.yaml        ← Overrides just for translation task
│
├── data/
│   ├── COMI-LINGUA/            ← Official benchmark dataset (CSV files)
│   ├── processed/              ← Cleaned, model-ready JSONL files
│   └── scripts/                ← Download & preprocessing helper scripts
│
├── models/                     ← Saved trained model checkpoints
│   ├── sentiment/
│   ├── ner/
│   ├── qa/
│   └── intent/
│
├── notebooks/                  ← Jupyter notebooks for experiments
│
├── scripts/
│   ├── train.py                ← CLI to fine-tune transformer models
│   ├── train_baselines.py      ← CLI to train TF-IDF / CRF baselines
│   └── evaluate.py             ← CLI to evaluate a trained model
│
├── src/                        ← All the core Python library code
│   ├── inference/
│   │   └── predictor.py        ← THE brain: UnifiedPredictor class
│   ├── models/
│   │   ├── sentiment_model.py  ← Transformer wrapper for sentiment
│   │   ├── ner_model.py        ← Transformer wrapper for NER
│   │   ├── qa_model.py         ← Transformer wrapper for QA
│   │   ├── intent_model.py     ← Transformer wrapper for intent
│   │   ├── translation_model.py← Seq2Seq wrapper for translation
│   │   └── baselines.py        ← TF-IDF + CRF classical models
│   ├── preprocessing/
│   │   ├── normalizer.py       ← Text cleaning pipeline
│   │   ├── tokenizer_utils.py  ← Tokenization helpers
│   │   └── augmentation.py     ← Data augmentation tools
│   ├── training/
│   │   └── trainer.py          ← Orchestrates training for any task
│   ├── evaluation/
│   │   ├── metrics.py          ← BLEU, F1, Exact Match, etc.
│   │   └── error_analysis.py   ← Categorizes errors by type
│   └── utils/
│       ├── config.py           ← YAML config loader with inheritance
│       └── lang_utils.py       ← Script detection (Devanagari vs Latin)
│
├── tests/                      ← Unit tests
└── requirements.txt            ← Python dependencies
```

**The Most Important File:** `src/inference/predictor.py` — the `UnifiedPredictor` class is the single entry point for all five tasks. Whether you call from Python or from the web app, everything goes through here.

---

## 3. The Five NLP Tasks

### 3.1 Translation

- Uses pre-trained **MarianMT models** from Helsinki-NLP:
  - `opus-mt-hi-en` for Hindi/Hinglish → English
  - `opus-mt-en-hi` for English → Hindi
- If the input has **Devanagari characters** (e.g. `मुझे`), it uses the neural Marian model.
- If the input is **Romanized Hinglish** (e.g. `"Mujhe jaana hai"`), it uses a word-map dictionary fallback.
- Translation models are loaded **lazily** (only when first needed) to save memory.

### 3.2 Sentiment Analysis

- Fine-tunes **MuRIL** (`google/muril-base-cased`) — a BERT-style model pre-trained on 17 Indian languages.
- Adds a classification head to predict 3 classes: **Positive**, **Negative**, **Neutral**.
- Trained on the `HingSent` dataset (Hinglish sentiment-labeled tweets).
- Fallback order: Transformer → TF-IDF → Keyword matching.

### 3.3 Named Entity Recognition (NER)

- Uses MuRIL in a **token classification** setup — every word in the sentence gets a label.
- Labels use the **BIO tagging scheme**: `B-PER`, `I-PER`, `B-LOC`, `O`, etc.
- Entity types: `PER` (person), `LOC` (location), `ORG` (organization), `MISC` (other).
- Trained on the **COMI-LINGUA** NER dataset.
- Fallback order: Transformer → CRF → Capitalization-based rules.

### 3.4 Extractive Question Answering

- Takes a **question** + a **context passage**, and finds the exact text span that answers it.
- MuRIL learns to predict the **start and end positions** of the answer in the context.
- If the transformer is uncertain (confidence < 20%), a rule-based fallback kicks in:
  - "Kahan/Where" → look for known city/location names
  - "Kaun/Who" → look for capitalized proper names
  - "Kitne/How many" → look for numbers
  - "Kab/When" → look for time patterns

### 3.5 Intent Detection

- Predicts which of **30 conversational intents** a message belongs to (e.g., `book_flight`, `get_weather`).
- Safety rules always run **first** — mental health distress, emergency, and threat keywords are flagged immediately before any model runs.
- Then a prioritized cascade runs:
  1. Transformer model
  2. TF-IDF baseline
  3. Rule-based keyword matching
  4. Zero-shot classification via `facebook/bart-large-mnli` (last resort)

---

## 4. How a Request Flows Through the System

Step-by-step trace for: `predictor.predict("Virat Kohli ne Mumbai me match khela", task="ner")`

```
Step 1: NORMALIZATION
   CodeMixedNormalizer cleans the text
   → strips URLs, @mentions, emojis, elongation, spelling variants

Step 2: SCRIPT DETECTION
   detect_script() counts Devanagari vs Latin characters
   → dominant_script: "latin", is_mixed_script: False

Step 3: MODEL LOADING (lazy)
   → Looks in models/ner/config.json
   → Found → loads NERModel from disk
   → Not found → tries CRF → tries rule-based heuristics

Step 4: INFERENCE
   → NERModel tokenizes text into subwords
   → MuRIL assigns a BIO tag to each token
   → Subword tags merged back to word-level spans

Step 5: RESULT
   {
     "task": "ner",
     "entities": [
       {"entity": "PER", "word": "Virat Kohli", "score": 0.95},
       {"entity": "LOC", "word": "Mumbai", "score": 0.91}
     ],
     "dominant_script": "latin",
     "model_type": "transformer"
   }
```

---

## 5. The Three-Tier Reliability System

The system **never crashes** because of a missing model — it always degrades gracefully.

```
┌─────────────────────────────────────────────────┐
│  TIER 1: Deep Learning                          │
│  Fine-tuned MuRIL Transformer (best accuracy)  │
│  Loaded from: models/{task}/config.json         │
└───────────────────────┬─────────────────────────┘
                        │ model not found / error
                        ▼
┌─────────────────────────────────────────────────┐
│  TIER 2: Classical ML                           │
│  TF-IDF + Logistic Regression (classification) │
│  CRF Sequence Tagger (NER)                      │
│  Loaded from: models/{task}/baseline_*.joblib   │
└───────────────────────┬─────────────────────────┘
                        │ baseline not found / error
                        ▼
┌─────────────────────────────────────────────────┐
│  TIER 3: Heuristics / Rules                     │
│  Keyword matching (sentiment, intent)           │
│  Regex + entity lists (NER, QA)                 │
│  Word-map translation (translation)             │
│  Zero-shot LLM (intent only, last resort)       │
└─────────────────────────────────────────────────┘
```

The `model_type` field in every response tells you which tier was actually used.

---

## 6. Preprocessing Pipeline

Every input text passes through `CodeMixedNormalizer` before reaching any model:

```
Raw Input
  1. Unicode Normalization (NFC)    — standardizes multi-byte characters
  2. URL Removal                    — https://t.co/xyz → (removed)
  3. Username Removal               — @viratKohli → (removed)
  4. Hashtag Processing             — #IPL2024 → "IPL2024"
  5. Emoji Removal (optional)       — 😍 → (removed if configured)
  6. Elongation Reduction           — "sooooo" → "sooo" (max 3 repeats)
  7. Punctuation Normalization      — "WOW!!!" → "WOW!"
  8. Romanized Hindi Normalization  — "bohot" → "bahut", "nhi" → "nahi"
  9. Whitespace Cleanup             — collapse extra spaces, strip edges
Clean Output
```

### Script Detection

After normalizing, the system checks the Unicode ranges of every character:
- Devanagari: U+0900–U+097F  
- Latin (English): U+0041–U+007A

The **dominant script** is whichever type has more characters. If >10% of characters come from a second script → `is_mixed_script = True`. This is shown in the web app and attached to every API response.

---

## 7. Model Types & How They Work

### 7.1 MuRIL — The Primary Backbone

MuRIL (**Multilingual Representations for Indian Languages**) is pre-trained by Google on 17 Indian languages plus their transliterated (Romanized) forms. Because it saw Romanized Hindi during pre-training, it understands Hinglish naturally.

**For classification (Sentiment, Intent):**
```
Text → Tokenizer (subwords) → MuRIL Encoder → [CLS] vector → Linear Head → Softmax → Label + Confidence
```

**For token classification (NER):**
```
Text → Tokenizer → MuRIL Encoder → Per-token vectors → Linear Head → BIO tags per token
```

**For extractive QA:**
```
[Question] + [SEP] + [Context] → MuRIL → Two linear heads → Start position + End position → context[start:end]
```

### 7.2 MarianMT — For Translation

A **sequence-to-sequence** encoder-decoder transformer. Unlike BERT, it *generates* new text:
```
Source Text → Encoder → Hidden States → Decoder (word-by-word) → Translated Text
```

### 7.3 TF-IDF + Classifier (Classical Baseline)

- TF-IDF converts text into character n-gram features (chunks of 1–3 characters). Character n-grams handle Hinglish spelling variation better than word n-grams.
- Features are fed into **Logistic Regression** or **Linear SVM**.

### 7.4 CRF (Classical NER Baseline)

For each word, hand-crafted features are extracted (prefix/suffix, capitalization, is digit, has Devanagari, neighboring words). A Conditional Random Field then labels the full sequence, respecting tag transition constraints (e.g., `I-PER` can't follow `B-LOC`).

### 7.5 Zero-Shot Classifier (Intent Fallback)

`facebook/bart-large-mnli` (a Natural Language Inference model) decides whether the text "entails" each possible intent label — no task-specific training required. Used only when all other tiers fail.

---

## 8. Training Workflow

```
Raw Dataset (COMI-LINGUA / HuggingFace Hub)
        ↓
  data/scripts/preprocess.py
        ↓
  data/processed/{task}/train.jsonl + val.jsonl
        ↓
  python scripts/train.py --task sentiment --backbone google/muril-base-cased
        ↓
  src/training/trainer.py   (loads config, data, dispatches to model)
        ↓
  src/models/{task}_model.py  (tokenizes, runs HuggingFace Trainer API)
        ↓
  models/{task}/  (model.safetensors, config.json, tokenizer.json, label_map.json)
```

### Default Training Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `learning_rate` | 2e-5 | Step size for weight updates |
| `batch_size` | 16 | Examples processed at once |
| `num_epochs` | 10 | Full passes through training data |
| `weight_decay` | 0.01 | Prevents overfitting |
| `warmup_ratio` | 0.1 | Gradually increase LR at start |
| `fp16` | true (GPU) | Half-precision for speed |
| `early_stopping` | 3 epochs | Stop if no improvement |

### Data Augmentation (for low-resource tasks)

Since Hinglish training data is scarce:
1. **Back-Translation**: English → Hindi → English, creating natural paraphrases with Hindi words mixed in.
2. **Synthetic Code-Mixing**: Randomly replaces ~30% of English words with Romanized Hindi equivalents using a hand-built dictionary (`"good"` → `"acha"`, `"today"` → `"aaj"`).

---

## 9. Evaluation System

| Task | Metrics | Notes |
|------|---------|-------|
| Sentiment / Intent | Accuracy, Macro F1 | Macro averages each class equally |
| NER | Entity-level Precision / Recall / F1 (seqeval) | Only exact span + type matches count |
| QA | Exact Match (EM), Token F1 | EM=1 only if answer is character-perfect |
| Translation | BLEU, ROUGE-1/2/L | Standard MT evaluation metrics |

**Error Analysis** (`error_analysis.py`) categorizes *why* mistakes happen:
- Language-switch boundary errors (fails right where Hindi ↔ English switches)
- Script confusion (Devanagari vs Romanized text)
- Entity boundary errors (right type, wrong span)
- Out-of-vocabulary (unseen words during training)

---

## 10. Configuration System

Uses **YAML inheritance** — task configs only override what they need to change:

```
config/base.yaml          ← Global defaults (paths, training params, models)
       +
config/sentiment.yaml     ← Only: num_labels=3, backbone, batch_size=32
       =
Final merged config       ← Complete config with all values filled in
```

`config.py` uses a recursive `deep_merge()` so nested dicts are merged key-by-key, not replaced wholesale.

```python
cfg = load_config("sentiment")                         # base + sentiment overrides
cfg = load_config()                                    # base only
cfg = load_config("ner", overrides={"training": {"num_epochs": 5}})  # + code overrides
```

---

## 11. The Web Application

`app/streamlit_app.py` is a Streamlit app. Streamlit reruns Python on every user interaction and renders results as a web page.

```
User types text in browser
        ↓
Streamlit reruns script
        ↓
@st.cache_resource  ← UnifiedPredictor loaded ONCE per session (not per click)
        ↓
predictor.predict(text, task)
        ↓
Result dict → rendered as cards, entity highlights, confidence bars
```

**Key features:**
- **Task Selector** (sidebar): Switch between the 5 tasks
- **Linguistic Telemetry**: Shows dominant script, is-mixed status, code-mix %
- **QA Benchmark Presets**: Pre-loaded question+context pairs for demos
- **Span Highlighting**: Extracted QA answer is highlighted in the context
- **NER Visualization**: Entities shown with color-coded inline tags
- **Caching**: `@st.cache_resource` prevents transformer models from reloading on every button click

---

## 12. Data & Datasets

### COMI-LINGUA (Primary Benchmark)

Stored in `data/COMI-LINGUA/`:

| File | Task | Content |
|------|------|---------|
| `NER_train/test.csv` | NER | Tokens labeled PERSON, LOCATION, ORGANISATION, DATE |
| `MT_train/test.csv` | Translation | Parallel Hinglish / English / Devanagari Hindi |
| `MLI_train.csv` | Multi-lingual inference | Sentence-pair entailment labels |
| `POS_train.csv` | Part-of-speech | Word-level POS tags |

### External Datasets (HuggingFace Hub)

| Dataset | HF ID | Used For |
|---------|-------|----------|
| HingSent | `l3cube-pune/HingSent` | Sentiment training |
| IndicQA | `ai4bharat/IndicQA` | QA training |
| SQuAD v2 | `squad_v2` | QA transfer learning |
| ATIS | `atis` | Intent base (English) |

---

## 13. Dependency Map

### How the Modules Connect

```
app/streamlit_app.py
    └── src/inference/predictor.py  (UnifiedPredictor)
            ├── src/preprocessing/normalizer.py    (always runs first)
            ├── src/utils/lang_utils.py            (script detection)
            ├── src/utils/config.py                (load model paths)
            └── src/models/*.py                    (loaded lazily per task)

scripts/train.py
    └── src/training/trainer.py
            ├── src/utils/config.py
            └── src/models/{task}_model.py

src/evaluation/
    ├── metrics.py        (sacrebleu, rouge_score, seqeval, sklearn)
    └── error_analysis.py (src/utils/lang_utils.py)
```

### Key Libraries

| Library | Role |
|---------|------|
| `transformers` | Loads/runs MuRIL, MarianMT, BART |
| `torch` | Deep learning engine |
| `datasets` | HuggingFace dataset loading |
| `scikit-learn` | TF-IDF, Logistic Regression, SVM, metrics |
| `python-crfsuite` | CRF model for NER baseline |
| `seqeval` | Entity-level NER evaluation |
| `sacrebleu` | Standardized BLEU for translation |
| `streamlit` | Web app framework |
| `pyyaml` | Config file parsing |

---

## Summary: The Full Picture

```
User Input (text + task choice)
    │
    ▼
UnifiedPredictor.predict()
    │
    ├─► Normalize text (CodeMixedNormalizer)
    ├─► Detect script (Devanagari / Latin / Mixed)
    ├─► Load best available model (Tier 1 → 2 → 3)
    │
    └─► Task branch:
         Translation  → MarianMT neural translation / word-map fallback
         Sentiment    → MuRIL classifier / TF-IDF / keyword rules
         NER          → MuRIL token tagger / CRF / capitalization rules
         QA           → MuRIL span extractor / regex heuristics
         Intent       → Safety rules → Transformer → TF-IDF → Rules → Zero-shot
    │
    ▼
Result dict: {task, answer/label/entities, confidence, model_type, script_info}
    │
    ▼
Streamlit web app  OR  Python API caller
```

---
*Generated from full codebase review — Oct 2026*
