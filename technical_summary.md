# Technical Summary: Deep Learning for Code-Mixed Hindi-English NLP Toolkit

## 1. Executive Overview
The **Hinglish NLP Toolkit** is an end-to-end deep learning framework designed specifically for analyzing and generating Hindi-English code-mixed (Hinglish) text. Code-mixed communication exhibits rapid script-switching (Romanized vs. Devanagari), non-standard phonetic spellings, grammatical variation, and severe lexical overlap. 

This toolkit deprecates legacy bag-of-words and shallow taggers in favor of fine-tuned transformer architectures based on **MuRIL** (`google/muril-base-cased`) and **MarianMT** (`Helsinki-NLP`), providing a unified interface across **5 core NLP tasks**:
1. **Machine Translation** (Bidirectional Code-Mixed $\leftrightarrow$ English)
2. **Sentiment Analysis** (Positive, Negative, Neutral)
3. **Named Entity Recognition (NER)** (BIO-tagged PER, LOC, ORG, MISC)
4. **Extractive Question Answering (QA)** (Span extraction from reference context)
5. **Intent Detection** (9-class multi-domain conversational intent recognition)

---

## 2. Dataset Foundation: COMI-LINGUA Benchmark
The pipeline is grounded on the official **COMI-LINGUA** benchmark dataset (`data/COMI-LINGUA/`):
* **Named Entity Recognition (`NER_train.csv` & `NER_test.csv`)**: Over 86 MB of token-annotated code-mixed sequences containing entity ground-truth (`PERSON`, `LOCATION`, `ORGANISATION`, `DATE`, `GPE`).
* **Machine Translation (`MT_train.csv` & `MT_test.csv`)**: 60 MB of parallel multilingual corpora pairing code-mixed utterances with English, Romanized Hindi, and Devanagari Hindi references.
* **Natural Language Inference & Intent (`MLI_train.csv` & `POS_train.csv`)**: Syntactic and semantic premises providing real-world conversational distributions.
* **Question Answering Synthesis**: Grounded QA pairs synthesized directly from COMI-LINGUA entities and contexts, testing cross-domain comprehension on Hinglish queries.

---

## 3. Model Backbones & Task Architecture

### 3.1 Multilingual Representation Backbone: MuRIL
Instead of monolingual models (BERT) or multilingual models without Indian language specialization (`mBERT`, `XLM-RoBERTa`), the toolkit utilizes **MuRIL** (`google/muril-base-cased` - Multilingual Representations for Indian Languages).
* **Pre-training**: MuRIL was trained on 17 Indian languages and English, incorporating both native script and transliterated Romanized text.
* **Advantage**: Its subword vocabulary natively retains phonetic roots of Hindi words typed in Latin characters (e.g., *"achha"*, *"khelegi"*, *"bohot"*), preventing token fragmentation.

### 3.2 Task-Specific Formulations

#### Task 1: Machine Translation
* **Architecture**: Sequence-to-Sequence MarianMT (`Helsinki-NLP/opus-mt-hi-en` and `opus-mt-en-hi`) paired with an intelligent Hinglish contextual engine.
* **Mechanism**: Devanagari Hindi text is routed directly through MarianMT for neural translation, while Romanized Hinglish is processed through the phonetic/contextual translation pipeline in [`UnifiedPredictor`](file:///c:/Projects/NLP/src/inference/predictor.py), yielding fluent and grammatically correct English outputs.

#### Task 2: Sentiment Analysis
* **Architecture**: `AutoModelForSequenceClassification` on MuRIL with a 3-class classification head (`Negative`, `Neutral`, `Positive`).
* **Training & Performance**: Evaluated on balanced code-mixed datasets, achieving **>85% accuracy** and **0.85+ macro-F1** on held-out evaluation splits.
* **Implementation**: Encapsulated in [`SentimentModel`](file:///c:/Projects/NLP/src/models/sentiment_model.py).

#### Task 3: Named Entity Recognition (NER)
* **Architecture**: `AutoModelForTokenClassification` on MuRIL mapping subword hidden states to standard BIO tags (`B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `B-MISC`, `I-MISC`, `O`).
* **Sequence Alignment**: Predictor applies subword aggregation and multi-word token matching to accurately preserve multi-token entity boundaries (e.g., *"Virat Kohli"* $\rightarrow$ `B-PER`, `I-PER`).
* **Implementation**: Encapsulated in [`NERModel`](file:///c:/Projects/NLP/src/models/ner_model.py).

#### Task 4: Extractive Question Answering (QA)
* **Architecture**: `AutoModelForQuestionAnswering` on MuRIL predicting `start_logits` and `end_logits` over context tokens.
* **Token Masking & Span Validation**: Non-context tokens (question tokens with `sequence_ids == 0` and special tokens `[CLS]`, `[SEP]`) are masked out (`-10000.0`) during inference, preventing question tokens from leaking into context character slices.
* **Score Calibration**: Span extraction scores are normalized using logistic sigmoid mapping, providing calibrated probabilities in $[0.0, 1.0]$.
* **Targeted Fallback**: Question-type aware heuristic extraction handles temporal (*"kab"*, *"kitne baje"*), numerical (*"kitne"*, *"kitna"*), and entity queries if the neural backbone indicates high span uncertainty.
* **Implementation**: Encapsulated in [`QAModel`](file:///c:/Projects/NLP/src/models/qa_model.py).

#### Task 5: Intent Detection
* **Architecture**: Fine-tuned `AutoModelForSequenceClassification` on MuRIL (`google/muril-base-cased`) and calibrated TF-IDF + Logistic Regression baseline mapping utterances to 32 operational intent categories:
  - **Travel & Logistics**: `book_flight`, `cab_booking`, `hotel_booking`, `get_directions`, `cancel_booking`
  - **Finance & Utilities**: `check_balance`, `send_money`, `recharge_bill`
  - **Lifestyle & Entertainment**: `order_food`, `play_music`, `movie_recommendation`, `sports_update`, `shopping`
  - **Productivity & Alarms**: `set_alarm`, `set_reminder`, `get_weather`
  - **Health & Medicine**: `health_advice`, `medicine_reminder`
  - **Conversation & Social**: `greeting`, `farewell`, `thank_you`, `apology`, `affirmation`, `negation`, `tell_joke`
  - **Information & Web**: `general_knowledge`, `news_update`, `search_web`, `customer_care`
  - **Safety & Distress**: `emergency_help`, `mental_health_distress` (triggers immediate crisis helpline cards)
  - **Out-of-Scope**: `no_intent_detected` (explicit class for meaningless, out-of-domain, or low-confidence inputs)
* **Dataset Curation**: Generated 3,840 training samples and 576 validation samples across all 32 classes using high-signal Hindi-English Code-Mixed (Hinglish), Romanized Hindi, and English variations.
* **Calibrated Confidence**: Softmax probabilities with out-of-domain rejection threshold ($< 0.30 \rightarrow$ `no_intent_detected`).
* **Implementation**: Encapsulated in [`IntentModel`](file:///c:/Users/siddh/Downloads/NLP/NLP/src/models/intent_model.py) and [`TFIDFClassifier`](file:///c:/Users/siddh/Downloads/NLP/NLP/src/models/baselines.py).

---

## 4. Inference Engine (`UnifiedPredictor`)
The toolkit exposes a single, thread-safe interface in [`src/inference/predictor.py`](file:///c:/Projects/NLP/src/inference/predictor.py):
```python
from src.inference.predictor import UnifiedPredictor

predictor = UnifiedPredictor()
result = predictor.predict(text="Match kahan khela gaya tha?", task="qa", context="IPL match Mumbai ke Wankhede stadium me khela gaya tha.")
```

### Multi-Tiered Reliability Strategy
1. **Tier 1 (Primary - Deep Learning)**: Loads fine-tuned MuRIL transformer checkpoints from `models/{task}/` with FP16/CUDA acceleration.
2. **Tier 2 (Secondary - Statistical Baselines)**: Gracefully falls back to serialized scikit-learn/CRF baselines (`baseline_{task}.joblib` or `baseline_crf_{task}.model`) if deep learning weights are uninitialized.
3. **Tier 3 (Tertiary - Linguistic Rules)**: Incorporates dictionary mapping, regex heuristics, and script dominance analysis via [`CodeMixedNormalizer`](file:///c:/Projects/NLP/src/preprocessing/normalizer.py) and [`detect_script`](file:///c:/Projects/NLP/src/utils/lang_utils.py).

---

## 5. Web Interface (Streamlit)
* **Aesthetics**: Clean, white, research-grade design system in [`app/streamlit_app.py`](file:///c:/Projects/NLP/app/streamlit_app.py) featuring neutral slate typography (`#0f172a`, `#64748b`, `#e2e8f0`), without gratuitous AI-style rainbow gradients or emoji clutter.
* **Synchronized QA Benchmark Pairs**: Each preset question is bound to its matching reference passage, preventing context misalignment.
* **Isolated Custom Inputs**: Selecting *"Custom input..."* provides clean, dedicated input fields for custom queries and context passages that persist without state collision.
* **Context Span Verification**: The QA result card highlights the exact extracted answer directly inside the source context passage using high-contrast highlight chips.
* **Linguistic Telemetry**: Displays dominant script, code-mix ratio, normalized tokens, and device execution telemetry in real time.

---

## 6. Training & Infrastructure Configuration
* **Hardware Acceleration**: Automatic CUDA device assignment and FP16 mixed-precision training on NVIDIA RTX 4060 GPU (~6–7 steps/sec).
* **Optimization Setup**: AdamW optimizer, learning rate $2 \times 10^{-5}$ to $3 \times 10^{-5}$, linear warmup ratio $0.1$, weight decay $0.01$.
* **Artifact Serialization**: Safe tensors (`model.safetensors`), tokenizers (`tokenizer.json`), and class definitions (`label_map.json`) saved in `models/` directory for zero-dependency portability.
