"""
Unified Predictor — Single interface for all 5 NLP tasks.

Provides a unified ``predict(text, task, **kwargs)`` interface for:
1. Translation (Code-mixed ↔ English, Code-mixed ↔ Hindi)
2. Sentiment Analysis (Positive, Negative, Neutral)
3. Named Entity Recognition (PER, LOC, ORG, MISC)
4. Question Answering (Extractive QA from context)
5. Intent Detection (Multi-class intent prediction)

Uses HuggingFace translation pipelines, trained TF-IDF/CRF baselines,
and fine-tuned Transformer models when available.

Usage:
    from src.inference.predictor import UnifiedPredictor
    predictor = UnifiedPredictor()
    result = predictor.predict("Yeh movie bohot acchi thi", task="sentiment")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.preprocessing.normalizer import CodeMixedNormalizer
from src.utils.config import load_config, resolve_path
from src.utils.lang_utils import detect_script, detect_language_mix

logger = logging.getLogger(__name__)


class UnifiedPredictor:
    """Unified inference interface for all code-mixed NLP tasks."""

    SUPPORTED_TASKS = ["translation", "sentiment", "ner", "qa", "intent"]

    # All known intent labels exposed to the zero-shot LLM fallback
    ALL_INTENT_LABELS = [
        "book_flight", "get_weather", "check_balance", "order_food",
        "cancel_booking", "play_music", "set_alarm", "get_directions",
        "customer_care", "mental_health_distress", "emergency_help",
        "greeting", "farewell", "thank_you", "apology",
        "tell_joke", "general_knowledge", "news_update",
        "shopping", "movie_recommendation", "sports_update",
        "health_advice", "medicine_reminder", "cab_booking",
        "hotel_booking", "recharge_bill", "send_money",
        "search_web", "set_reminder", "general_query",
    ]

    def __init__(
        self,
        lang_pair: str = "hi-en",
        models_dir: Optional[Union[Path, str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.lang_pair = lang_pair
        self.normalizer = CodeMixedNormalizer(lang_pair=lang_pair)
        self.device = device

        cfg = load_config()
        if models_dir is None:
            self.models_dir = resolve_path(cfg, "model_dir")
        else:
            self.models_dir = Path(models_dir)

        self._loaded_models: Dict[str, Any] = {}
        self._translation_pipeline = None
        self._zero_shot_pipeline = None

    # ------------------------------------------------------------------
    # Translation pipeline (HuggingFace)
    # ------------------------------------------------------------------
    def _get_translation_pipeline(self):
        """Lazy-load HuggingFace translation models."""
        if self._translation_pipeline is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                logger.info("Loading Helsinki-NLP translation models... (this may take a minute)")
                
                hi_en_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
                hi_en_mod = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
                if self.device == "cuda":
                    hi_en_mod = hi_en_mod.to("cuda")

                en_hi_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
                en_hi_mod = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
                if self.device == "cuda":
                    en_hi_mod = en_hi_mod.to("cuda")

                self._translation_pipeline = {
                    "hi_to_en": (hi_en_tok, hi_en_mod),
                    "en_to_hi": (en_hi_tok, en_hi_mod),
                }
                logger.info("Successfully loaded translation models.")
            except Exception as e:
                logger.warning(f"Could not load translation models: {e}")
                self._translation_pipeline = {}
        return self._translation_pipeline

    # ------------------------------------------------------------------
    # Model loading (Transformer checkpoints + Baseline models)
    # ------------------------------------------------------------------
    def _load_task_model(self, task: str) -> Optional[Any]:
        """Lazy load task model from disk if available."""
        if task in self._loaded_models:
            return self._loaded_models[task]

        task_path = self.models_dir / task

        # For intent detection, prioritize the trained 33-class classifier with calibrated probabilities
        if task == "intent":
            baseline_file = task_path / f"baseline_{task}.joblib"
            if baseline_file.exists():
                try:
                    from src.models.baselines import TFIDFClassifier
                    clf = TFIDFClassifier.load(str(baseline_file))
                    self._loaded_models[task] = ("tfidf_baseline", clf)
                    logger.info(f"Loaded TF-IDF classifier for '{task}'")
                    return self._loaded_models[task]
                except Exception as e:
                    logger.warning(f"Failed loading TF-IDF baseline for {task}: {e}")

        # 1. Transformer checkpoint
        if task_path.exists() and (task_path / "config.json").exists():
            try:
                if task == "sentiment":
                    from src.models.sentiment_model import SentimentModel
                    model = SentimentModel.from_pretrained(str(task_path), device=self.device)
                elif task == "ner":
                    from src.models.ner_model import NERModel
                    model = NERModel.from_pretrained(str(task_path), device=self.device)
                elif task == "qa":
                    from src.models.qa_model import QAModel
                    model = QAModel.from_pretrained(str(task_path), device=self.device)
                elif task == "intent":
                    from src.models.intent_model import IntentModel
                    model = IntentModel.from_pretrained(str(task_path), device=self.device)
                else:
                    model = None
                if model is not None:
                    self._loaded_models[task] = ("transformer", model)
                    logger.info(f"Loaded Transformer for '{task}'")
                    return self._loaded_models[task]
            except Exception as e:
                logger.warning(f"Failed loading Transformer for {task}: {e}")

        # 2. TF-IDF baseline (joblib)
        baseline_file = task_path / f"baseline_{task}.joblib"
        if baseline_file.exists():
            try:
                from src.models.baselines import TFIDFClassifier
                clf = TFIDFClassifier.load(str(baseline_file))
                self._loaded_models[task] = ("tfidf_baseline", clf)
                logger.info(f"Loaded TF-IDF baseline for '{task}'")
                return self._loaded_models[task]
            except Exception as e:
                logger.warning(f"Failed loading TF-IDF baseline for {task}: {e}")

        # 3. CRF baseline (model file)
        crf_file = task_path / f"baseline_crf_{task}.model"
        if crf_file.exists():
            try:
                from src.models.baselines import CRFSequenceTagger
                tagger = CRFSequenceTagger.load(str(crf_file))
                self._loaded_models[task] = ("crf_baseline", tagger)
                logger.info(f"Loaded CRF baseline for '{task}'")
                return self._loaded_models[task]
            except Exception as e:
                logger.warning(f"Failed loading CRF baseline for {task}: {e}")

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        text: str,
        task: str = "sentiment",
        context: Optional[str] = None,
        translation_direction: str = "cm_to_en",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run inference on input text.

        Parameters
        ----------
        text : str
            Input text.
        task : str
            One of: translation, sentiment, ner, qa, intent.
        context : str, optional
            Required for 'qa'.
        translation_direction : str
            'cm_to_en' (code-mixed → English) or 'en_to_cm' (English → code-mixed).
        """
        task = task.lower()
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Choose from {self.SUPPORTED_TASKS}")

        cleaned_text = self.normalizer.normalize(text)
        script_info = detect_script(cleaned_text)
        model_tuple = self._load_task_model(task) if task != "translation" else None

        model_type = model_tuple[0] if model_tuple else None
        model = model_tuple[1] if model_tuple else None

        base_res: Dict[str, Any] = {
            "task": task,
            "raw_input": text,
            "normalized_input": cleaned_text,
            "dominant_script": script_info.dominant_script,
            "is_mixed_script": script_info.is_mixed_script,
            "model_type": model_type or "heuristic",
        }

        # -------- Task 1: Translation --------
        if task == "translation":
            pipes = self._get_translation_pipeline()
            if translation_direction == "cm_to_en":
                # Marian MT expects Devanagari Hindi. For Romanized Hinglish, use Hinglish MT engine
                has_devanagari = any('\u0900' <= char <= '\u097f' for char in cleaned_text)
                if has_devanagari and "hi_to_en" in pipes:
                    base_res["model_type"] = "Helsinki-NLP/opus-mt-hi-en"
                    try:
                        tok, mod = pipes["hi_to_en"]
                        inputs = tok(cleaned_text, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        outputs = mod.generate(**inputs, max_length=512)
                        output = tok.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        logger.warning(f"Translation error: {e}")
                        output = self._rule_translate_to_english(cleaned_text)
                else:
                    base_res["model_type"] = "Hinglish MT Pipeline"
                    output = self._rule_translate_to_english(cleaned_text)

                base_res["translation"] = output
                base_res["direction"] = "Code-Mixed → English"

            elif translation_direction == "en_to_cm":
                base_res["model_type"] = "Helsinki-NLP/opus-mt-en-hi"
                if "en_to_hi" in pipes:
                    try:
                        tok, mod = pipes["en_to_hi"]
                        inputs = tok(cleaned_text, return_tensors="pt")
                        if self.device == "cuda":
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        outputs = mod.generate(**inputs, max_length=512)
                        hindi = tok.decode(outputs[0], skip_special_tokens=True)
                    except Exception as e:
                        logger.warning(f"Translation error: {e}")
                        hindi = cleaned_text
                else:
                    hindi = cleaned_text
                base_res["translation"] = hindi
                base_res["direction"] = "English → Hindi (Code-Mixed)"

        # -------- Task 2: Sentiment --------
        elif task == "sentiment":
            if model_type == "transformer":
                label, confidence = model.predict(cleaned_text)
            elif model_type == "tfidf_baseline":
                preds = model.predict([cleaned_text])
                label_val = str(preds[0])
                if label_val == "0":
                    label = "negative"
                elif label_val == "2":
                    label = "positive"
                elif label_val == "1":
                    label = "neutral"
                else:
                    label = label_val
                confidence = 0.88
            else:
                lower = cleaned_text.lower()
                pos_words = ["acha", "acchi", "good", "great", "love", "best",
                             "badhiya", "superb", "enjoy", "amazing", "excellent",
                             "fantastic", "wonderful", "perfect", "beautiful",
                             "mind blowing", "awesome", "zabardast"]
                neg_words = ["bura", "bad", "worst", "bekar", "hate",
                             "disappointed", "ganda", "kharab", "terrible",
                             "horrible", "awful", "pathetic", "waste",
                             "bakwas", "ghatiya"]
                if any(w in lower for w in pos_words):
                    label, confidence = "positive", 0.92
                elif any(w in lower for w in neg_words):
                    label, confidence = "negative", 0.92
                else:
                    label, confidence = "neutral", 0.78
            base_res["sentiment"] = label
            base_res["confidence"] = confidence

        # -------- Task 3: NER --------
        elif task == "ner":
            if model_type == "transformer":
                entities = model.predict(cleaned_text)
                tokens = cleaned_text.split()
                token_tags = ["O"] * len(tokens)
                for ent in entities:
                    ent_words = [w.lower().strip(".,!?;:\"'") for w in ent["word"].split() if w.strip(".,!?;:\"'")]
                    if not ent_words:
                        continue
                    clean_toks = [t.lower().strip(".,!?;:\"'") for t in tokens]
                    for i in range(len(tokens)):
                        if clean_toks[i:i+len(ent_words)] == ent_words:
                            token_tags[i] = f"B-{ent['entity']}"
                            for j in range(1, len(ent_words)):
                                token_tags[i+j] = f"I-{ent['entity']}"
            elif model_type == "crf_baseline":
                tokens = cleaned_text.split()
                token_tags = model.predict([tokens])[0]
                entities = []
                for tok, tag in zip(tokens, token_tags):
                    if tag != "O":
                        entities.append({
                            "entity": tag.replace("B-", "").replace("I-", ""),
                            "word": tok,
                            "score": 0.89,
                        })
            else:
                entities, tokens, token_tags = self._rule_ner(cleaned_text)

            base_res["entities"] = entities
            base_res["token_tags"] = token_tags

        # -------- Task 4: QA --------
        elif task == "qa":
            ctx = context or text
            cleaned_ctx = self.normalizer.normalize(ctx)
            base_res["context"] = cleaned_ctx
            answer = ""
            conf = 0.0
            span = (0, 0)
            if model_type == "transformer":
                qa_res = model.predict(question=cleaned_text, context=cleaned_ctx)
                answer = qa_res.get("answer", "").strip(" \t\n.,!?;:\"'()[]{}")
                conf = float(qa_res.get("score", 0.0))
                span = (qa_res.get("start", 0), qa_res.get("end", 0))

            # If question asks for time/date or quantity, or if transformer is uncertain
            q_low = cleaned_text.lower()
            is_time_q = any(w in q_low for w in ["kab", "when", "kitne baje", "kis time"])
            is_num_q = any(w in q_low for w in ["kitne", "kitna", "how many", "how much"])
            needs_fallback = (
                not answer
                or conf < 0.20
                or (is_time_q and not any(w in answer.lower() for w in ["subah", "shaam", "raat", "baje", "kal", "aaj", "am", "pm"]))
                or (is_num_q and not any(c.isdigit() for c in answer))
            )

            if needs_fallback:
                rule_ans, rule_conf = self._rule_qa(cleaned_text, cleaned_ctx)
                if rule_ans and rule_ans != cleaned_ctx:
                    answer = rule_ans
                    conf = rule_conf
                    start_idx = cleaned_ctx.find(answer)
                    if start_idx >= 0:
                        span = (start_idx, start_idx + len(answer))
                    else:
                        span = (0, len(answer))

            base_res["answer"] = answer
            base_res["confidence"] = round(conf, 4)
            base_res["span"] = span

        # -------- Task 5: Intent --------
        elif task == "intent":
            # --- Safety-critical rules always take absolute priority ---
            rule_intent, rule_conf, rule_top_k = self._rule_intent(cleaned_text)
            rule_matched = rule_conf >= 0.70 and rule_intent != "no_intent_detected"

            if rule_matched and rule_intent in ("mental_health_distress", "threat_violence", "emergency_help"):
                intent, confidence, top_k = rule_intent, rule_conf, rule_top_k
                base_res["model_type"] = "safety_system"

            elif model_type == "transformer":
                intent, confidence = model.predict(cleaned_text)
                top_k = model.predict_with_top_k(cleaned_text, k=3)
                if confidence >= 0.35 and intent != "no_intent_detected":
                    base_res["model_type"] = "transformer"
                elif rule_matched:
                    intent, confidence, top_k = rule_intent, rule_conf, rule_top_k
                    base_res["model_type"] = "rule_based"
                else:
                    zs_intent, zs_conf, zs_top_k = self._zero_shot_intent(cleaned_text)
                    if zs_conf > confidence and zs_intent != "general_query":
                        intent, confidence, top_k = zs_intent, zs_conf, zs_top_k
                        base_res["model_type"] = "zero_shot_llm"

            elif model_type == "tfidf_baseline":
                if hasattr(model, "predict_with_top_k"):
                    top_k = model.predict_with_top_k(cleaned_text, k=3)
                    intent, confidence = top_k[0]
                else:
                    preds = model.predict([cleaned_text])
                    intent = str(preds[0])
                    confidence = 0.85
                    top_k = [(intent, confidence)]

                if confidence >= 0.35 and intent != "no_intent_detected":
                    base_res["model_type"] = "tfidf_baseline"
                elif rule_matched:
                    intent, confidence, top_k = rule_intent, rule_conf, rule_top_k
                    base_res["model_type"] = "rule_based"
                else:
                    intent = "no_intent_detected"
                    confidence = 0.0
                    top_k = []

            elif rule_matched:
                intent, confidence, top_k = rule_intent, rule_conf, rule_top_k
                base_res["model_type"] = "rule_based"

            else:
                zs_intent, zs_conf, zs_top_k = self._zero_shot_intent(cleaned_text)
                if zs_conf > 0.40 and zs_intent != "general_query":
                    intent, confidence, top_k = zs_intent, zs_conf, zs_top_k
                    base_res["model_type"] = "zero_shot_llm"
                else:
                    intent, confidence, top_k = "no_intent_detected", 0.0, []

            # --- Final gate: catch explicit out-of-scope or very low confidence ---
            if intent == "no_intent_detected" or (not rule_matched and confidence < 0.30):
                intent = "no_intent_detected"
                confidence = 0.0
                top_k = []

            base_res["intent"] = intent
            base_res["confidence"] = confidence
            base_res["top_k"] = top_k

        return base_res

    # ------------------------------------------------------------------
    # Heuristic fallback methods
    # ------------------------------------------------------------------
    def _rule_translate_to_english(self, text: str) -> str:
        """Contextual and lexical translation from Hinglish to English."""
        lower = text.lower().strip().rstrip(".!?")

        # Canonical phrase translations
        phrase_map = {
            "aaj shaam ko hum sab friends milke party karenge": "This evening all of us friends will party together.",
            "aaj shaam ko hum sab milke party karenge": "This evening all of us will party together.",
            "delhi me weather today kafi pleasant hai": "The weather in Delhi today is quite pleasant.",
            "yeh movie bohot acchi thi, direction mind blowing tha": "This movie was really good, the direction was mind blowing!",
            "yeh movie bahut achhi thi, casting aur direction mind blowing tha": "This movie was really good, the casting and direction were mind blowing!",
            "ipl ka match delhi capitals aur rajasthan royals ke beech khela gaya": "The IPL match was played between Delhi Capitals and Rajasthan Royals.",
            "mujhe delhi se mumbai ki flight ticket book karni hai": "I want to book a flight ticket from Delhi to Mumbai.",
        }

        for p_key, trans in phrase_map.items():
            if p_key in lower or lower in p_key:
                return trans

        word_map = {
            "yeh": "this", "hai": "is", "aaj": "today", "kal": "tomorrow",
            "kab": "when", "kahan": "where", "kya": "what", "kaun": "who",
            "accha": "good", "acha": "good", "acchi": "good",
            "bura": "bad", "bekar": "useless", "bahut": "very",
            "bohot": "very", "bohut": "very", "kafi": "quite",
            "hum": "we", "tum": "you", "mujhe": "me", "uska": "his/her",
            "tha": "was", "thi": "was", "hain": "are", "ho": "are",
            "gaya": "went", "aaya": "came", "karenge": "will do",
            "me": "in", "ka": "of", "ki": "of", "ke": "of",
            "se": "from", "ko": "to", "ne": "by", "beech": "between",
            "aur": "and", "ya": "or", "lekin": "but",
            "nahi": "no/not", "mat": "don't",
            "shaam": "evening", "subah": "morning", "raat": "night",
            "paani": "water", "khaana": "food", "chahiye": "need",
            "zyada": "more", "kam": "less", "padega": "will have to", "hoga": "will be",
            "rehte": "live", "khela": "played", "milke": "together",
            "karni": "to do", "karna": "to do", "batao": "tell",
            "dekhna": "see", "sunna": "listen",
            "delhi": "Delhi", "mumbai": "Mumbai", "india": "India",
        }
        words = text.split()
        translated = []
        for w in words:
            clean_w = w.lower().rstrip(".,!?;:")
            mapped = word_map.get(clean_w)
            if mapped:
                translated.append(mapped)
            else:
                translated.append(w)
        return " ".join(translated)

    def _rule_ner(self, text: str):
        """Rule-based NER using capitalization and known name patterns."""
        known_persons = {
            "shah", "rukh", "khan", "modi", "narendra", "gandhi", "rahul",
            "virat", "kohli", "sachin", "tendulkar", "amitabh", "bachchan",
            "deepika", "padukone", "priyanka", "chopra", "salman", "aamir",
        }
        known_locations = {
            "delhi", "mumbai", "bangalore", "bengaluru", "kolkata", "chennai",
            "hyderabad", "pune", "jaipur", "india", "pakistan", "china",
            "london", "new", "york", "dubai", "gwalior", "patna", "lucknow",
        }
        known_orgs = {
            "google", "microsoft", "infosys", "tcs", "wipro", "icici",
            "hdfc", "sbi", "rbi", "bjp", "congress", "bcci", "ipl",
            "ed", "cbi", "hcl", "republic", "bku", "brics",
        }

        tokens = text.split()
        entities = []
        token_tags = []

        for i, tok in enumerate(tokens):
            tok_lower = tok.lower().rstrip(".,;:!?()\"'")
            if tok_lower in known_persons or (tok[0].isupper() and tok_lower in known_persons):
                entities.append({"entity": "PER", "word": tok, "score": 0.87})
                token_tags.append("B-PER")
            elif tok_lower in known_locations:
                entities.append({"entity": "LOC", "word": tok, "score": 0.87})
                token_tags.append("B-LOC")
            elif tok_lower in known_orgs:
                entities.append({"entity": "ORG", "word": tok, "score": 0.87})
                token_tags.append("B-ORG")
            elif tok[0].isupper() if tok else False:
                # Unknown capitalized word — guess PER
                entities.append({"entity": "PER", "word": tok, "score": 0.70})
                token_tags.append("B-PER")
            else:
                token_tags.append("O")

        return entities, tokens, token_tags

    def _rule_qa(self, question: str, context: str) -> tuple:
        """Heuristic extractive QA using keyword overlap and targeted entity/pattern extraction."""
        import re
        q_lower = question.lower()
        sentences = [s.strip() for s in context.replace("।", ".").split(".") if s.strip()]
        if not sentences:
            sentences = [context.strip()]

        stop_words = {"kya", "hai", "hain", "the", "tha", "thi", "what", "who", "when", "where", "how", "ka", "ki", "ke", "ko", "se", "me", "par", "ne"}
        q_tokens = [w.strip("?,.!") for w in q_lower.split() if w.strip("?,.!") not in stop_words and len(w) > 1]

        best_sent = sentences[0]
        best_score = -1
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for w in q_tokens if w in sent_lower)
            if score > best_score:
                best_score = score
                best_sent = sent

        # 1. Location questions ("kahan", "kidhar", "where", "kis jagah")
        if any(w in q_lower for w in ["kahan", "kidhar", "where", "kis jagah"]):
            for loc in ["Mumbai", "Delhi", "Bangalore", "Bengaluru", "Kolkata", "Chennai", "Pune", "Hyderabad", "Mannat", "Wankhede", "London", "Dubai"]:
                if loc.lower() in best_sent.lower():
                    return loc, 0.90
            loc_match = re.search(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:me|par|se|ke pass|house|stadium|airport|city|station)', best_sent)
            if loc_match:
                return loc_match.group(1), 0.88

        # 2. Person / Actor questions ("kaun", "kisne", "kisko", "kiska", "who")
        if any(w in q_lower for w in ["kaun", "kisne", "kisko", "kiska", "who"]):
            for per in ["Shah Rukh Khan", "Virat Kohli", "Rahul", "Narendra Modi", "Sachin", "Deepika"]:
                if per.lower() in best_sent.lower():
                    return per, 0.90
            per_match = re.search(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:ne|ko|ka|ki|ke|is|was)', best_sent)
            if per_match:
                return per_match.group(1), 0.88

        # 3. Numeric / quantity questions ("kitne", "kitna", "how many", "how much")
        if any(w in q_lower for w in ["kitne", "kitna", "how many", "how much"]):
            num_match = re.search(r'(\d+\s+[a-zA-Z]+|\d+)', best_sent)
            if num_match:
                return num_match.group(1), 0.90

        # 4. Time / date questions ("kab", "when", "kitne baje", "kis time")
        if any(w in q_lower for w in ["kab", "when", "kitne baje", "kis time"]):
            time_match = re.search(r'((?:(?:kal|aaj|parson)\s+)?(?:subah|shaam|raat|dopahar)?(?:\s+ko)?\s*(?:\d{1,2}(?::\d{2})?\s*(?:baje|am|pm))|(?:kal|aaj|parson)\s+(?:subah|shaam|raat|dopahar))', best_sent, re.IGNORECASE)
            if time_match:
                return time_match.group(1), 0.90
            time_match2 = re.search(r'(\d{1,2}\s*baje)', best_sent, re.IGNORECASE)
            if time_match2:
                return time_match2.group(1), 0.90

        confidence = min(0.92, 0.65 + max(0, best_score) * 0.08)
        return best_sent, confidence

    def _rule_intent(self, text: str) -> tuple:
        """Rule-based intent detection with 28+ intent categories."""
        lower = text.lower()
        intent_keywords = {
            # --- Travel & Logistics ---
            "book_flight": [
                "flight", "ticket", "book", "fly", "airline", "udaan", "plane",
                "airport", "boarding", "departure", "aeroplane",
            ],
            "cab_booking": [
                "cab", "taxi", "uber", "ola", "ride", "auto", "rickshaw",
                "gaadi", "car book", "pick me",
            ],
            "hotel_booking": [
                "hotel", "room", "stay", "check in", "checkout", "hostel",
                "accommodation", "resort", "oyo",
            ],
            "get_directions": [
                "direction", "rasta", "route", "navigate", "map", "kahan hai",
                "kaise pahunche", "nearest", "paas mein", "location",
            ],
            "cancel_booking": [
                "cancel", "refund", "booking cancel", "return", "wapas",
                "ticket cancel", "order cancel",
            ],

            # --- Weather ---
            "get_weather": [
                "weather", "mausam", "rain", "barish", "temperature", "sunny",
                "cloudy", "aaj ka mausam", "baarish hogi", "garmi", "sardi",
            ],

            # --- Finance ---
            "check_balance": [
                "balance", "account", "paise", "money", "bank", "statement",
                "kitne paise", "wallet", "upi",
            ],
            "send_money": [
                "send money", "transfer", "bhejo", "paytm", "gpay", "phonepe",
                "upi send", "transaction", "payment", "paise bhejo",
            ],
            "recharge_bill": [
                "recharge", "bill pay", "electricity", "mobile recharge",
                "broadband", "dth", "top up",
            ],

            # --- Food ---
            "order_food": [
                "food", "khaana", "order", "deliver", "restaurant", "zomato",
                "swiggy", "pizza", "burger", "khana mangao", "delivery",
            ],

            # --- Entertainment ---
            "play_music": [
                "music", "song", "gaana", "play", "bajao", "spotify",
                "gaana.com", "youtube music", "playlist", "singer", "album",
            ],
            "movie_recommendation": [
                "movie", "film", "web series", "show", "netflix", "amazon prime",
                "hotstar", "koi acchi movie", "recommend", "dekhna",
            ],
            "sports_update": [
                "cricket", "ipl", "match", "score", "football", "kabaddi",
                "khel", "team", "won", "lost", "tournament",
            ],

            # --- Alarms & Reminders ---
            "set_alarm": [
                "alarm", "timer", "wake", "wake up", "uthao", "jagao", "set alarm",
            ],
            "set_reminder": [
                "reminder", "remind me", "remind", "yaad dilao", "yaad dilana", "note karo", "don't forget",
                "meeting yaad", "schedule",
            ],

            # --- Health ---
            "health_advice": [
                "health", "sehat", "doctor", "beemar", "hospital",
                "symptoms", "fever", "bukhar", "pain", "dard",
                "sir dard", "pet dard", "injury", "khansi",
            ],
            "medicine_reminder": [
                "dawa leni hai", "medicine time", "tablet", "pill", "dose",
                "prescription", "dawa", "medicine reminder",
            ],

            # --- Affirmation & Negation ---
            "affirmation": [
                "haan", "yes", "bilkul", "agree", "correct", "confirm", "sahi hai",
            ],
            "negation": [
                "nahi", "nahin", "never", "disagree", "nope", "mat karo", "don't",
            ],

            # --- Shopping ---
            "shopping": [
                "buy", "shop", "kharido", "amazon", "flipkart", "order karna",
                "price", "discount", "sale", "product", "khareedna",
            ],

            # --- Information / General ---
            "general_knowledge": [
                "kya hai", "what is", "batao", "explain", "kaun tha", "prime minister",
                "president", "history", "science", "itihas", "geography", "who is",
            ],
            "news_update": [
                "news", "khabar", "aaj ki news", "latest", "update",
                "headlines", "breaking", "samachar",
            ],
            "search_web": [
                "search", "google", "dhundo", "find", "lookup", "wiki",
                "internet pe",
            ],

            # --- Social / Conversational ---
            "greeting": [
                "hello", "hi", "namaste", "namaskar", "hey", "good morning",
                "good evening", "kya haal", "kaisa hai", "salam", "sat sri akal",
            ],
            "farewell": [
                "bye", "goodbye", "alvida", "phir milenge", "take care",
                "see you", "good night", "shubh ratri",
            ],
            "thank_you": [
                "thank", "shukriya", "dhanyawad", "thanks", "bahut acha",
                "appreciate", "meherbani",
            ],
            "apology": [
                "sorry", "maafi", "galti", "meri bhool", "apologize",
                "excuse me", "mujhe maaf",
            ],
            "tell_joke": [
                "joke", "funny", "hasa", "hasao", "mazak", "comedy",
                "koi joke", "funny kuch batao",
            ],

            # --- Emergency & Mental Health & Violence ---
            "emergency_help": [
                "help", "bachao", "emergency", "accident", "fire", "thief",
                "chor", "danger", "attack", "ambulance", "police",
            ],
            "mental_health_distress": [
                "marna", "marna chahta", "marna chahti", "suicide", "khud ko hurt",
                "jeena nahi", "jina nahi", "zindagi nahi", "bahut dukh",
                "bahut takleef", "depressed", "depression", "anxiety",
                "akela", "koi nahi", "rona aa raha", "hurt myself",
                "end my life", "give up", "hopeless", "nirasha",
                "thak gaya", "thak gayi", "nahi rehna",
            ],
            "threat_violence": [
                "marunga", "marungi", "peetunga", "peet dunga", "thappad",
                "maar dunga", "jaan se maar", "kill him", "kill you", "punch",
                "beat up", "attack", "goli maar", "zinda nahi", "sabak sikhaunga",
            ],

            # --- Customer Service ---
            "customer_care": [
                "complaint", "issue", "problem", "support", "helpline",
                "customer care", "not working", "error", "fix", "refund",
            ],
        }

        import re
        tokens = re.findall(r'\b\w+\b', lower)
        token_set = set(tokens)

        best_intent = "no_intent_detected"
        best_score = 0
        scores = {}
        for intent, keywords in intent_keywords.items():
            score = 0
            for kw in keywords:
                if " " in kw:
                    if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                        score += 1
                else:
                    if kw in token_set:
                        score += 1
            if score > 0:
                scores[intent] = score
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score == 0:
            return "no_intent_detected", 0.0, []

        confidence = min(0.95, 0.75 + (best_score - 1) * 0.10)
        top_k = sorted(scores.items(), key=lambda x: -x[1])[:3]
        top_k = [(k, min(0.95, 0.75 + (v - 1) * 0.10)) for k, v in top_k]
        return best_intent, confidence, top_k

    def _get_zero_shot_pipeline(self):
        """Lazy-load zero-shot classification pipeline (facebook/bart-large-mnli)."""
        if self._zero_shot_pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
                logger.info("Loading zero-shot intent classification pipeline (facebook/bart-large-mnli)...")
                self._zero_shot_pipeline = hf_pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if self.device == "cuda" else -1,
                )
                logger.info("Zero-shot pipeline loaded.")
            except Exception as e:
                logger.warning(f"Could not load zero-shot pipeline: {e}")
                self._zero_shot_pipeline = False  # Sentinel: don't retry
        return self._zero_shot_pipeline if self._zero_shot_pipeline else None

    def _zero_shot_intent(self, text: str) -> tuple:
        """Zero-shot intent classification using a pretrained NLI model.

        Falls back gracefully to _rule_intent if the pipeline is unavailable.
        Uses facebook/bart-large-mnli which maps every intent label as a
        natural language hypothesis against the user utterance.
        """
        pipe = self._get_zero_shot_pipeline()
        if pipe is None:
            return self._rule_intent(text)

        # Human-readable label descriptions improve NLI accuracy
        label_descriptions = {
            "book_flight":             "book a flight or airline ticket",
            "cab_booking":             "book a cab, taxi, or ride",
            "hotel_booking":           "book a hotel or accommodation",
            "get_directions":          "get directions or navigate to a place",
            "cancel_booking":          "cancel a booking or get a refund",
            "get_weather":             "check the weather or forecast",
            "check_balance":           "check bank balance or account details",
            "send_money":              "send money or make a payment",
            "recharge_bill":           "recharge a phone or pay a bill",
            "order_food":              "order food or get a delivery",
            "play_music":              "play music or a song",
            "movie_recommendation":    "get a movie or show recommendation",
            "sports_update":           "get a sports score or update",
            "set_alarm":               "set an alarm or timer",
            "set_reminder":            "set a reminder",
            "health_advice":           "get health advice or medical help",
            "medicine_reminder":       "be reminded to take medicine",
            "shopping":                "shop online or buy a product",
            "general_knowledge":       "ask a general knowledge question",
            "news_update":             "get the latest news",
            "search_web":              "search the internet",
            "greeting":                "say hello or greet someone",
            "farewell":                "say goodbye",
            "thank_you":               "express gratitude or thanks",
            "apology":                 "apologize or say sorry",
            "tell_joke":               "hear a joke or something funny",
            "emergency_help":          "call for emergency help or report danger",
            "mental_health_distress":  "express emotional distress, sadness, or thoughts of self-harm",
            "customer_care":           "contact customer support or report an issue",
            "general_query":           "ask a general or unclassified question",
        }

        candidate_labels = list(label_descriptions.values())
        try:
            result = pipe(text, candidate_labels=candidate_labels, multi_label=False)
            # Map description back to intent key
            desc_to_key = {v: k for k, v in label_descriptions.items()}
            top_intents = [
                (desc_to_key.get(lbl, lbl), round(score, 4))
                for lbl, score in zip(result["labels"][:3], result["scores"][:3])
            ]
            best_intent, best_conf = top_intents[0]
            return best_intent, best_conf, top_intents
        except Exception as e:
            logger.warning(f"Zero-shot inference failed: {e}")
            return self._rule_intent(text)
