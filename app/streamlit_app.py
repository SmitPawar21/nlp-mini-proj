"""
Streamlit Web Application — Unified Code-Mixed NLP Demonstration Interface.
Clean, white, professional research-grade UI for Hinglish Language Understanding.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.inference.predictor import UnifiedPredictor
from src.preprocessing.normalizer import CodeMixedNormalizer
from src.utils.lang_utils import detect_script

# ---------------------------------------------------------------------------
# Page Configuration & Clean White Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hinglish NLP Toolkit",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Base Theme: Clean White, Neutral Slate Typography */
:root {
    --bg-page: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --card-bg: #ffffff;
}

body, .stApp {
    background-color: #ffffff;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

/* Typography */
.header-container {
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
}

.main-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.025em;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 400;
}

/* Clean Cards */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.result-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.75rem 1rem;
}

.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.15rem;
    color: #0f172a;
    font-weight: 600;
}

/* Status Chips / Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.825rem;
    font-weight: 600;
}

.badge-positive {
    background-color: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.badge-negative {
    background-color: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

.badge-neutral {
    background-color: #f8fafc;
    color: #475569;
    border: 1px solid #e2e8f0;
}

.badge-intent {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

.entity-chip {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.15rem;
}

.entity-per { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
.entity-loc { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.entity-org { background: #faf5ff; color: #6b21a8; border: 1px solid #e9d5ff; }
.entity-misc { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }

/* Buttons & Inputs */
.stButton > button {
    background-color: #0f172a;
    color: #ffffff;
    border-radius: 6px;
    border: 1px solid #0f172a;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.55rem 1.25rem;
    transition: background-color 0.15s ease-in-out;
}

.stButton > button:hover {
    background-color: #1e293b;
    border-color: #1e293b;
    color: #ffffff;
}

.stTextArea textarea {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    font-size: 0.925rem;
    color: #0f172a;
}

.stTextArea textarea:focus {
    border-color: #0f172a;
    box-shadow: 0 0 0 1px #0f172a;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Settings")

lang_pair = st.sidebar.selectbox(
    "Language Pair",
    ["hi-en (Hindi-English)", "ta-en (Tamil-English) [Roadmap]"],
    index=0,
)
lang_code = "hi-en" if "hi-en" in lang_pair else "ta-en"

task_choice = st.sidebar.radio(
    "Select NLP Task",
    [
        "Sentiment Analysis",
        "Intent Detection",
        "Translation",
        "Named Entity Recognition (NER)",
        "Question Answering (QA)",
    ],
    index=0,
)

translation_dir = "cm_to_en"
if task_choice == "Translation":
    dir_choice = st.sidebar.radio(
        "Translation Direction",
        ["Code-Mixed (Hinglish) -> English", "English -> Code-Mixed Hindi"],
        index=0,
    )
    translation_dir = "cm_to_en" if "English" in dir_choice.split("->")[1] else "en_to_cm"

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #64748b; line-height: 1.5;">
<strong>Architecture Overview</strong><br>
Fine-tuned MuRIL transformer backbones with domain-adapted fallback classifiers for robust Hindi-English code-mixed natural language understanding.
</div>
""", unsafe_allow_html=True)

TASK_MAP = {
    "Translation": "translation",
    "Sentiment Analysis": "sentiment",
    "Named Entity Recognition (NER)": "ner",
    "Question Answering (QA)": "qa",
    "Intent Detection": "intent",
}
selected_task = TASK_MAP[task_choice]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="main-title">Hinglish NLP Toolkit</div>
    <div class="sub-title">Unified Language Understanding & Generation for Hindi-English Code-Mixed Text</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached Predictor & Normalizer
# ---------------------------------------------------------------------------
@st.cache_resource
def get_predictor(lang: str) -> UnifiedPredictor:
    return UnifiedPredictor(lang_pair=lang)

predictor = get_predictor(lang_code)
normalizer = CodeMixedNormalizer(lang_pair=lang_code)

# ---------------------------------------------------------------------------
# Curated Sample Inputs
# ---------------------------------------------------------------------------
SAMPLES = {
    "Sentiment Analysis": [
        "Yeh movie bahut achhi thi, casting aur direction mind blowing tha!",
        "Khaana ekdum bekar tha, waste of money and time.",
        "Flight timing change ho gaya, okayish service.",
        "Product ki quality bahut achhi hai, highly recommended!",
        "Worst customer care ever, phone bhi attend nahi karte.",
        "Average movie thi, ek baar dekh sakte hain.",
    ],
    "Intent Detection": [
        "Mujhe Delhi se Mumbai ki flight ticket book karni hai.",
        "Aaj Bangalore me mausam kaisa hai, barish hogi kya?",
        "Mere account ka balance kitna hai check karo.",
        "Ek pizza order karo Dominos se dinner ke liye.",
        "Kal subah 6 baje ka alarm set karo.",
        "Meri flight booking cancel kardo refund chahiye.",
        "Arijit Singh ke romantic gaane bajao Spotify pe.",
        "Airport pahunchne ka sabse fastest rasta batao.",
    ],
    "Translation": {
        "cm_to_en": [
            "Aaj shaam ko hum sab friends milke party karenge.",
            "Delhi me weather today kafi pleasant hai.",
            "Yeh movie bohot acchi thi, direction mind blowing tha!",
            "IPL ka match Delhi Capitals aur Rajasthan Royals ke beech khela gaya.",
            "Mujhe Delhi se Mumbai ki flight ticket book karni hai.",
        ],
        "en_to_cm": [
            "The weather in Delhi is very pleasant today.",
            "I want to book a flight ticket from Delhi to Mumbai.",
            "This movie was really good, the direction was amazing!",
            "The IPL match was played between Delhi Capitals and Rajasthan Royals.",
        ],
    },
    "Named Entity Recognition (NER)": [
        "Shah Rukh Khan Mumbai me Mannat me rehte hain.",
        "Google ne Bangalore me naya R&D office open kiya hai.",
        "Narendra Modi Prime Minister of India ne Delhi me speech di.",
        "Virat Kohli ne BCCI ko letter likha hai.",
        "Infosys aur TCS ne Pune campus expand kiya.",
    ],
    "Question Answering (QA)": [
        {
            "question": "Shah Rukh Khan kahan rehte hain?",
            "context": "Shah Rukh Khan Indian film actor hain. Woh Mumbai me Mannat house me rehte hain. Unhone bohot saari superhit movies me acting ki hai.",
        },
        {
            "question": "Google ne naya office kahan khola hai?",
            "context": "Google ne Bangalore me naya R&D office open kiya hai jisme advanced machine learning systems develop honge.",
        },
        {
            "question": "Virat Kohli ne kisko letter likha?",
            "context": "Virat Kohli ne BCCI ko letter likha hai jisme unhone upcoming test matches ke preparation par discuss kiya.",
        },
        {
            "question": "Match kahan khela gaya tha?",
            "context": "IPL ka match Mumbai ke Wankhede stadium me khela gaya tha aur stadium spectators se full tha.",
        },
        {
            "question": "Match kisne jeeta?",
            "context": "Kal raat ke thrilling match me India ne Australia ko 5 wickets se hara kar match jeet liya.",
        },
        {
            "question": "Meeting kab shuru hogi?",
            "context": "Kal subah 10 baje conference hall me project ki urgent meeting shuru hogi.",
        },
        {
            "question": "Kitne baje train aayegi?",
            "context": "Rajdhani express shaam ko 6 baje platform number 4 par aayegi.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Main Layout: Two Column
# ---------------------------------------------------------------------------
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### Input Text")

    if selected_task == "qa":
        qa_samples = SAMPLES["Question Answering (QA)"]
        qa_options = ["Custom input..."] + [f"{item['question']}" for item in qa_samples]

        selected_qa_idx = st.selectbox(
            "Select benchmark QA pair or enter custom input:",
            range(len(qa_options)),
            format_func=lambda i: qa_options[i],
            index=1,
            key="qa_preset_select",
        )

        if selected_qa_idx == 0:
            default_q = ""
            default_ctx = ""
        else:
            default_q = qa_samples[selected_qa_idx - 1]["question"]
            default_ctx = qa_samples[selected_qa_idx - 1]["context"]

        user_text = st.text_area(
            "Question (Code-Mixed / Hinglish):",
            value=default_q,
            height=85,
            placeholder="Enter question here (e.g. Match kahan khela gaya tha?)...",
            key=f"qa_question_input_{selected_qa_idx}",
        )
        context_text = st.text_area(
            "Reference Context Passage:",
            value=default_ctx,
            height=120,
            placeholder="Enter context passage here...",
            key=f"qa_context_input_{selected_qa_idx}",
        )
    else:
        if selected_task == "translation":
            sample_list = SAMPLES["Translation"].get(translation_dir, [])
        else:
            sample_list = SAMPLES.get(task_choice, [])

        options = ["Custom input..."] + sample_list
        selected_idx = st.selectbox(
            "Select benchmark example or enter custom input:",
            range(len(options)),
            format_func=lambda i: options[i],
            index=1 if len(options) > 1 else 0,
            key=f"preset_select_{selected_task}",
        )

        if selected_idx == 0:
            default_val = ""
        else:
            default_val = options[selected_idx]

        if selected_task == "translation" and translation_dir == "en_to_cm":
            input_label = "English Input"
            placeholder = "Enter standard English text here..."
        else:
            input_label = "Code-Mixed (Hinglish) Input"
            placeholder = "Enter Hindi-English code-mixed text here..."

        user_text = st.text_area(
            input_label,
            value=default_val,
            height=110,
            placeholder=placeholder,
            key=f"text_input_{selected_task}_{selected_idx}",
        )
        context_text = ""

    submit = st.button("Run Inference", type="primary", use_container_width=True)

    # Preprocessing telemetry
    if user_text:
        cleaned = normalizer.normalize(user_text)
        script = detect_script(cleaned)
        
        st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### Linguistic Analysis")
        
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Dominant Script</div>
                <div class="metric-value">{script.dominant_script.capitalize()}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Script Mix</div>
                <div class="metric-value">{"Mixed" if script.is_mixed_script else "Single"}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Mix Ratio</div>
                <div class="metric-value">{script.mix_ratio:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Normalized Tokens", expanded=False):
            st.code(f"Original  : {user_text}\nNormalized: {cleaned}", language="text")

with col_output:
    st.markdown("#### Model Predictions")

    if submit and user_text:
        with st.spinner("Processing inference pipeline..."):
            result = predictor.predict(
                text=user_text,
                task=selected_task,
                context=context_text if selected_task == "qa" else None,
                translation_direction=translation_dir,
            )

        model_type = result.get("model_type", "transformer")

        # 1. Sentiment Analysis
        if selected_task == "sentiment":
            sentiment = result["sentiment"].capitalize()
            conf = result["confidence"]
            badge_class = f"badge-{sentiment.lower()}"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">Classification Result</span>
                    <span class="badge {badge_class}">{sentiment}</span>
                </div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;">
                    Predicted Label: {sentiment}
                </div>
                <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.75rem;">
                    Model Confidence: <strong>{conf:.1%}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(conf)

        # 2. Intent Detection
        elif selected_task == "intent":
            intent = result["intent"]
            conf = result["confidence"]
            top_k = result.get("top_k", [])
            model_type_intent = result.get("model_type", "")

            # --- Mental health crisis banner ---
            if intent == "mental_health_distress":
                st.markdown("""
                <div style="background: #fef2f2; border: 2px solid #f87171; border-radius: 10px;
                            padding: 1rem 1.25rem; margin-bottom: 1rem;">
                    <div style="font-size: 1rem; font-weight: 700; color: #b91c1c; margin-bottom: 0.4rem;">
                        🆘 Crisis Support Detected
                    </div>
                    <div style="font-size: 0.9rem; color: #7f1d1d; line-height: 1.6;">
                        It sounds like you may be going through a very difficult time.
                        You are not alone — please reach out for help:<br><br>
                        <strong>iCall (India):</strong> 9152987821<br>
                        <strong>Vandrevala Foundation:</strong> 1860-2662-345 (24/7)<br>
                        <strong>AASRA:</strong> 9820466627<br>
                        <strong>International:</strong> <a href="https://www.befrienders.org" target="_blank">befrienders.org</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- Threat / Violence warning banner ---
            if intent == "threat_violence":
                st.markdown("""
                <div style="background: #fff1f2; border: 2px solid #f43f5e; border-radius: 10px;
                            padding: 1rem 1.25rem; margin-bottom: 1rem;">
                    <div style="font-size: 1rem; font-weight: 700; color: #be123c; margin-bottom: 0.4rem;">
                        ⚠️ Violence / Threat Detected
                    </div>
                    <div style="font-size: 0.9rem; color: #881337; line-height: 1.6;">
                        This utterance expresses intent involving physical harm, violence, or aggression towards another person.<br>
                        <strong>Category:</strong> <code>threat_violence</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- No intent detected card ---
            if intent == "no_intent_detected":
                st.markdown("""
                <div style="border: 1.5px dashed #cbd5e1; border-radius: 10px;
                            padding: 1.25rem 1.5rem; background: #f8fafc; margin-bottom: 0.75rem;">
                    <div style="font-size: 1rem; font-weight: 700; color: #64748b; margin-bottom: 0.4rem;">
                        🤷 No Intent Detected
                    </div>
                    <div style="font-size: 0.875rem; color: #94a3b8; line-height: 1.6;">
                        The input did not match any of the 33 supported intent categories
                        with sufficient confidence.<br><br>
                        <strong>Tip:</strong> Try being more specific — e.g.,
                        <em>"Mujhe flight book karni hai"</em> or <em>"Aaj ka mausam kaisa hai?"</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # --- Intent result card (normal) ---
                llm_badge = ""
                if model_type_intent == "zero_shot_llm":
                    llm_badge = "<span style='font-size:0.7rem; background:#e0e7ff; color:#3730a3; padding:2px 8px; border-radius:99px; font-weight:600; margin-left:8px;'>⚡ Zero-Shot LLM</span>"
                elif model_type_intent == "safety_system":
                    llm_badge = "<span style='font-size:0.7rem; background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:99px; font-weight:600; margin-left:8px;'>🛡️ Safety System</span>"

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">
                        <span class="result-title">Identified Intent {llm_badge}</span>
                        <span class="badge badge-intent">{intent}</span>
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;">
                        Primary Intent: <code>{intent}</code>
                    </div>
                    <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.5rem;">
                        Confidence: <strong>{conf:.1%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if top_k:
                    st.markdown("<div style='font-size: 0.825rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin: 0.75rem 0 0.25rem 0;'>Candidate Distribution</div>", unsafe_allow_html=True)
                    for cand_intent, cand_score in top_k:
                        c1, c2 = st.columns([3, 1])
                        c1.markdown(f"`{cand_intent}`")
                        c2.markdown(f"**{cand_score:.1%}**")

            with st.expander("📋 All Supported Intents (33)", expanded=False):
                intent_list = [
                    "affirmation", "apology", "book_flight", "cab_booking", "cancel_booking",
                    "check_balance", "customer_care", "emergency_help", "farewell", "general_knowledge",
                    "get_directions", "get_weather", "greeting", "health_advice", "hotel_booking",
                    "medicine_reminder", "mental_health_distress", "movie_recommendation", "negation",
                    "news_update", "no_intent_detected", "order_food", "play_music", "recharge_bill",
                    "search_web", "send_money", "set_alarm", "set_reminder", "shopping",
                    "sports_update", "tell_joke", "thank_you", "threat_violence",
                ]
                cols = st.columns(3)
                for i, lbl in enumerate(intent_list):
                    cols[i % 3].markdown(f"• `{lbl}`")

        # 3. Translation
        elif selected_task == "translation":
            direction = result.get("direction", "")
            translated = result.get("translation", "")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">Translation Direction</span>
                    <span class="badge badge-neutral">{direction}</span>
                </div>
                <div style="font-size: 1.05rem; color: #0f172a; line-height: 1.6; margin-top: 0.5rem; padding: 0.75rem; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;">
                    {translated}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 4. Named Entity Recognition
        elif selected_task == "ner":
            entities = result.get("entities", [])
            st.markdown("""
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">Detected Entities</span>
                </div>
            """, unsafe_allow_html=True)

            if not entities:
                st.info("No named entities detected in input.")
            else:
                for ent in entities:
                    ent_type = ent["entity"].lower()
                    chip_cls = f"entity-{ent_type}"
                    st.markdown(f"""
                    <div style="margin-bottom: 0.4rem;">
                        <span class="entity-chip {chip_cls}">{ent['entity']}</span>
                        <strong style="color: #0f172a;">{ent['word']}</strong>
                        <span style="font-size: 0.8rem; color: #64748b; margin-left: 0.5rem;">({ent['score']:.1%})</span>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 5. Question Answering
        elif selected_task == "qa":
            answer = result.get("answer", "")
            conf = min(1.0, max(0.0, float(result.get("confidence", 0.0))))
            ctx = result.get("context", context_text)

            highlighted_ctx = ctx
            if answer and answer in ctx:
                highlighted_ctx = ctx.replace(
                    answer,
                    f"<mark style='background-color: #fef08a; color: #854d0e; padding: 2px 6px; border-radius: 4px; font-weight: 600;'>{answer}</mark>",
                    1
                )

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <span class="result-title">Extracted Answer</span>
                    <span class="badge badge-positive">{conf:.1%} Match</span>
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin: 0.5rem 0;">
                    {answer if answer else '<span style="color: #94a3b8; font-weight: normal;">No clear answer span identified</span>'}
                </div>
                <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 0.75rem;">
                    Extraction Confidence: <strong>{conf:.1%}</strong>
                </div>
                <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-top: 0.75rem; margin-bottom: 0.25rem;">
                    Context Verification:
                </div>
                <div style="font-size: 0.95rem; line-height: 1.6; color: #334155; padding: 0.75rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;">
                    {highlighted_ctx}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(conf)

        # Telemetry footer
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.5rem; text-align: right;">
            Model Engine: <code>{model_type}</code> | Device: <code>{predictor.device or 'auto'}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 1px dashed #cbd5e1; border-radius: 8px; padding: 2.5rem 1rem; text-align: center; color: #94a3b8; font-size: 0.9rem;">
            Select a sample input or type your own text, then click <strong>Run Inference</strong>.
        </div>
        """, unsafe_allow_html=True)
