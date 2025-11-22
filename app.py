"""
CT Learner - Emotion + Paul’s Critical Thinking Rubric Analyzer
Single-file Streamlit app.

Features:
- Upload TXT / PDF / DOCX student submissions (batch)
- Transformer-based emotion inference (Hugging Face model)
- Rule-based Ekman+Shame/Pride detection with lexicons, negation & amplifiers
- Fusion of model and rule-based scores
- Sentence-level explanations and trigger highlights
- Paul’s Critical Thinking rubric automatic heuristic scorer + suggestions
- Export results to CSV / XLSX
- Defensive programming & user-friendly feedback

Notes:
- Google Docs extraction is a placeholder (requires OAuth + google-api libraries).
- Default HF model is configurable. For large models, prefer a GPU instance or smaller model.
"""

import os
import io
import re
import json
import math
import tempfile
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import numpy as np

# File extraction
import docx
import pdfplumber

# NLP & HF
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Utilities
from datetime import datetime

# ---------------------
# Configuration & Lexica
# ---------------------
DEFAULT_HF_MODEL = "j-hartmann/emotion-english-roberta-large"  # change if you need smaller model
EKMAN_PLUS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "shame", "pride"]
# Minimal rule lexicons (expand/replace with domain-specific lists)
EKMAN_CUES = {
    "anger": ["angry", "furious", "annoy", "rage", "irritat", "outrag", "resent"],
    "disgust": ["disgust", "disgusted", "gross", "revolting", "repuls", "nausea"],
    "fear": ["afraid", "fear", "scared", "terrify", "anxious", "panic", "nervou", "worried"],
    "joy": ["happy", "joy", "delight", "pleased", "glad", "excited", "elated", "satisfied"],
    "sadness": ["sad", "depress", "unhappy", "sorrow", "grief", "mourn", "disappoint"],
    "surprise": ["surpris", "astonish", "startl", "shocked", "unexpected"],
    "shame": ["ashamed", "shame", "embarrass", "humiliat", "guilty"],
    "pride": ["proud", "pride", "accomplish", "achievement", "succeeded", "confident"],
}
AMPLIFIERS = ["very", "extremely", "absolutely", "incredibly", "so", "really", "totally", "deeply"]
NEGATIONS = ["not", "never", "no", "n't", "hardly", "scarcely", "rarely", "none"]

# ---------------------
# Paul’s Critical Thinking Rubric (source text integrated)
# ---------------------
PAUL_CT_RUBRIC = {
    "Clarity": {
        "description": "Demonstrate clarity in conversation; provide examples to illustrate the point as appropriate.",
        "feedback_q": "Could you elaborate further; give an example or illustrate what you mean?"
    },
    "Accuracy": {
        "description": "Provide accurate and verifiable information to support the ideas/position.",
        "feedback_q": "How could we check on that; verify or test; find out if that is true?"
    },
    "Relevance": {
        "description": "Respond to the issues/question/problem with related information. Avoid irrelevant details.",
        "feedback_q": "How does that relate to the problem; bear on the question; help us with the issue?"
    },
    "Significance": {
        "description": "Able to identify the central idea. Contribute with important and new points.",
        "feedback_q": "Is this the most important problem to consider? Which of these facts are most important?"
    },
    "Logic": {
        "description": "Organize each piece of information in a logical order so it makes sense to others.",
        "feedback_q": "Does all this make sense together? Does what you say follow from the evidence?"
    },
    "Precision": {
        "description": "Select specific information, stay focused and avoid redundancy.",
        "feedback_q": "Could you be more specific; be more exact; give more details?"
    },
    "Fairness": {
        "description": "Demonstrate open-mindedness, consider pros and cons and challenge assumptions.",
        "feedback_q": "Am I sympathetically representing the viewpoints of others? Do I have vested interests?"
    },
    "Depth": {
        "description": "Being thorough; examine the intricacies in the argument.",
        "feedback_q": "What are some of the complexities of this question? What difficulties must we deal with?"
    },
    "Breadth": {
        "description": "Able to offer / consider alternative views or solutions.",
        "feedback_q": "Do we need another perspective? What are alternative ways?"
    }
}

# ---------------------
# Helper: extraction functions
# ---------------------
def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def extract_text_from_docx_bytes(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
        f.write(b); f.flush()
        tmp = f.name
    try:
        doc = docx.Document(tmp)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

def extract_text_from_pdf_bytes(b: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b); f.flush()
        tmp = f.name
    try:
        text_pages = []
        with pdfplumber.open(tmp) as pdf:
            for p in pdf.pages:
                text_pages.append(p.extract_text() or "")
        return "\n".join(text_pages)
    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

# Google Doc placeholder (requires user setup)
def extract_text_from_gdoc(doc_id: str, creds_path: str = None) -> str:
    raise NotImplementedError("Google Docs extraction requires Google API credentials and setup. See README or comments.")

# ---------------------
# Preprocessing
# ---------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)  # remove zero-width
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_split(text: str) -> List[str]:
    # basic sentence splitter using punctuation; can replace with spacy for better results
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"\w+['-]?\w*|\w+", s.lower())

# ---------------------
# Rule-based Ekman scorer
# ---------------------
def get_context(tokens: List[str], idx: int, window=3) -> List[str]:
    return tokens[max(0, idx-window): min(len(tokens), idx+window+1)]

def rule_score_text(text: str) -> Tuple[Dict[str, float], List[Tuple[str, float, str]]]:
    """
    Returns:
      - normalized scores dict for EKMAN_PLUS keys
      - triggers: list of tuples (emotion, weight, sentence)
    """
    scores = Counter()
    triggers = []
    sents = sentence_split(text)
    for sent in sents:
        tokens = tokenize_simple(sent)
        for emo, cues in EKMAN_CUES.items():
            for i, tok in enumerate(tokens):
                for cue in cues:
                    if tok.startswith(cue):
                        weight = 1.0
                        ctx = get_context(tokens, i, window=3)
                        if any(a in ctx for a in AMPLIFIERS):
                            weight *= 1.8
                        if any(n in ctx for n in NEGATIONS):
                            weight *= -0.8
                        scores[emo] += weight
                        triggers.append((emo, weight, sent))
    # normalize to 0..1 using absolute max
    if scores:
        maxabs = max(abs(v) for v in scores.values())
        normalized = {k: float(scores.get(k, 0.0)) / (maxabs if maxabs>0 else 1.0) for k in EKMAN_PLUS}
    else:
        normalized = {k: 0.0 for k in EKMAN_PLUS}
    return normalized, triggers

# ---------------------
# Hugging Face model wrapper (cached)
# ---------------------
@st.cache_resource
def load_transformer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # attempt to move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return {"tok": tokenizer, "model": model, "device": device}
    except Exception as e:
        st.error(f"Failed to load model '{model_name}': {e}")
        raise

def model_predict_texts(texts: List[str], tok, model, device, batch_size=16) -> List[Dict[str, float]]:
    """
    Returns list of dicts mapping label -> probability.
    Handles single-label (softmax) and multi-label (sigmoid) heuristically.
    """
    results = []
    if not texts:
        return results
    model.eval()
    num = len(texts)
    for i in range(0, num, batch_size):
        batch = texts[i: i+batch_size]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.detach().cpu()
            # determine if multi-label: heuristically if more than len(EKMAN_PLUS) labels OR config
            is_multi_label = (getattr(model.config, "problem_type", "") == "multi_label_classification") or (model.config.num_labels > len(EKMAN_PLUS) + 2)
            if is_multi_label:
                probs = torch.sigmoid(logits).numpy()
            else:
                probs = torch.softmax(logits, dim=-1).numpy()
        # map labels
        id2label = getattr(model.config, "id2label", None)
        for p in probs:
            if id2label:
                label_map = [id2label[i].lower() for i in range(len(p))]
                prob_dict = dict(zip(label_map, p.tolist()))
            else:
                prob_dict = {f"label_{i}": float(val) for i, val in enumerate(p.tolist())}
            # attempt to map to EKMAN_PLUS: take max matching synonyms, else 0
            mapped = {}
            synonyms = {
                "joy": ["joy", "happiness", "happy"],
                "fear": ["fear", "afraid", "anxiety"],
                "sadness": ["sad", "sadness", "sorrow"],
                "anger": ["anger", "angry", "annoy"],
                "disgust": ["disgust", "disgusted"],
                "surprise": ["surprise", "surprised", "astonish"],
                "shame": ["shame", "ashamed", "guilt", "guilty"],
                "pride": ["pride", "proud", "accomplish"]
            }
            lower_prob = {k.lower(): float(v) for k, v in prob_dict.items()}
            for e in EKMAN_PLUS:
                best = 0.0
                for syn in synonyms.get(e, [e]):
                    best = max(best, lower_prob.get(syn, 0.0))
                mapped[e] = float(best)
            results.append(mapped)
    return results

# ---------------------
# Fusion
# ---------------------
def fuse_scores(model_scores: Dict[str, float], rule_scores: Dict[str, float], model_w: float = 0.6) -> Dict[str, float]:
    rule_w = 1.0 - model_w
    keys = set(model_scores.keys()) | set(rule_scores.keys())
    fused = {}
    for k in keys:
        fused[k] = model_w * float(model_scores.get(k, 0.0)) + rule_w * float(rule_scores.get(k, 0.0))
    return fused

# ---------------------
# Paul CT Rubric heuristic scorer
# ---------------------
def heuristic_ct_scores(text: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Returns:
      - scores: 0..1 for each CT standard
      - suggestions: feedback question for each standard (text)
    Heuristics are simple and explainable; replace/add teacher rules as needed.
    """
    sents = sentence_split(text)
    tokens = tokenize_simple(text)
    word_count = len(tokens)
    scores = {}
    suggestions = {}
    # Clarity: measured by presence of examples/illustrations (e.g., "for example", "for instance", "e.g.", "such as")
    clarity_indicators = ["for example", "for instance", "e.g.", "such as", "to illustrate"]
    clarity_score = 0.0
    if any(phrase in text.lower() for phrase in clarity_indicators):
        clarity_score = 1.0
    else:
        # if short text (<50 words) clarity may be lower
        clarity_score = 0.3 if word_count < 50 else 0.5
    scores["Clarity"] = clarity_score
    suggestions["Clarity"] = PAUL_CT_RUBRIC["Clarity"]["feedback_q"]

    # Accuracy: presence of references, numbers, dates, or verifiable claims (heuristic)
    accuracy_indicators = ["http", "www.", "cite", "according to", "%", "data", "study", "reported", "survey"]
    accuracy_score = 1.0 if any(ind in text.lower() for ind in accuracy_indicators) else 0.4
    scores["Accuracy"] = accuracy_score
    suggestions["Accuracy"] = PAUL_CT_RUBRIC["Accuracy"]["feedback_q"]

    # Relevance: fraction of sentences mentioning main topic phrase (approximate: first sentence nouns)
    # Heuristic: consider overlap with first sentence (assumed thesis)
    if sents:
        first = tokenize_simple(sents[0])
        overlap_counts = sum(1 for sent in sents[1:] if any(w in tokenize_simple(sent) for w in first[:5]))
        relevance_score = min(1.0, (overlap_counts+1) / max(1, len(sents)))
    else:
        relevance_score = 0.0
    scores["Relevance"] = relevance_score
    suggestions["Relevance"] = PAUL_CT_RUBRIC["Relevance"]["feedback_q"]

    # Significance: presence of central idea (short heuristic: text has a sentence with 'main' or 'central' or 'important' or 'key')
    sign_ind = ["main", "central", "important", "key", "primary"]
    sign_score = 1.0 if any(w in text.lower() for w in sign_ind) else min(0.9, 0.6 + 0.01 * (word_count/100))
    scores["Significance"] = sign_score
    suggestions["Significance"] = PAUL_CT_RUBRIC["Significance"]["feedback_q"]

    # Logic: cohesion markers and connectors presence (therefore, because, thus, hence, however, but)
    connectors = ["therefore", "because", "thus", "hence", "however", "but", "consequently", "as a result", "so that"]
    logic_score = min(1.0, sum(1 for c in connectors if c in text.lower()) * 0.25)
    scores["Logic"] = logic_score
    suggestions["Logic"] = PAUL_CT_RUBRIC["Logic"]["feedback_q"]

    # Precision: presence of exact terms, numbers, dates; less hedging words
    hedges = ["maybe", "perhaps", "might", "could", "seems", "appears"]
    precision_score = max(0.0, 1.0 - 0.2 * sum(1 for h in hedges if h in text.lower()))
    # Penalize extreme brevity
    if word_count < 40:
        precision_score *= 0.5
    scores["Precision"] = precision_score
    suggestions["Precision"] = PAUL_CT_RUBRIC["Precision"]["feedback_q"]

    # Fairness: presence of "on the other hand", "however", "although", "consider", "pros and cons"
    fairness_ind = ["on the other hand", "although", "consider", "pros and cons", "however", "both", "despite"]
    fairness_score = 1.0 if any(p in text.lower() for p in fairness_ind) else 0.45
    scores["Fairness"] = fairness_score
    suggestions["Fairness"] = PAUL_CT_RUBRIC["Fairness"]["feedback_q"]

    # Depth: count of subordinate clauses / complexity markers (because, although, since, whereas)
    depth_ind = ["because", "although", "since", "whereas", "in depth", "intricacy", "complex", "complexity"]
    depth_score = min(1.0, 0.25 * sum(1 for d in depth_ind if d in text.lower()) + 0.3)
    scores["Depth"] = depth_score
    suggestions["Depth"] = PAUL_CT_RUBRIC["Depth"]["feedback_q"]

    # Breadth: presence of alternatives/other views ("alternatively", "another view", "different perspective")
    breadth_ind = ["alternatively", "another view", "different perspective", "other view", "in contrast"]
    breadth_score = 1.0 if any(p in text.lower() for p in breadth_ind) else 0.4
    scores["Breadth"] = breadth_score
    suggestions["Breadth"] = PAUL_CT_RUBRIC["Breadth"]["feedback_q"]

    # Normalize scores to range 0..1 explicitly
    for k in scores:
        scores[k] = float(max(0.0, min(1.0, scores[k])))
    return scores, suggestions

# ---------------------
# Utility & UI helpers
# ---------------------
def safe_extract_all_files(files) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: {"filename": str, "text": str}
    """
    out = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        try:
            b = f.read()
            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf_bytes(b)
            elif name.lower().endswith(".docx"):
                text = extract_text_from_docx_bytes(b)
            else:
                # assume txt or fallback
                text = extract_text_from_txt_bytes(b)
            text = clean_text(text)
            if not text:
                st.warning(f"Warning: extracted empty text from {name}. If this is a scanned PDF, OCR is required.")
            out.append({"filename": name, "text": text})
        except Exception as e:
            st.error(f"Failed to extract {name}: {e}")
            out.append({"filename": name, "text": ""})
    return out

def validate_fused_results(fused_results):
    if fused_results is None:
        return False, "Pipeline returned None."
    if not isinstance(fused_results, list):
        return False, f"Expected list but got {type(fused_results)}"
    if len(fused_results) == 0:
        return False, "No results were produced."
    for idx, item in enumerate(fused_results):
        if not isinstance(item, (list, tuple)):
            return False, f"Item {idx} is not a tuple/list: {item}"
        if len(item) != 5:
            return False, f"Item {idx} has {len(item)} elements instead of 5: {item}"
    return True, None

def safe_json(d: Dict) -> str:
    try:
        return json.dumps(d, ensure_ascii=False)
    except Exception:
        return str(d)

# ---------------------
# Streamlit app UI
# ---------------------
st.set_page_config(page_title="CT Learner", layout="wide", initial_sidebar_state="expanded")
st.title("CT Learner — Emotions + Paul’s Critical Thinking Rubric")
st.markdown(
    "Upload student submissions (TXT, PDF, DOCX). The app runs a transformer-based emotion model, "
    "a rule-based Ekman-style detector (Ekman + shame/pride), fuses results, and computes a heuristic "
    "Paul CT Rubric score with feedback prompts."
)

# Sidebar controls
with st.sidebar:
    st.header("Upload & Settings")
    uploaded = st.file_uploader("Upload submissions (multiple allowed)", accept_multiple_files=True, type=['txt','pdf','docx'])
    st.markdown("---")
    model_name = st.text_input("Hugging Face model", value=DEFAULT_HF_MODEL)
    batch_size = st.number_input("Model batch size", value=8, min_value=1, max_value=64)
    model_weight = st.slider("Transformer weight in fusion (model vs rule)", min_value=0.0, max_value=1.0, value=0.6)
    st.markdown("Model device: " + ("GPU (cuda)" if torch.cuda.is_available() else "CPU"))
    run_btn = st.button("Analyze")

# Show rubric in sidebar (collapsible)
with st.sidebar.expander("Paul’s CT Rubric (summary)"):
    for k, v in PAUL_CT_RUBRIC.items():
        st.markdown(f"**{k}** — {v['description']}")
        st.caption(f"Feedback Q: {v['feedback_q']}")

# Main workflow
if run_btn:
    # Basic validation
    if not uploaded:
        st.error("Please upload at least one file.")
        st.stop()

    st.info(f"Processing {len(uploaded)} file(s)...")
    submissions = safe_extract_all_files(uploaded)

    # Load model (try/catch)
    try:
        hf = load_transformer(model_name)
        tokenizer = hf["tok"]; model = hf["model"]; device = hf["device"]
    except Exception:
        st.error("Transformer model failed to load. Please check model name or network.")
        st.stop()

    texts = [s["text"] for s in submissions]
    # model inference
    with st.spinner("Running transformer model inference..."):
        try:
            model_preds = model_predict_texts(texts, tokenizer, model, device, batch_size=int(batch_size))
        except Exception as e:
            st.error(f"Model inference failed: {e}")
            model_preds = [{k:0.0 for k in EKMAN_PLUS} for _ in texts]

    # rule-based inference
    with st.spinner("Running rule-based Ekman scorer..."):
        rule_outs = [rule_score_text(t) for t in texts]
        rule_scores = [r[0] for r in rule_outs]
        rule_triggers = [r[1] for r in rule_outs]

    # CT rubric scoring
    with st.spinner("Computing Paul CT rubric heuristics..."):
        ct_scores_all = []
        ct_suggestions_all = []
        for t in texts:
            s, sug = heuristic_ct_scores(t)
            ct_scores_all.append(s)
            ct_suggestions_all.append(sug)

    # Fuse and assemble results
    fused_results = []
    rows = []
    for meta, mscore, rscore, triggers, ct_scores, ct_suggest in zip(submissions, model_preds, rule_scores, rule_triggers, ct_scores_all, ct_suggestions_all):
        fused = fuse_scores(mscore, rscore, model_w=float(model_weight))
        row = {
            "filename": meta.get("filename", "untitled"),
            "model_conf": float(max(mscore.values())) if mscore else 0.0,
            "rule_conf": float(max(abs(v) for v in rscore.values())) if rscore else 0.0,
            "fused_conf": float(max(fused.values())) if fused else 0.0,
            "model_scores": safe_json(mscore),
            "rule_scores": safe_json(rscore),
            "fused_scores": safe_json(fused),
            "ct_scores": safe_json(ct_scores),
            "ct_suggestions": safe_json(ct_suggest),
            "text_preview": meta.get("text","")[:500]
        }
        rows.append(row)
        fused_results.append((meta, mscore, rscore, fused, triggers))

    # Validate fused_results before display
    ok, msg = validate_fused_results(fused_results)
    if not ok:
        st.error(f"Processing pipeline error: {msg}")
        st.stop()

    # Display summary table
    df_summary = pd.DataFrame(rows)
    st.subheader("Summary of analyzed submissions")
    st.dataframe(df_summary[["filename", "model_conf", "rule_conf", "fused_conf", "text_preview"]])

    # Export options
    csv_bytes = df_summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name=f"ctlearner_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    # Excel export
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="results")
    st.download_button("Download XLSX", data=towrite.getvalue(), file_name=f"ctlearner_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Detailed per-document view
    st.markdown("---")
    st.header("Detailed analysis")
    for i, (meta, mscore, rscore, fused, triggers) in enumerate(fused_results):
        st.subheader(f"{i+1}. {meta.get('filename','untitled')}")
        left, right = st.columns([2,1])
        with left:
            st.markdown("**Extracted text (preview)**")
            txt = meta.get("text","")
            if len(txt) > 1000:
                st.write(txt[:1000] + " ...")
                if st.checkbox(f"Show full text for {meta.get('filename')}", key=f"full_{i}"):
                    st.write(txt)
            else:
                st.write(txt or "*No text extracted.*")

            st.markdown("**Rule-based emotion triggers (sentence -> weight)**")
            if triggers:
                for emo, w, s in triggers[:20]:
                    st.markdown(f"- **{emo}** ({w:.2f}): {s}")
            else:
                st.write("No rule triggers found.")

        with right:
            st.markdown("**Transformer (mapped) scores**")
            st.table(pd.DataFrame.from_dict(mscore, orient="index", columns=["probability"]).sort_values("probability", ascending=False))
            st.markdown("**Rule-based scores**")
            st.table(pd.DataFrame.from_dict(rscore, orient="index", columns=["score"]).sort_values("score", ascending=False))
            st.markdown("**Fused scores**")
            st.table(pd.DataFrame.from_dict(fused, orient="index", columns=["score"]).sort_values("score", ascending=False))
            st.markdown("**Paul CT Rubric (heuristic scores)**")
            ct_scores = ct_scores_all[i]
            ct_sugs = ct_suggestions_all[i]
            st.table(pd.DataFrame.from_dict(ct_scores, orient="index", columns=["score"]).round(3))
            st.markdown("**Feedback suggestions (from rubric)**")
            for k, q in ct_sugs.items():
                st.markdown(f"- **{k}**: {q}")

    st.success("Analysis complete ✅")
