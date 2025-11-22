# app.py - CT Learner (single-file Streamlit app)
# Python 3.10+
# Author: CT Learner template (adapt/extend for your deployment)
# Notes:
# - Uses j-hartmann/emotion-english-roberta-large as default transformer.
# - Rule-based Ekman implementation with cue lists inline (expandable).
# - Batch inference, GPU if available via torch.
# - Exports CSV/XLSX and uses Streamlit for UI.

import io
import os
import re
import json
import math
import tempfile
from typing import List, Dict, Tuple
from functools import lru_cache

import streamlit as st
import pandas as pd
import numpy as np

# Text extraction
import docx
import pdfplumber

# NLP + Model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Utilities
from collections import Counter, defaultdict
from datetime import datetime
from openpyxl import Workbook

# ---------- Configuration ----------
HF_MODEL = "j-hartmann/emotion-english-roberta-large"  # recommended default
EKMAN_LABELS = ["anger","disgust","fear","joy","sadness","surprise","neutral"]  # j-hartmann labels
# We'll augment with shame/pride for rule-system (user requested)
EKMAN_PLUS = ["anger","disgust","fear","joy","sadness","surprise","shame","pride"]

# ---------- Lexicons for rule-based Ekman detection ----------
# Minimal cue lists; expand these with domain knowledge or external lexica.
# These are illustrative, focused on student essay language. We include stem-like patterns.
EKMAN_CUES = {
    "anger": ["angry","furious","annoyed","rage","irritat","outrag","resent"],
    "disgust": ["disgust","disgusted","gross","revolting","repuls","nausea"],
    "fear": ["afraid","fear","scared","terrify","anxious","panic","nervou","worried"],
    "joy": ["happy","joy","delight","pleased","glad","excited","elated","satisfied"],
    "sadness": ["sad","depress","unhappy","sorrow","grief","mourn","disappoint"],
    "surprise": ["surpris","astonish","startl","shocked","unexpected"],
    "shame": ["ashamed","shame","embarrass","humiliat","guilty"],
    "pride": ["proud","pride","accomplish","achievement","succeeded","confident"]
}

# Intensifiers (amplifiers) and negations
AMPLIFIERS = ["very","extremely","absolutely","incredibly","so","really","totally"]
NEGATIONS = ["not","never","no","n't","hardly","scarcely","rarely"]

# ---------- Helper functions: Text extraction ----------
def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return file_bytes.decode('latin-1', errors='ignore')

def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    doc = docx.Document(tmp_path)
    paragraphs = [p.text for p in doc.paragraphs]
    os.unlink(tmp_path)
    return "\n".join(paragraphs)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_chunks = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes); tmp.flush(); tmp_path = tmp.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text_chunks.append(page.extract_text() or "")
    finally:
        os.unlink(tmp_path)
    return "\n".join(text_chunks)

# Google Docs extraction placeholder: requires OAuth2 credentials and Drive/Docs API client setup
def extract_text_from_gdoc(doc_id: str, creds_json_path: str) -> str:
    """
    Placeholder: to enable, create Google Cloud Service Account or OAuth credentials,
    install google-api-python-client and google-auth, then implement docs_service.documents().get().
    For security reasons we don't embed keys here.
    """
    raise NotImplementedError("Google Docs extraction requires Google API setup. See README for steps.")

# ---------- Preprocessing ----------
def clean_text(text: str) -> str:
    # normalize whitespace, remove weird control chars, keep sentences
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\u200b|\u200c|\u200d', '', text)  # zero-width chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sentence_split(text: str) -> List[str]:
    # simple sentence splitter (expandable)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ---------- Rule-based Ekman scorer ----------
def tokenize_simple(sentence: str) -> List[str]:
    # lowercase, basic tokenization
    return re.findall(r"\w+['-]?\w*|\w+", sentence.lower())

def get_context_window(tokens: List[str], idx: int, window: int = 3) -> List[str]:
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    return tokens[start:end]

def rule_score_text(text: str) -> Tuple[Dict[str,float], List[Tuple[str,int,str]]]:
    """
    Returns:
      - scores: weighted counts per emotion in EKMAN_PLUS
      - triggers: list of (emotion, weight, triggering_sentence)
    """
    scores = Counter()
    triggers = []
    sentences = sentence_split(text)
    for sent in sentences:
        tokens = tokenize_simple(sent)
        for emo, cues in EKMAN_CUES.items():
            for cue in cues:
                # simple substring match on tokens (stem-like)
                for i, tok in enumerate(tokens):
                    if tok.startswith(cue):  # allows stem matching
                        # base weight
                        weight = 1.0
                        window = get_context_window(tokens, i, window=3)
                        # amplifiers
                        if any(a in window for a in AMPLIFIERS):
                            weight *= 1.8
                        # negation handling: if any negation in window invert or reduce
                        if any(n in window for n in NEGATIONS):
                            weight *= -0.8  # negation flips or reduces
                        scores[emo] += weight
                        triggers.append((emo, weight, sent))
    # Normalize to 0..1 scale per emotion
    max_val = max((abs(v) for v in scores.values()), default=1.0)
    normalized = {emo: float(scores.get(emo,0.0))/max_val for emo in EKMAN_PLUS}
    return normalized, triggers

# ---------- Transformer model wrapper ----------
@st.cache_resource
def load_hf_model(model_name=HF_MODEL, device: int = 0):
    # Load tokenizer + model; pipeline provides convenience for multi-label
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # device mapping
    if torch.cuda.is_available() and device >= 0:
        model.to(torch.device("cuda"))
    return tokenizer, model

def hf_predict_batch(texts: List[str], tokenizer, model, batch_size:int=16, threshold=0.2):
    """
    Returns per-text probability dict for model labels and top triggers (sentences).
    Uses softmax or sigmoid depending on model config (we attempt to detect).
    """
    device = next(model.parameters()).device
    results = []
    # detect multi-label vs single-label via config
    is_multi_label = getattr(model.config, "problem_type", None) == "multi_label_classification" or (model.config.num_labels>7)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            if is_multi_label:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for p in probs:
            # map to labels
            labels = getattr(model.config, "id2label", None)
            if labels:
                label_map = [labels[i] for i in range(len(p))]
            else:
                label_map = [f"label_{i}" for i in range(len(p))]
            prob_dict = dict(zip(label_map, p.tolist()))
            results.append(prob_dict)
    return results

# ---------- Fusion logic ----------
def fuse_scores(model_scores: Dict[str,float], rule_scores: Dict[str,float], weights=(0.6,0.4)):
    """
    Simple weighted average fusion. Use keys union; missing keys treated as 0.
    weights: (model_weight, rule_weight)
    """
    all_keys = set(model_scores.keys()) | set(rule_scores.keys())
    fused = {}
    for k in all_keys:
        fused[k] = weights[0]*model_scores.get(k,0.0)+weights[1]*rule_scores.get(k,0.0)
    return fused

# ---------- Utility: highlight triggers for UI ----------
def highlight_text(text: str, triggers: List[Tuple[str,float,str]]) -> str:
    # naive approach: wrap trigger sentences with span with emoji/class
    html = text
    # unique sentences only
    seen = set()
    for emo, w, sent in triggers:
        if sent in seen: continue
        seen.add(sent)
        safe = st.utils.scriptrunner.util.escape_html(sent) if hasattr(st, "utils") else sent
        # color classes could be added by emotion mapping; here we wrap
        replacement = f"<mark title='{emo} ({w:.2f})'>{safe}</mark>"
        html = html.replace(sent, replacement)
    return html

# ---------- App UI ----------
st.set_page_config(page_title="CT Learner — Emotion Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("CT Learner — Emotion & Ekman Fusion for Student Assignments 🧠📚")

with st.sidebar:
    st.header("Upload / Settings")
    uploaded_files = st.file_uploader("Upload student submissions (txt, pdf, docx). You can upload many.", accept_multiple_files=True)
    use_google_docs = st.checkbox("Analyze Google Doc (paste Doc ID) (requires API setup)", value=False)
    hf_model_name = st.text_input("Hugging Face model", value=HF_MODEL)
    model_batch = st.number_input("Model batch size", min_value=1, max_value=128, value=16)
    fusion_model_weight = st.slider("Transformer weight in fusion", 0.0, 1.0, 0.6)
    run_button = st.button("Analyze submissions")

st.write("Model choice note: default is j-hartmann/emotion-english-roberta-large (Ekman-style).")

# Load model (lazy)
device_idx = 0 if torch.cuda.is_available() else -1
if st.button("Load model now (optional)"):
    with st.spinner("Loading model..."):
        tokenizer, model = load_hf_model(hf_model_name, device=device_idx)
    st.success("Model loaded.")

# Process uploads
if run_button:
    if not uploaded_files and not use_google_docs:
        st.warning("Please upload files or enable Google Docs analysis.")
    else:
        # load model now
        tokenizer, model = load_hf_model(hf_model_name, device=device_idx)
        # collect submissions
        submissions = []
        meta = []
        if uploaded_files:
            for f in uploaded_files:
                name = f.name
                data = f.read()
                try:
                    if name.lower().endswith(".pdf"):
                        txt = extract_text_from_pdf(data)
                    elif name.lower().endswith(".docx"):
                        txt = extract_text_from_docx(data)
                    else:
                        txt = extract_text_from_txt(data)
                except Exception as e:
                    txt = ""
                    st.error(f"Failed to extract {name}: {e}")
                submissions.append(clean_text(txt))
                meta.append({"filename": name})
        # google docs placeholder
        if use_google_docs:
            st.info("Google Docs processing selected — please see README for API setup. (Not performed here.)")

        # Show quick preview
        preview_col, results_col = st.columns([1,2])
        with preview_col:
            st.subheader("Preview (first 3)")
            for i,txt in enumerate(submissions[:3]):
                st.markdown(f"**{meta[i]['filename']}**")
                st.write(txt[:400]+"..." if len(txt)>400 else txt)

        # Transformer inference
        with st.spinner("Running transformer inference..."):
            model_probs = hf_predict_batch(submissions, tokenizer, model, batch_size=model_batch)

        # Map model label names to canonical ekman-like labels where possible
        # If the model uses Ekman labels (j-hartmann), they match. For GoEmotions, mapping required (best-effort)
        def normalize_model_scores(prob_dict):
            # attempt to map tokens to EKMAN_PLUS; fallback: use keys as-is
            normalized = {k.lower(): v for k,v in prob_dict.items()}
            # create reduced mapping for EKMAN_PLUS
            mapped = {}
            for e in EKMAN_PLUS:
                # exact match or synonyms
                if e in normalized:
                    mapped[e] = normalized[e]
                else:
                    # try synonyms mapping (e.g., joy -> happiness)
                    synonyms = {
                        "joy":["joy","happiness","happy","joyful"],
                        "fear":["fear","afraid","anxiety","anxious"],
                        "sadness":["sad","sadness","sorrow"],
                        "anger":["anger","angry","annoy"],
                        "disgust":["disgust","disgusted"],
                        "surprise":["surprise","surprised","astonished"],
                        "shame":["shame","ashamed","embarrass"],
                        "pride":["pride","proud","accomplish"]
                    }
                    val = 0.0
                    for syn in synonyms.get(e,[]):
                        val = max(val, normalized.get(syn, 0.0))
                    mapped[e] = float(val)
            return mapped

        model_mapped = [normalize_model_scores(d) for d in model_probs]

        # Rule-based inference
        with st.spinner("Running rule-based Ekman scorer..."):
            rule_results = [rule_score_text(t) for t in submissions]
            rule_scores = [r[0] for r in rule_results]
            rule_triggers = [r[1] for r in rule_results]

        # Fusion and assemble dataframe
        fused_results = []
        rows = []
        for i,(meta_item, mscore, rscore) in enumerate(zip(meta, model_mapped, rule_scores)):
            fused = fuse_scores(mscore, rscore, weights=(fusion_model_weight, 1-fusion_model_weight))
            # normalized confidences
            model_conf = max(mscore.values()) if mscore else 0.0
            rule_conf = max(rscore.values()) if rscore else 0.0
            fused_conf = max(fused.values()) if fused else 0.0
            row = {
                "filename": meta_item["filename"],
                "model_conf": float(model_conf),
                "rule_conf": float(rule_conf),
                "fused_conf": float(fused_conf),
                "model_scores": json.dumps(mscore),
                "rule_scores": json.dumps(rscore),
                "fused_scores": json.dumps(fused),
            }
            rows.append(row)
            fused_results.append((mscore, rscore, fused, rule_triggers[i]))

        df = pd.DataFrame(rows)

        # Display table and charts
        with results_col:
            st.subheader("Results summary")
            st.dataframe(df[["filename","model_conf","rule_conf","fused_conf"]])
            st.download_button("Export CSV", df.to_csv(index=False), file_name=f"ctlearner_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
            # Excel
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="results")
            st.download_button("Export Excel", towrite.getvalue(), file_name=f"ctlearner_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx")

        # Per-file detailed view
        st.markdown("---")
        st.header("Detailed per-submission analysis")
        for i,(meta_item, mscore, rscore, fused, triggers) in enumerate(fused_results):
            st.subheader(meta_item["filename"])
            cols = st.columns([1,1,1])
            labels = list(fused.keys())
            # Show bar charts as DataFrame (Streamlit handles visuals)
            df_viz = pd.DataFrame({
                "emotion": labels,
                "model": [mscore.get(l,0.0) for l in labels],
                "rule": [rscore.get(l,0.0) for l in labels],
                "fused":[fused.get(l,0.0) for l in labels]
            })
            st.table(df_viz)
            st.markdown("**Top model emotions (sorted):**")
            for k,v in sorted(mscore.items(), key=lambda x:-x[1])[:3]:
                st.write(f"{k}: {v:.3f}")
            st.markdown("**Top rule emotions (sorted):**")
            for k,v in sorted(rscore.items(), key=lambda x:-abs(x[1]))[:3]:
                st.write(f"{k}: {v:.3f}")
            st.markdown("**Fused top:**")
            for k,v in sorted(fused.items(), key=lambda x:-x[1])[:3]:
                st.write(f"{k}: {v:.3f}")

            # Show rule triggers
            st.markdown("**Rule triggers (sentence highlights)**")
            if triggers:
                for emo, w, sent in triggers[:10]:
                    st.markdown(f"- **{emo}** ({w:.2f}): {sent}")
            else:
                st.write("No rule triggers found.")

            st.markdown("**Model probabilities**")
            st.json(mscore)

        st.success("Completed analysis ✅")
