# PFE Spark Triage — Streamlit in Snowflake
# Ce fichier est deployé directement dans Snowflake (SiS)
# Acces natif aux donnees + modeles depuis le Stage

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
from pathlib import Path

# ── Session Snowflake native (pas besoin de credentials) ──────────────────
from snowflake.snowpark.context import get_active_session
session = get_active_session()

# ── Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spark Ticket Triage",
    page_icon="?",
    layout="wide",
)

COLORS = {
    "Bug": "#FF4B4B", "Improvement": "#0068C9",
    "Sub-task": "#09AB3B", "Other": "#FF8700",
    "Fixed": "#09AB3B", "Won't Fix": "#FF4B4B",
    "Duplicate": "#FF8700", "Not A Problem": "#888888",
    "Incomplete": "#FFBD45", "Invalid": "#C0C0C0",
    "Cannot Reproduce": "#A0522D",
}

# ── Chargement des modeles depuis le Stage ─────────────────────────────────
@st.cache_resource
def load_models():
    import joblib
    import sklearn

    tmp = tempfile.mkdtemp()

    # Telecharger les fichiers depuis le Stage Snowflake
    for fname in ["clf_resolution.pkl", "scaler.pkl", "meta.json"]:
        session.file.get(
            f"@PFE_SPARK.ML_MODELS.app_stage/models/{fname}",
            tmp
        )

    meta    = json.loads(open(f"{tmp}/meta.json").read())
    clf_res = joblib.load(f"{tmp}/clf_resolution.pkl")
    scaler  = joblib.load(f"{tmp}/scaler.pkl")

    # Essayer de charger DeBERTa si disponible
    deberta_tok   = None
    deberta_model = None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        deberta_dir = os.path.join(tmp, "deberta")
        os.makedirs(deberta_dir, exist_ok=True)
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                      "special_tokens_map.json", "spm.model", "model.safetensors",
                      "added_tokens.json"]:
            try:
                session.file.get(
                    f"@PFE_SPARK.ML_MODELS.app_stage/deberta/{fname}",
                    deberta_dir
                )
            except Exception:
                pass
        if os.path.exists(os.path.join(deberta_dir, "config.json")):
            deberta_tok   = AutoTokenizer.from_pretrained(
                deberta_dir, local_files_only=True
            )
            deberta_model = AutoModelForSequenceClassification.from_pretrained(
                deberta_dir,
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )
            deberta_model.eval()
    except Exception as e:
        st.sidebar.warning(f"DeBERTa non charge: {e}")

    return clf_res, scaler, meta, deberta_tok, deberta_model

def predict_issuetype_deberta(tok, model, text):
    import torch
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[0].numpy()
    classes = list(model.config.id2label.values())
    return classes[int(np.argmax(probs))], float(np.max(probs)), dict(zip(classes, probs.tolist()))

def predict_issuetype_sklearn(meta, clf_res, scaler, tab_feats):
    # Fallback sklearn si DeBERTa non disponible
    import joblib
    import tempfile
    tmp = tempfile.mkdtemp()
    session.file.get("@PFE_SPARK.ML_MODELS.app_stage/models/clf_issuetype_sklearn.pkl", tmp)
    clf_it = joblib.load(f"{tmp}/clf_issuetype_sklearn.pkl")
    feats  = [tab_feats.get(f, 0) for f in meta["tabular_features"]]
    X = scaler.transform([feats])
    X_full = np.hstack([np.zeros((1, meta["embedding_dim"])), X])
    pred  = clf_it.predict(X_full)[0]
    probs = clf_it.predict_proba(X_full)[0]
    return pred, float(np.max(probs)), dict(zip(clf_it.classes_, probs.tolist()))

def predict_resolution(clf_res, scaler, meta, tab_feats):
    feats  = [tab_feats.get(f, 0) for f in meta["tabular_features"]]
    X      = scaler.transform([feats])
    X_full = np.hstack([np.zeros((1, meta["embedding_dim"])), X])
    pred   = clf_res.predict(X_full)[0]
    probs  = clf_res.predict_proba(X_full)[0]
    return pred, float(np.max(probs)), dict(zip(clf_res.classes_, probs.tolist()))

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("Apache Spark Ticket Triage")
st.caption("Prediction automatique — issuetype + resolution | PFE 2026")

clf_res, scaler, meta, deberta_tok, deberta_model = load_models()

model_label = "DeBERTa + sklearn" if deberta_model else "sklearn uniquement"
st.success(f"Modeles charges ({model_label})")
st.divider()

# ── Formulaire ─────────────────────────────────────────────────────────────
st.subheader("Nouveau ticket JIRA")

col1, col2 = st.columns([3, 1])
with col1:
    summary = st.text_input("Summary *",
        placeholder="Ex: NullPointerException in SparkContext.scala line 847")
with col2:
    priority = st.selectbox("Priority", ["Blocker", "Critical", "Major", "Minor", "Trivial"])

description = st.text_area("Description",
    placeholder="Steps to reproduce, stack trace...", height=100)

col3, col4, col5 = st.columns(3)
with col3:
    has_parent = st.checkbox("Sous-tache (a un ticket parent)",
        help="Cocher si ce ticket est la sous-tache d'un autre ticket JIRA")
with col4:
    n_comments = st.number_input("Nb commentaires", min_value=0, value=0)
with col5:
    n_links = st.number_input("Nb liens", min_value=0, value=0)

btn = st.button("Analyser le ticket", type="primary", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────
if btn:
    if not summary.strip():
        st.warning("Le summary est obligatoire.")
        st.stop()

    # Construire le texte
    parent_flag  = "[HAS-PARENT] " if has_parent else "[NO-PARENT] "
    desc_len     = len(description.strip())
    desc_flag    = "[NO-DESCRIPTION] " if desc_len == 0 else ("[SHORT-DESC] " if desc_len < 80 else "")
    comment_flag = "[NO-COMMENTS] " if n_comments == 0 else ""
    s = summary.lower()
    kw = ""
    if any(k in s for k in ["fix","bug","error","fail","crash","exception","npe"]):
        kw = "[BUG-SIGNAL] "
    elif any(k in s for k in ["improve","enhance","optimize","performance","refactor"]):
        kw = "[IMPROVEMENT-SIGNAL] "

    text = (f"{parent_flag}{desc_flag}{comment_flag}{kw}"
            f"TICKET: {summary}\nPRI: {priority}\nDESC: {description[:800]}")[:512]

    tab_feats = {
        "n_total_changes": 0, "n_status_changes": 0, "n_priority_changes": 0,
        "n_assignee_changes": 0, "n_resolution_changes": 0, "was_escalated": 0,
        "n_people_involved": 1, "n_links_total": int(n_links),
        "n_duplicates": 0, "n_blocks": 0, "n_blocked_by": 0, "n_relates": int(n_links),
        "n_comments": int(n_comments), "n_commenters": min(int(n_comments), 3),
        "resolution_days": 0, "summary_length": len(summary),
        "description_length": desc_len, "n_container": 0,
        "has_parent": int(has_parent),
    }

    with st.spinner("Analyse en cours..."):
        if deberta_model:
            pred_it, conf_it, probs_it = predict_issuetype_deberta(deberta_tok, deberta_model, text)
        else:
            pred_it, conf_it, probs_it = predict_issuetype_sklearn(meta, clf_res, scaler, tab_feats)
        pred_res, conf_res, probs_res = predict_resolution(clf_res, scaler, meta, tab_feats)

    # ── Resultats ────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Type de ticket")
        color = COLORS.get(pred_it, "#888")
        st.markdown(f"""
        <div style="background:{color}22;border-left:5px solid {color};
                    padding:16px;border-radius:8px">
            <h2 style="color:{color};margin:0">{pred_it}</h2>
            <p style="margin:4px 0;color:gray">Confiance : <b>{conf_it*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
        for cls, p in sorted(probs_it.items(), key=lambda x: -x[1]):
            st.progress(float(p), text=f"{cls} — {p*100:.1f}%")

    with col_b:
        st.subheader("Resolution probable")
        color2 = COLORS.get(pred_res, "#888")
        st.markdown(f"""
        <div style="background:{color2}22;border-left:5px solid {color2};
                    padding:16px;border-radius:8px">
            <h2 style="color:{color2};margin:0">{pred_res}</h2>
            <p style="margin:4px 0;color:gray">Confiance : <b>{conf_res*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
        for cls, p in sorted(probs_res.items(), key=lambda x: -x[1])[:5]:
            st.progress(float(p), text=f"{cls} — {p*100:.1f}%")

    # ── Tickets similaires depuis Snowflake ────────────────────────────
    st.divider()
    st.subheader("Tickets similaires dans la base (Snowflake)")
    try:
        similar = session.sql(f"""
            SELECT key, issuetype, resolution, summary_clean
            FROM PFE_SPARK.MARTS_ML.MART_ML
            WHERE issuetype = '{pred_it}'
            AND resolution = '{pred_res}'
            AND split = 'train'
            ORDER BY RANDOM()
            LIMIT 5
        """).to_pandas()
        st.dataframe(similar[["key","issuetype","resolution","summary_clean"]],
                     use_container_width=True)
    except Exception as e:
        st.info(f"Tickets similaires non disponibles: {e}")

    with st.expander("Texte envoye au modele"):
        st.code(text)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("A propos")
    st.markdown("""
**PFE 2026 — Apache Spark Ticket Triage**

**Modeles :**
- DeBERTa-v3-base (issuetype)
  - 73.6% macro-F1
  - 79.6% accuracy
- LogisticRegression (resolution)
  - 80% accuracy

**Pipeline :**
- 42 083 tickets Apache Spark
- 8 841 parent_keys (API JIRA)
- dbt + Snowflake + ML + LLM
""")

    # Stats live depuis Snowflake
    st.divider()
    st.subheader("Stats live")
    try:
        stats = session.sql("""
            SELECT issuetype, COUNT(*) as n
            FROM PFE_SPARK.MARTS_ML.MART_ML
            GROUP BY issuetype ORDER BY n DESC
        """).to_pandas()
        st.dataframe(stats, use_container_width=True, hide_index=True)
    except Exception:
        pass
