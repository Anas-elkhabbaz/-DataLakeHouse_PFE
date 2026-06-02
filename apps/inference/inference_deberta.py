"""
Page d'infÃ©rence Streamlit â€” PFE Spark Triage
PrÃ©dit issuetype (DeBERTa) + resolution (sklearn) + conseil LLM (GROQ)

Lancer : streamlit run apps/inference/inference_deberta.py
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import torch
import numpy as np
import joblib

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Spark Ticket Triage",
    page_icon="âš¡",
    layout="wide",
)

ROOT        = Path(__file__).parent.parent.parent
DEBERTA_DIR = ROOT / "results" / "deberta_spark_best"
SKLEARN_DIR = ROOT / "results" / "sklearn_models"
META_PATH   = SKLEARN_DIR / "meta.json"

COLORS = {
    "Bug":         "#FF4B4B",
    "Improvement": "#0068C9",
    "Sub-task":    "#09AB3B",
    "Other":       "#FF8700",
    "Fixed":           "#09AB3B",
    "Won't Fix":       "#FF4B4B",
    "Duplicate":       "#FF8700",
    "Not A Problem":   "#888888",
    "Incomplete":      "#FFBD45",
    "Invalid":         "#C0C0C0",
    "Cannot Reproduce":"#A0522D",
}

ICONS = {
    "Bug": "ðŸ›", "Improvement": "âš¡", "Sub-task": "ðŸ”—", "Other": "ðŸ“‹",
    "Fixed": "âœ…", "Won't Fix": "ðŸš«", "Duplicate": "ðŸ“‹",
    "Not A Problem": "âœ“", "Incomplete": "â³", "Invalid": "âŒ",
    "Cannot Reproduce": "ðŸ”",
}

# â”€â”€ Chargement modÃ¨les (cache Streamlit) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_resource
def load_deberta():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not DEBERTA_DIR.exists():
        return None, None
    tok   = AutoTokenizer.from_pretrained(str(DEBERTA_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(DEBERTA_DIR), dtype=torch.float32
    )
    model.eval()
    return tok, model

@st.cache_resource
def load_sklearn():
    if not META_PATH.exists():
        return None, None, None
    meta    = json.loads(META_PATH.read_text(encoding="utf-8"))
    clf_res = joblib.load(SKLEARN_DIR / "clf_resolution.pkl")
    scaler  = joblib.load(SKLEARN_DIR / "scaler.pkl")
    return clf_res, scaler, meta

def predict_issuetype(tok, model, text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[0].numpy()
    classes = list(model.config.id2label.values())
    pred    = classes[int(np.argmax(probs))]
    conf    = float(np.max(probs))
    return pred, conf, dict(zip(classes, probs.tolist()))

def predict_resolution(clf_res, scaler, meta, tab_features: dict):
    feats = [tab_features.get(f, 0) for f in meta["tabular_features"]]
    X     = scaler.transform([feats])
    # Pas d'embedding disponible en temps rÃ©el â†’ utiliser seulement features tabulaires
    # On utilise un embedding nul (zÃ©ros) â€” c'est approximatif mais fonctionnel
    emb_dim = meta["embedding_dim"]
    X_full  = np.hstack([np.zeros((1, emb_dim)), X])
    pred    = clf_res.predict(X_full)[0]
    probs   = clf_res.predict_proba(X_full)[0]
    classes = clf_res.classes_
    conf    = float(np.max(probs))
    return pred, conf, dict(zip(classes, probs.tolist()))

def get_groq_advice(summary, description, issuetype, resolution, confidence_it, confidence_res):
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        prompt = f"""Tu es un expert Apache Spark et JIRA. Analyse ce ticket :

**Summary** : {summary}
**Description** : {description[:400] if description else '(vide)'}
**Type prÃ©dit** : {issuetype} (confiance {confidence_it:.0%})
**RÃ©solution probable** : {resolution} (confiance {confidence_res:.0%})

RÃ©ponds en 3 parties concises :
1. **Analyse** : Pourquoi ce type et cette rÃ©solution ? (2 phrases max)
2. **Actions recommandÃ©es** : 3 Ã©tapes concrÃ¨tes pour traiter ce ticket
3. **Attention** : 1 point de vigilance spÃ©cifique pour ce type de ticket

RÃ©ponds en franÃ§ais, de faÃ§on pratique et directe."""

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"*Conseil LLM indisponible : {e}*"

# â”€â”€ UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.title("âš¡ Apache Spark Ticket Triage")
st.caption("PrÃ©diction automatique du type et de la rÃ©solution â€” PFE 2026")

tok, deberta_model = load_deberta()
clf_res, scaler, meta = load_sklearn()

if tok is None:
    st.error(f"ModÃ¨le DeBERTa introuvable dans `{DEBERTA_DIR}`")
    st.stop()
if clf_res is None:
    st.error(f"ModÃ¨les sklearn introuvables dans `{SKLEARN_DIR}`")
    st.stop()

st.success("ModÃ¨les chargÃ©s â€” DeBERTa (issuetype) + sklearn (resolution) + GROQ (conseils)", icon="âœ…")
st.divider()

# â”€â”€ Formulaire â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.subheader("ðŸ“ Nouveau ticket")

col1, col2 = st.columns([3, 1])
with col1:
    summary = st.text_input("Summary *", placeholder="Ex: NullPointerException in SparkContext.scala line 847")
with col2:
    priority = st.selectbox("Priority", ["Blocker", "Critical", "Major", "Minor", "Trivial"])

description = st.text_area("Description",
    placeholder="Steps to reproduce, stack trace, expected vs actual behavior...",
    height=100)

col3, col4, col5 = st.columns(3)
with col3:
    has_parent = st.checkbox("Ce ticket a un ticket parent (Sub-task)",
                             help="Cocher si ce ticket est la sous-tÃ¢che d'un autre ticket")
with col4:
    n_comments = st.number_input("Nb commentaires", min_value=0, value=0)
with col5:
    n_links = st.number_input("Nb liens vers d'autres tickets", min_value=0, value=0)

btn = st.button("ðŸ” Analyser le ticket", type="primary", use_container_width=True)

# â”€â”€ PrÃ©diction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if btn:
    if not summary.strip():
        st.warning("Le summary est obligatoire.")
        st.stop()

    # Construire le texte (mÃªme logique que l'entraÃ®nement Kaggle)
    parent_flag = "[HAS-PARENT] " if has_parent else "[NO-PARENT] "
    desc_len    = len(description.strip())
    desc_flag   = ""
    if desc_len == 0:    desc_flag = "[NO-DESCRIPTION] "
    elif desc_len < 80:  desc_flag = "[SHORT-DESC] "
    comment_flag = "[NO-COMMENTS] " if n_comments == 0 else ""
    kw = ""
    s  = summary.lower()
    if any(k in s for k in ["fix","bug","error","fail","crash","exception","npe","broken"]):
        kw = "[BUG-SIGNAL] "
    elif any(k in s for k in ["improve","enhance","optimize","performance","refactor"]):
        kw = "[IMPROVEMENT-SIGNAL] "

    text = (f"{parent_flag}{desc_flag}{comment_flag}{kw}"
            f"TICKET: {summary}\nPRI: {priority}\nDESC: {description[:800]}")[:512]

    # Features tabulaires pour resolution
    tab_feats = {
        "n_total_changes": 0, "n_status_changes": 0, "n_priority_changes": 0,
        "n_assignee_changes": 0, "n_resolution_changes": 0, "was_escalated": 0,
        "n_people_involved": 1, "n_links_total": n_links,
        "n_duplicates": 0, "n_blocks": 0, "n_blocked_by": 0, "n_relates": n_links,
        "n_comments": n_comments, "n_commenters": min(n_comments, 3),
        "resolution_days": 0, "summary_length": len(summary),
        "description_length": desc_len, "n_container": 0,
        "has_parent": int(has_parent),
    }

    with st.spinner("Analyse en cours..."):
        pred_it,  conf_it,  probs_it  = predict_issuetype(tok, deberta_model, text)
        pred_res, conf_res, probs_res = predict_resolution(clf_res, scaler, meta, tab_feats)
        advice = get_groq_advice(summary, description, pred_it, pred_res, conf_it, conf_res)

    # â”€â”€ RÃ©sultats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Type de ticket")
        color = COLORS.get(pred_it, "#888")
        icon  = ICONS.get(pred_it, "ðŸ“‹")
        st.markdown(f"""
        <div style="background:{color}22;border-left:5px solid {color};
                    padding:16px;border-radius:8px;margin:8px 0">
            <h2 style="color:{color};margin:0">{icon} {pred_it}</h2>
            <p style="margin:4px 0;color:gray">Confiance : <b>{conf_it*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
        for cls, p in sorted(probs_it.items(), key=lambda x: -x[1]):
            st.progress(float(p), text=f"{cls} â€” {p*100:.1f}%")

    with col_b:
        st.subheader("RÃ©solution probable")
        color2 = COLORS.get(pred_res, "#888")
        icon2  = ICONS.get(pred_res, "ðŸ“‹")
        st.markdown(f"""
        <div style="background:{color2}22;border-left:5px solid {color2};
                    padding:16px;border-radius:8px;margin:8px 0">
            <h2 style="color:{color2};margin:0">{icon2} {pred_res}</h2>
            <p style="margin:4px 0;color:gray">Confiance : <b>{conf_res*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
        for cls, p in sorted(probs_res.items(), key=lambda x: -x[1])[:5]:
            st.progress(float(p), text=f"{cls} â€” {p*100:.1f}%")

    # â”€â”€ Conseil LLM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.divider()
    st.subheader("ðŸ’¡ Conseils de rÃ©solution (GROQ / Llama3)")
    st.markdown(advice)

    # â”€â”€ Debug â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with st.expander("Texte envoyÃ© au modÃ¨le DeBERTa"):
        st.code(text)

# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.header("â„¹ï¸ Ã€ propos")
    st.markdown("""
**PFE 2026 â€” Apache Spark Ticket Triage**

**ModÃ¨les :**
- `DeBERTa-v3-base` fine-tunÃ©
  - 73.6% macro-F1
  - 79.6% accuracy
- `LogisticRegression` sklearn
  - 80% accuracy (rÃ©solution)

**Data :**
- 42 083 tickets Apache Spark JIRA
- 8 841 parent_keys rÃ©cupÃ©rÃ©s via API

**Pipeline :**
dbt â†’ Snowflake â†’ DeBERTa â†’ GROQ
""")
    st.divider()
    st.caption("Powered by Snowflake + HuggingFace + GROQ")

