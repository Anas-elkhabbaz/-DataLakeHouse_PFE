import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
import html

from similar_reference_utils import build_similar_reference_rows

try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    SNOWFLAKE_AVAILABLE = True
except Exception:
    session = None
    SNOWFLAKE_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Spark Intel | AI Triage", page_icon="⚡", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM  (style uniquement — un seul :root a regler)
# ═══════════════════════════════════════════════════════════════════════════
def inject_pro_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@700;800&display=swap');

:root{
  --bg:#f4f6fb;
  --surface:#ffffff;
  --surface-2:#f8fbff;
  --ink:#0b1220;
  --text:#1f2937;
  --muted:#64748b;
  --border:#dfe7f1;
  --primary:#1f4ed8;
  --primary-dark:#173ea8;
  --accent:#2563eb;
  --success:#0f766e;
  --radius:20px;
  --radius-sm:12px;
  --shadow:0 20px 46px rgba(10, 24, 44, 0.12);
  --shadow-soft:0 10px 24px rgba(10, 24, 44, 0.05);
}

.stApp{ background:linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%) !important; color:var(--text); }
html, body, [class*="st-"], .stMarkdown, p, label, span, div{ font-family:'Inter', sans-serif; }
header[data-testid="stHeader"]{ background:transparent; height:0; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
.block-container{ max-width:1240px; padding-top:1.2rem !important; padding-bottom:3rem; }

.topbar{ display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; padding:6px 2px; }
.brand{ display:flex; align-items:center; gap:11px; }
.logo{ width:44px; height:44px; border-radius:14px; display:grid; place-items:center; font-size:18px;
  background:linear-gradient(135deg, var(--primary), var(--accent)); color:white; font-weight:700;
  box-shadow:0 12px 28px rgba(29,78,216,.22); }
.brandname{ font-family:'Sora',sans-serif; font-weight:800; font-size:17px; color:var(--ink); }
.brand-badge{ font-size:10.5px; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:#334155; background:#f1f5f9; border:1px solid #dce4ee; padding:6px 11px; border-radius:999px; }

.hero{ position:relative; overflow:hidden; border-radius:24px; padding:42px 44px;
  background:linear-gradient(135deg, #09111f 0%, #121d33 42%, #193b8a 100%);
  border:1px solid rgba(255,255,255,.10);
  box-shadow:0 24px 56px rgba(2, 8, 23, 0.22); margin-bottom:24px; }
.hero::before{ content:''; position:absolute; inset:0; background:linear-gradient(90deg, rgba(255,255,255,.08), transparent 40%, rgba(255,255,255,.04)); pointer-events:none; }
.hero::after{ content:''; position:absolute; inset:0; background:radial-gradient(circle at top right, rgba(96,165,250,.30), transparent 42%); pointer-events:none; }
.hero-badge{ display:inline-flex; align-items:center; gap:8px; font-size:10.5px; font-weight:700;
  letter-spacing:.16em; text-transform:uppercase; color:#dbeafe;
  background:rgba(37,99,235,.18); border:1px solid rgba(147,197,253,.24); padding:6px 12px; border-radius:999px; }
.hero h1{ font-family:'Sora',sans-serif; font-weight:800; font-size:clamp(28px,3.8vw,40px);
  line-height:1.02; letter-spacing:-.03em; color:#fff; margin:14px 0 10px; }
.hero p{ font-size:15px; line-height:1.6; color:#dce7f7; max-width:680px; margin:0; }
.hero-chips{ display:flex; gap:10px; margin-top:18px; flex-wrap:wrap; }
.hero-chips span{ font-size:12px; color:#e2e8f0; background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.12); padding:7px 12px; border-radius:999px; }

.sec-title{ font-family:'Sora',sans-serif; font-size:22px; font-weight:700; color:var(--ink); letter-spacing:-.02em; margin:6px 0 4px; }
.sec-sub{ font-size:14px; color:var(--muted); margin-bottom:18px; }

.info-card{ background:var(--surface); border:1px solid var(--border); border-radius:18px;
  padding:22px 20px; box-shadow:var(--shadow-soft); height:100%; transition:transform .16s ease, box-shadow .16s ease; }
.info-card:hover{ transform:translateY(-2px); box-shadow:var(--shadow); }
.ic-ico{ width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:20px;
  background:#eff6ff; margin-bottom:12px; color:#1d4ed8; }
.info-card h3{ font-family:'Sora',sans-serif; color:var(--ink); font-size:16px; font-weight:700; margin:0 0 8px; }
.info-card p{ font-size:14px; color:var(--muted); line-height:1.6; margin:0; }

.ws-title{ font-family:'Sora',sans-serif; font-size:26px; font-weight:700; color:var(--ink); letter-spacing:-.02em; display:flex; align-items:center; gap:10px; }

.form-anchor{ display:none; }
[data-testid="stVerticalBlock"]:has(> div > [data-testid="stMarkdownContainer"] > .form-anchor){
  background:var(--surface); border:1px solid var(--border); border-radius:22px;
  padding:24px 26px; box-shadow:var(--shadow); }
.form-head{ font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#475569; margin-bottom:6px; }

.stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="select"] > div{
  background:#fcfdff !important; color:var(--text) !important; border-radius:12px !important;
  border:1px solid var(--border) !important; font-size:14px !important; padding:8px 10px !important; }
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus{
  border-color:var(--primary) !important; box-shadow:0 0 0 3px rgba(37,99,235,.12) !important; }
label, .stCheckbox label p{ font-weight:600 !important; color:#334155 !important; }

.stButton > button{
  background:linear-gradient(135deg, var(--primary), var(--accent)) !important; color:white !important; border:none !important;
  border-radius:12px !important; padding:.78rem 1.45rem !important; font-weight:700 !important; font-size:14.5px !important;
  box-shadow:0 10px 24px rgba(37,99,235,.24); transition:transform .14s ease, box-shadow .14s ease, background .14s ease; }
.stButton > button:hover{ background:linear-gradient(135deg, var(--primary-dark), #1f4ed8) !important; transform:translateY(-1px);
  box-shadow:0 12px 28px rgba(37,99,235,.28); }
.stButton > button:active{ transform:translateY(0); }

.res-card{ background:var(--surface); border:1px solid var(--border); border-radius:18px;
  padding:20px 22px; box-shadow:var(--shadow-soft); border-top:3px solid var(--primary); }
.res-card:hover{ box-shadow:var(--shadow); }
.res-top{ display:flex; align-items:center; gap:11px; }
.res-ico{ width:40px; height:40px; border-radius:11px; display:grid; place-items:center; font-size:19px; }
.res-label{ font-size:10.5px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
.res-value{ font-family:'Sora',sans-serif; font-size:28px; font-weight:800; margin-top:13px; line-height:1.1; }

.reco-card{ background:var(--surface-2); border:1px solid #e2e8f0; border-left:4px solid var(--primary);
  border-radius:16px; padding:20px 22px; box-shadow:var(--shadow-soft); font-size:14.5px; line-height:1.7; color:var(--text); }
.reco-card b, .reco-card strong{ color:var(--ink); }

.sim-card{ background:var(--surface-2); border:1px solid #e2e8f0; border-left:3px solid var(--primary);
  border-radius:12px; padding:12px 14px; box-shadow:0 4px 12px rgba(15,23,42,.03); margin-bottom:8px; display:flex; align-items:center; gap:12px; }
.sim-key{ font-family:'Inter',monospace; font-weight:700; font-size:13px; color:var(--primary);
  background:#eaf2ff; padding:5px 10px; border-radius:8px; white-space:nowrap; }
.sim-text{ font-size:13.5px; color:#475569; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

.app-footer{ text-align:center; color:#94a3b8; font-size:12px; margin-top:34px; padding-top:18px; border-top:1px solid var(--border); }

[data-testid="stVerticalBlock"]{ gap:0.55rem; }
::selection{ background:rgba(37,99,235,.20); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  LOGIQUE ML (STABLE — inchangee)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    import joblib
    if not SNOWFLAKE_AVAILABLE or session is None:
        return None, None, None, None

    tmp = tempfile.mkdtemp()
    for fname in ["clf_resolution.pkl", "scaler.pkl", "meta.json", "clf_issuetype_sklearn.pkl"]:
        try:
            session.file.get(f"@PFE_SPARK.ML_MODELS.app_stage/models/{fname}", tmp)
        except Exception:
            pass
    meta = json.loads(open(f"{tmp}/meta.json").read())
    return (joblib.load(f"{tmp}/clf_resolution.pkl"),
            joblib.load(f"{tmp}/clf_issuetype_sklearn.pkl"),
            joblib.load(f"{tmp}/scaler.pkl"),
            meta)


def fallback_predict(summary, description):
    text = f"{summary} {description}".lower()
    if any(token in text for token in ["memory", "leak", "crash", "error", "bug"]):
        issue_type = "Bug"
        resolution = "Fixed"
    elif any(token in text for token in ["feature", "request", "roadmap"]):
        issue_type = "New Feature"
        resolution = "Incomplete"
    elif "duplicate" in text:
        issue_type = "Sub-task"
        resolution = "Duplicate"
    else:
        issue_type = "Task"
        resolution = "Incomplete"
    return issue_type, resolution

# ── Couche corrective : recommandation "que faire" ───────────────────────────
# Primaire : IA generative via l'API Google Gemini (si GEMINI_API_KEY configuree).
# Repli deterministe : moteur de regles contextuel (NLP) ci-dessous, qui garantit
# la disponibilite du service meme sans cle API ou en cas d'erreur reseau.
def _recommandation_regles(summary, description, priority, it_p, res_p):
    it   = str(it_p).strip().lower()
    res  = str(res_p).strip().lower()
    prio = str(priority).strip().lower()
    texte_ticket = f"{summary} {description}".lower()

    if any(token in texte_ticket for token in ["memory", "leak", "oom", "out of memory", "heap"]):
        contexte = "Le ticket contient des indices de problème mémoire ou de saturation de ressources."
    elif any(token in texte_ticket for token in ["timeout", "slow", "latency", "performance", "hang"]):
        contexte = "Le ticket évoque une dégradation de performance ou un blocage fonctionnel."
    elif any(token in texte_ticket for token in ["exception", "stacktrace", "traceback", "error", "null"]):
        contexte = "Le ticket montre clairement un symptôme d'erreur technique ou une exception remontée."
    elif any(token in texte_ticket for token in ["feature", "request", "enhancement", "roadmap"]):
        contexte = "Le ticket semble davantage lié à une évolution fonctionnelle ou à une demande de produit."
    else:
        contexte = "Le ticket nécessite une lecture attentive des détails techniques et du contexte métier."

    # 1) Assignation selon le type d'incident
    assign = {
        "bug":           "Assigner à l'équipe de maintenance du module concerné.",
        "improvement":   "Placer dans le backlog produit pour priorisation par le PO.",
        "new feature":   "Transmettre au Product Owner pour évaluation roadmap.",
        "sub-task":      "Rattacher la sous-tâche à son ticket parent et suivre l'avancement global.",
        "task":          "Assigner à l'équipe technique en charge de l'opérationnel.",
        "test":          "Orienter vers l'équipe QA pour couverture de test.",
        "documentation": "Confier à l'équipe documentation.",
    }.get(it, "Effectuer un triage manuel : le type d'incident est ambigu (catégorie 'Other').")

    # 2) Action selon la résolution probable
    action = {
        "fixed":            "Vérifier qu'un correctif ou un commit existe déjà et prévoir un test de non-régression.",
        "won't fix":        "Documenter la justification puis fermer après validation du PO.",
        "wontfix":          "Documenter la justification puis fermer après validation du PO.",
        "invalid":          "Demander des précisions au rapporteur et confirmer la version concernée avant fermeture.",
        "duplicate":        "Rechercher le ticket d'origine et fusionner les deux entrées si nécessaire.",
        "incomplete":       "Demander les logs complets, les étapes de reproduction et le contexte exact.",
        "cannot reproduce": "Demander l'environnement, la version exacte et les détails de reproduction.",
        "not a problem":    "Clarifier avec le rapporteur le comportement attendu et la cible fonctionnelle.",
    }.get(res, "Examiner manuellement le ticket : la résolution prédite est incertaine.")

    # 3) Étapes concrètes à réaliser
    if it in {"bug", "task", "test"}:
        etapes = [
            "Valider le symptôme principal et l'impact utilisateur ou système.",
            "Vérifier les logs, la stacktrace et les derniers changements liés au module.",
            "Définir un plan d'action court : reproduction, correctif, test ou escalade si besoin.",
        ]
    elif it in {"new feature", "improvement"}:
        etapes = [
            "Qualifier la demande et vérifier s'elle répond à un besoin métier réel.",
            "Identifier les contraintes techniques, les dépendances et le périmètre à couvrir.",
            "Préparer une proposition de priorisation ou d'implémentation pour le PO et l'équipe.",
        ]
    else:
        etapes = [
            "Rassembler les informations utiles autour du ticket et des échanges associés.",
            "Confirmer la bonne compréhension du besoin avant toute prise en charge.",
            "Définir le prochain pas opérationnel avec l'équipe concernée.",
        ]

    # 4) Urgence selon la priorité
    if prio in ("blocker", "critical"):
        urgence = "Priorité haute : traiter immédiatement et escalader si nécessaire."
    elif prio == "major":
        urgence = "À planifier dans le sprint en cours."
    else:
        urgence = "Peut être placé dans le backlog, sans urgence particulière."

    texte = (
        f"Assignation : {assign}\n"
        f"Contexte : {contexte}\n"
        f"Action principale : {action}\n"
        f"Étapes concrètes :\n"
        f"1. {etapes[0]}\n"
        f"2. {etapes[1]}\n"
        f"3. {etapes[2]}\n"
        f"Priorité : {urgence}"
    )
    return texte, "Moteur de règles contextuel (NLP)"


def _recommandation_gemini(summary, description, priority, it_p, res_p):
    """Génère l'explication « Que faire ? » via l'API Google Gemini.

    Retourne (texte, source) ou lève une exception si la clé est absente / l'appel échoue
    (le repli sur le moteur de règles est alors assuré par generer_recommandation()).
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY absente")

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)

    prompt = (
        "Tu es un assistant de triage d'incidents logiciels (projet Apache Spark, tickets JIRA). "
        "À partir des prédictions d'un modèle, rédige une recommandation actionnable en FRANÇAIS, "
        "concise et structurée exactement avec ces lignes :\n"
        "Module concerné : ...\nDiagnostic : ...\nAction : ...\n"
        "Étapes : 1) ... 2) ... 3) ...\nUrgence : ...\n\n"
        f"Type d'incident prédit : {it_p}\n"
        f"Résolution probable prédite : {res_p}\n"
        f"Priorité JIRA : {priority}\n"
        f"Résumé du ticket : {summary}\n"
        f"Description : {description[:1500]}\n"
    )
    resp = model.generate_content(prompt)
    texte = (resp.text or "").strip()
    if not texte:
        raise RuntimeError("Réponse Gemini vide")
    return texte, f"Google Gemini ({model_name})"


def generer_recommandation(summary, description, priority, it_p, res_p):
    """Explication « Que faire ? » : IA générative (Gemini) en priorité, repli sur le moteur de règles."""
    try:
        return _recommandation_gemini(summary, description, priority, it_p, res_p)
    except Exception:
        return _recommandation_regles(summary, description, priority, it_p, res_p)


def topbar():
    st.markdown("""
<div class="topbar">
<div class="brand"><span class="logo">S</span><span class="brandname">Spark Intelligence</span></div>
<span class="brand-badge">AI Triage</span>
</div>
""", unsafe_allow_html=True)


def load_similar_references(issue_type, summary="", description="", limit=3):
    try:
        if not SNOWFLAKE_AVAILABLE or session is None:
            raise RuntimeError("Snowflake not configured")

        sim_df = session.sql(
            f"SELECT key, summary_clean, issuetype, resolution FROM PFE_SPARK.MARTS_ML.MART_ML "
            f"WHERE issuetype='{str(issue_type).strip()}' LIMIT {limit}"
        ).to_pandas()

        if sim_df.empty:
            raise ValueError("No historical rows returned")

        sim_df = sim_df.rename(columns=str.lower)
        sim_df = sim_df[["key", "summary_clean", "issuetype", "resolution"]].copy()
        sim_df["similarity"] = 0.0
        sim_df["source"] = "snowflake"
        return sim_df.head(limit)
    except Exception:
        return build_similar_reference_rows(issue_type, summary, description, limit=limit)

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE D'ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════
def render_home():
    topbar()
    st.markdown("""
<div class="hero">
<span class="hero-badge">Apache Spark · Triage intelligent</span>
<h1>Spark Intelligence AI</h1>
<p>La solution intelligente pour automatiser le diagnostic et le triage de vos tickets
     JIRA Apache Spark — type d'incident, resolution probable et cas similaires en quelques secondes.</p>
<div class="hero-chips">
<span>Triage en quelques secondes</span>
<span>Base sur l'historique des tickets</span>
<span>References similaires suggerees</span>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Pourquoi utiliser cette plateforme ?</div>'
                '<div class="sec-sub">Un assistant de triage concu pour faire gagner du temps aux equipes.</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="info-card">
<div class="ic-ico">⏱️</div>
<h3>Gain de productivite</h3>
<p>Reduisez le temps de triage manuel. L'IA analyse instantanement les logs et resumes
               techniques pour vous orienter.</p>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
<div class="ic-ico">●</div>
<h3>Decision assistee</h3>
<p>Le ticket est compare a des milliers de cas historiques pour predire la resolution
               la plus probable (Fixed, Won't Fix, etc.).</p>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="info-card">
<div class="ic-ico">▣</div>
<h3>Base de connaissances</h3>
<p>Accedez aux tickets similaires deja resolus pour ne jamais repartir de zero
               sur un probleme deja rencontre.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Acceder au terminal d'analyse  →"):
            navigate_to('app')

    st.markdown('<div class="app-footer">© 2026 · PFE Apache Spark Ticket Triage</div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
def render_app():
    topbar()
    nav1, nav2 = st.columns([5, 1])
    with nav1:
        st.markdown('<div class="ws-title">🛠 Analyseur de tickets</div>', unsafe_allow_html=True)
    with nav2:
        if st.button("← Accueil"):
            navigate_to('home')

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.info("Mode local : l'interface s'affiche bien. Si Snowflake n'est pas configuré, une prédiction de secours est utilisée pour tester le flux.")

    clf_res, clf_it, scaler, meta = load_models()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Carte formulaire (enveloppe reelle via l'ancre + :has)
    with st.container():
        st.markdown('<div class="form-anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="form-head">Informations du ticket</div>', unsafe_allow_html=True)

        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            summary = st.text_input("Resume du ticket (Summary) *",
                                     placeholder="Ex: Memory leak in TaskSetManager")
        with col_f2:
            priority = st.selectbox("Priorite JIRA",
                                    ["Blocker", "Critical", "Major", "Minor", "Trivial"])

        description = st.text_area("Description technique complete", height=150,
                                   placeholder="Collez ici les logs ou la stacktrace...")

        c_a, c_b, c_c = st.columns(3)
        with c_a:
            has_parent = st.checkbox("Le ticket est une sous-tache")
        with c_b:
            n_comments = st.number_input("Nombre de commentaires", 0)
        with c_c:
            n_links = st.number_input("Nombre de liens JIRA", 0)

        btn_analyze = st.button("Lancer l'analyse predictive")

    if btn_analyze:
        if not summary:
            st.error("Le resume est obligatoire.")
        else:
            with st.spinner("Analyse semantique en cours..."):
                if clf_res is None or clf_it is None or scaler is None or meta is None:
                    it_p, res_p = fallback_predict(summary, description)
                else:
                    # --- LOGIQUE ML (inchangee) ---
                    tab_feats = {"n_links_total": n_links, "n_comments": n_comments,
                                 "summary_length": len(summary), "description_length": len(description),
                                 "has_parent": int(has_parent)}
                    feats_list = [tab_feats.get(f, 0) for f in meta["tabular_features"]]
                    X = scaler.transform([feats_list])
                    X_full = np.hstack([np.zeros((1, meta["embedding_dim"])), X])
                    it_p = clf_it.predict(X_full)[0]
                    res_p = clf_res.predict(X_full)[0]

            # Resultats
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Recommandations</div>', unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""<div class="res-card">
<div class="res-top">
<span class="res-ico" style="background:#EAF0FE;color:#2563EB">#</span>
<span class="res-label">Type d'incident</span></div>
<div class="res-value" style="color:#2563EB">{html.escape(str(it_p))}</div>
</div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="res-card" style="border-top-color:#10B981">
<div class="res-top">
<span class="res-ico" style="background:#E7F8F1;color:#10B981">✓</span>
<span class="res-label">Resolution probable</span></div>
<div class="res-value" style="color:#10B981">{html.escape(str(res_p))}</div>
</div>""", unsafe_allow_html=True)

            # Recommandation "que faire" (moteur de regles, hors Cortex)
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title" style="font-size:19px">Que faire ?</div>',
                        unsafe_allow_html=True)
            reco, source = generer_recommandation(summary, description, priority, it_p, res_p)
            st.markdown(
                f'<div class="reco-card">{html.escape(reco).replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)
            st.caption(f"Genere via : {source}")

            # References similaires
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="sec-title" style="font-size:19px">Références historiques similaires</div>',
                        unsafe_allow_html=True)
            sim_df = load_similar_references(it_p, summary, description, limit=3)
            if sim_df.empty:
                st.caption("Aucun ticket similaire disponible pour le moment.")
            else:
                for _, row in sim_df.iterrows():
                    key = html.escape(str(row.get("key", "")))
                    txt = html.escape(str(row.get("summary_clean", ""))[:120])
                    st.markdown(f'<div class="sim-card"><span class="sim-key">{key}</span>'
                                f'<span class="sim-text">{txt}...</span></div>',
                                unsafe_allow_html=True)

                if str(sim_df.get("source", "").iloc[0] if not sim_df.empty else "").lower() == "fallback":
                    st.caption("Données historiques non disponibles en mode local : des références de secours sont affichées.")

    st.markdown('<div class="app-footer">© 2026 · PFE Apache Spark Ticket Triage</div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  ROUTAGE
# ═══════════════════════════════════════════════════════════════════════════
inject_pro_css()
if st.session_state.page == 'home':
    render_home()
else:
    render_app()
