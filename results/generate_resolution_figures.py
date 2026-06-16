"""
Génère les figures d'évaluation du modèle de prédiction de la RÉSOLUTION pour le rapport PFE,
à partir du modèle réellement déployé (results/sklearn_models/clf_resolution.pkl) et des
données de validation (PFE_SPARK.MARTS_ML.MART_ML + embeddings_cache.npz).

Ne ré-entraîne pas : charge le modèle sauvegardé et l'évalue sur la validation.
Sorties (dans le dossier images/ du rapport) :
  - res_distribution.png  : répartition des classes de résolution (train vs validation)
  - res_confusion.png     : matrice de confusion (validation)
  - res_metrics.png       : précision / rappel / F1 par classe
Et results/confusion_matrix_resolution.csv (vraie matrice).

Usage : python results/generate_resolution_figures.py
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

RESULTS = Path(__file__).resolve().parent
MODELS  = RESULTS / "sklearn_models"
EMBED   = RESULTS / "embeddings_cache.npz"
OUT     = RESULTS.parents[1] / "Rapport final" / "PFE_Anas_UIR" / "images"
OUT.mkdir(parents=True, exist_ok=True)

C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE = "#4c72b0", "#55a868", "#dd8452", "#c44e52", "#8172b3"
PALETTE = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, "#937860", "#8c8c8c"]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "white"})

TABULAR_FEATURES = [
    "n_total_changes", "n_status_changes", "n_priority_changes",
    "n_assignee_changes", "n_resolution_changes", "was_escalated",
    "n_people_involved", "n_links_total", "n_duplicates", "n_blocks",
    "n_blocked_by", "n_relates", "n_comments", "n_commenters",
    "resolution_days", "summary_length", "description_length",
    "n_container", "has_parent",
]

# ── Données ──────────────────────────────────────────────────────────────────
print("Connexion Snowflake...")
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"], role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH", database="PFE_SPARK", schema="MARTS_ML",
)
cur = conn.cursor()
cur.execute(f"""
    SELECT key, split, resolution, {', '.join(TABULAR_FEATURES)}
    FROM PFE_SPARK.MARTS_ML.MART_ML
    WHERE split IN ('train', 'validation')
    ORDER BY key
""")
rows = cur.fetchall()
cols = [d[0].lower() for d in cur.description]
df = pd.DataFrame(rows, columns=cols)
conn.close()
print(f"Chargé {len(df):,} lignes")

train = df[df["split"] == "train"].reset_index(drop=True)
val   = df[df["split"] == "validation"].reset_index(drop=True)

# ── Modèle + embeddings ──────────────────────────────────────────────────────
cache = np.load(EMBED, allow_pickle=True)
val_emb_res = cache["val_emb_res"]
scaler  = joblib.load(MODELS / "scaler.pkl")
clf     = joblib.load(MODELS / "clf_resolution.pkl")
meta    = json.loads((MODELS / "meta.json").read_text(encoding="utf-8"))
feats   = [f for f in meta["tabular_features"] if f in val.columns]

val_tab = scaler.transform(val[feats].fillna(0).clip(upper=1e6))
X_val   = np.hstack([val_emb_res, val_tab])
y_true  = val["resolution"].values
y_pred  = clf.predict(X_val)

acc = (y_pred == y_true).mean()
labels = list(clf.classes_)
p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
macro_f1 = f1.mean()
print(f"Resolution — accuracy={acc:.4f}  macro-F1={macro_f1:.4f}")
print(classification_report(y_true, y_pred, zero_division=0))

# ── 1. Distribution des classes (train vs val) ──────────────────────────────
order = ["Fixed", "Duplicate", "Won't Fix", "Not A Problem", "Invalid", "Cannot Reproduce", "Incomplete"]
order = [c for c in order if c in set(train["resolution"]) | set(val["resolution"])]
tr_counts = train["resolution"].value_counts()
va_counts = val["resolution"].value_counts()
x = np.arange(len(order)); w = 0.4
fig, ax = plt.subplots(figsize=(9, 4.6))
b1 = ax.bar(x - w/2, [tr_counts.get(c, 0) for c in order], w, label=f"Train ({len(train):,})".replace(",", " "), color=C_RED)
b2 = ax.bar(x + w/2, [va_counts.get(c, 0) for c in order], w, label=f"Validation ({len(val):,})".replace(",", " "), color="#f0a3a3")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha="right")
ax.set_ylabel("Nombre de tickets"); ax.set_title("Distribution des classes de résolution — Train vs Validation")
ax.set_yscale("log"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
for b in list(b1) + list(b2):
    h = b.get_height()
    if h > 0:
        ax.text(b.get_x()+b.get_width()/2, h, f"{int(h):,}".replace(",", " "), ha="center", va="bottom", fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "res_distribution.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  -> res_distribution.png")

# ── 2. Matrice de confusion ─────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=labels)
pd.DataFrame(cm, index=labels, columns=labels).to_csv(RESULTS / "confusion_matrix_resolution.csv")
fig, ax = plt.subplots(figsize=(7.2, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
ax.set_title("Matrice de confusion — résolution (validation)")
thr = cm.max() / 2 if cm.max() else 1
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thr else "#1f2937", fontsize=9)
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(OUT / "res_confusion.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  -> res_confusion.png")

# ── 3. Précision / Rappel / F1 par classe ────────────────────────────────────
idx = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.bar(idx - w, p,  w, label="Précision", color=C_BLUE)
ax.bar(idx,     r,  w, label="Rappel",    color=C_GREEN)
ax.bar(idx + w, f1, w, label="F1-score",  color=C_ORANGE)
ax.axhline(macro_f1, ls="--", color=C_RED, lw=1, label=f"Macro-F1 = {macro_f1:.2f}")
ax.set_xticks(idx); ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Précision, Rappel et F1-score par classe — résolution")
ax.legend(ncol=2, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "res_metrics.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("  -> res_metrics.png")

print("\n=== Récapitulatif par classe (pour la prose) ===")
for c, pp, rr, ff, ss in zip(labels, p, r, f1, sup):
    print(f"  {c:18} P={pp:.2f} R={rr:.2f} F1={ff:.2f} support={ss}")
print(f"  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
