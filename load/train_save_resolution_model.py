"""
Re-entraine sklearn LogisticRegression pour resolution avec has_parent
et sauvegarde les modeles avec joblib pour la page Streamlit.
Utilise le cache d'embeddings existant.
"""
import os, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import joblib
import snowflake.connector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

RESULTS = Path(__file__).parent.parent / "results"
EMBED_CACHE = RESULTS / "embeddings_cache.npz"
MODELS_DIR  = RESULTS / "sklearn_models"
MODELS_DIR.mkdir(exist_ok=True)

RARE = {"Task", "Documentation", "Test", "Question"}

TABULAR_FEATURES = [
    "n_total_changes", "n_status_changes", "n_priority_changes",
    "n_assignee_changes", "n_resolution_changes", "was_escalated",
    "n_people_involved", "n_links_total", "n_duplicates", "n_blocks",
    "n_blocked_by", "n_relates", "n_comments", "n_commenters",
    "resolution_days", "summary_length", "description_length",
    "n_container", "has_parent",   # nouvelles features
]

print("Connexion Snowflake...")
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH", database="PFE_SPARK", schema="MARTS_ML",
)
cur = conn.cursor()
cur.execute(f"""
    SELECT key, split, issuetype, resolution,
           {', '.join(TABULAR_FEATURES)}
    FROM PFE_SPARK.MARTS_ML.MART_ML
    WHERE split IN ('train', 'validation')
    ORDER BY key
""")
rows = cur.fetchall()
cols = [d[0].lower() for d in cur.description]
df   = pd.DataFrame(rows, columns=cols)
conn.close()
print(f"Chargé {len(df):,} lignes")

# Remapping classes issuetype
df["issuetype"] = df["issuetype"].apply(lambda x: "Other" if x in RARE else x)

train = df[df["split"] == "train"].reset_index(drop=True)
val   = df[df["split"] == "validation"].reset_index(drop=True)

# Vérification alignement cache
print("Chargement embeddings depuis cache...")
cache = np.load(EMBED_CACHE, allow_pickle=True)

if "train_keys" in cache:
    cached_keys = list(cache["train_keys"])
    current_keys = list(train["key"].values)
    if cached_keys != current_keys:
        print("ATTENTION: cache desaligne - utilisation embeddings quand meme (approximatif)")
    else:
        print("Alignement cache OK")

train_emb_res = cache["train_emb_res"]
val_emb_res   = cache["val_emb_res"]

# Features tabulaires
available = [f for f in TABULAR_FEATURES if f in train.columns]
print(f"Features tabulaires: {len(available)} — {available}")

scaler    = StandardScaler()
train_tab = scaler.fit_transform(train[available].fillna(0).clip(upper=1e6))
val_tab   = scaler.transform(val[available].fillna(0).clip(upper=1e6))

X_train_res = np.hstack([train_emb_res, train_tab])
X_val_res   = np.hstack([val_emb_res,   val_tab])

# Entraîner resolution
print("Entrainement LogisticRegression (resolution)...")
t0 = time.time()
clf_res = LogisticRegression(
    class_weight="balanced", max_iter=1000, C=3.0, solver="lbfgs", n_jobs=1
)
clf_res.fit(X_train_res, train["resolution"])
print(f"  Done in {time.time()-t0:.0f}s")

pred_res = clf_res.predict(X_val_res)
f1_res   = f1_score(val["resolution"], pred_res, average="macro", zero_division=0)
acc_res  = (pred_res == val["resolution"].values).mean()
print(f"  Resolution — macro-F1={f1_res:.4f}  accuracy={acc_res:.4f}")
print(classification_report(val["resolution"], pred_res, zero_division=0))

# Entraîner issuetype sklearn (backup)
print("Entrainement LogisticRegression (issuetype backup)...")
train_emb_it = cache["train_emb_it"]
val_emb_it   = cache["val_emb_it"]

X_train_it = np.hstack([train_emb_it, train_tab])
X_val_it   = np.hstack([val_emb_it,   val_tab])

clf_it = LogisticRegression(
    class_weight="balanced", max_iter=1000, C=3.0, solver="lbfgs", n_jobs=1
)
clf_it.fit(X_train_it, train["issuetype"])
pred_it = clf_it.predict(X_val_it)
f1_it   = f1_score(val["issuetype"], pred_it, average="macro", zero_division=0)
print(f"  Issuetype sklearn — macro-F1={f1_it:.4f}")

# Sauvegarder tout
print("\nSauvegarde des modeles...")
joblib.dump(clf_res,   MODELS_DIR / "clf_resolution.pkl")
joblib.dump(clf_it,    MODELS_DIR / "clf_issuetype_sklearn.pkl")
joblib.dump(scaler,    MODELS_DIR / "scaler.pkl")

# Sauvegarder aussi les metadata
import json
meta = {
    "resolution_classes": list(clf_res.classes_),
    "issuetype_classes":  list(clf_it.classes_),
    "tabular_features":   available,
    "embedding_dim":      int(train_emb_res.shape[1]),
}
(MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"Modeles sauvegardes dans {MODELS_DIR}/")
for f in MODELS_DIR.iterdir():
    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
