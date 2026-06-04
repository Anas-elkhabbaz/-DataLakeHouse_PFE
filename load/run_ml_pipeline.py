"""
V7 Hybrid RCA — Pipeline amélioré (LogisticRegression + features mixtes)
Steps:
  1. Fetch mart_ml depuis Snowflake (texte + features tabulaires)
  2. Embed text_noco avec sentence-transformers (all-mpnet-base-v2)
  3. Combiner embeddings + features tabulaires normalisées
  4. Entraîner LogisticRegression avec class_weight='balanced'
  5. Évaluer (accuracy, macro-F1, confusion matrix)
  6. Sauvegarder localement + upload Snowflake
"""
import os, time, json
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sentence_transformers import SentenceTransformer
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Classes issuetype rares (<2%) fusionnées dans "Other" — trop peu d'exemples
# pour apprendre une frontière distincte avec la qualité de texte actuelle.
RARE_ISSUETYPES = {"Task", "Documentation", "Test", "Question"}

load_dotenv()

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

EMBED_CACHE = RESULTS / "embeddings_cache.npz"

# ── 1. Connect + fetch data ─────────────────────────────────────────────────
print("Connecting to Snowflake …")
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH",
    database="PFE_SPARK",
    schema="MARTS_ML",
)
cur = conn.cursor()

TABULAR_FEATURES = [
    "n_total_changes", "n_status_changes", "n_priority_changes",
    "n_assignee_changes", "n_resolution_changes", "was_escalated",
    "n_people_involved", "n_links_total", "n_duplicates", "n_blocks",
    "n_blocked_by", "n_relates", "n_comments", "n_commenters",
    "resolution_days", "summary_length", "description_length",
]

print("Fetching mart_ml ...")
cur.execute(f"""
    SELECT key, split, issuetype, resolution,
           text_for_it, text_for_res, text_noco,
           {', '.join(TABULAR_FEATURES)}
    FROM PFE_SPARK.MARTS_ML.MART_ML
    WHERE split IN ('train', 'validation')
    ORDER BY key
""")
rows = cur.fetchall()
cols = [d[0].lower() for d in cur.description]
df = pd.DataFrame(rows, columns=cols)
print(f"  Loaded {len(df):,} rows  (train={len(df[df.split=='train']):,}, val={len(df[df.split=='validation']):,})")

train = df[df["split"] == "train"].reset_index(drop=True)
val   = df[df["split"] == "validation"].reset_index(drop=True)

# Correction 2 : grouper les classes issuetype rares -> "Other"
# Task/Documentation/Test/Question < 2% chacune, textes trop similaires aux autres
before = train["issuetype"].value_counts().to_dict()
train["issuetype"] = train["issuetype"].apply(lambda x: "Other" if x in RARE_ISSUETYPES else x)
val["issuetype"]   = val["issuetype"].apply(lambda x: "Other" if x in RARE_ISSUETYPES else x)
after = train["issuetype"].value_counts().to_dict()
print("  issuetype apres grouping:", after)

# ── 2. Embed with sentence-transformers ─────────────────────────────────────
model_name = "all-mpnet-base-v2"

# Colonnes texte : separees par tache pour eviter la fuite de label
IT_COL  = "text_for_it"   # sans TYPE: -> predire issuetype
RES_COL = "text_for_res"  # avec TYPE: + commentaires -> predire resolution

if EMBED_CACHE.exists():
    print(f"Loading cached embeddings from {EMBED_CACHE} ...")
    cache = np.load(EMBED_CACHE, allow_pickle=True)
    # Verifier l'alignement cache/donnees via les cles
    if "train_keys" in cache:
        cached_train_keys = list(cache["train_keys"])
        current_train_keys = list(train["key"].values)
        if cached_train_keys != current_train_keys:
            print("  [ATTENTION] Cles du cache ne correspondent pas aux donnees actuelles.")
            print("  Suppression du cache et re-embedding necessaire.")
            EMBED_CACHE.unlink()
            print("  Cache supprime. Relancez le pipeline.")
            raise SystemExit(1)
        print("  Alignement cache/donnees : OK")
    if "train_emb_it" in cache:
        train_emb_it  = cache["train_emb_it"]
        val_emb_it    = cache["val_emb_it"]
        train_emb_res = cache["train_emb_res"]
        val_emb_res   = cache["val_emb_res"]
    else:
        train_emb_it = train_emb_res = cache["train_emb"]
        val_emb_it   = val_emb_res   = cache["val_emb"]
    print(f"  train_emb_it shape: {train_emb_it.shape}")
else:
    print(f"Loading model {model_name} ...")
    model = SentenceTransformer(model_name)

    def embed(texts, label):
        t0 = time.time()
        emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True)
        print(f"  {label}: {time.time()-t0:.0f}s — shape {emb.shape}")
        return emb

    train_emb_it  = embed(train[IT_COL].fillna("").tolist(),  "train it")
    val_emb_it    = embed(val[IT_COL].fillna("").tolist(),    "val it")
    train_emb_res = embed(train[RES_COL].fillna("").tolist(), "train res")
    val_emb_res   = embed(val[RES_COL].fillna("").tolist(),   "val res")

    np.savez_compressed(EMBED_CACHE,
                        train_emb_it=train_emb_it, val_emb_it=val_emb_it,
                        train_emb_res=train_emb_res, val_emb_res=val_emb_res,
                        train_keys=np.array(train["key"].values),
                        val_keys=np.array(val["key"].values))
    print(f"  Embeddings cached -> {EMBED_CACHE}")

# ── 3. Construire les matrices X (embeddings + features tabulaires) ──────────
print("Normalisation des features tabulaires ...")
scaler = StandardScaler()
train_tab = scaler.fit_transform(
    train[TABULAR_FEATURES].fillna(0).clip(upper=1e6)
)
val_tab = scaler.transform(
    val[TABULAR_FEATURES].fillna(0).clip(upper=1e6)
)

X_train = np.hstack([train_emb_it,  train_tab])
X_val   = np.hstack([val_emb_it,    val_tab])
X_train_res = np.hstack([train_emb_res, train_tab])
X_val_res   = np.hstack([val_emb_res,   val_tab])
print(f"  X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# ── 4. Oversampling + entraînement ──────────────────────────────────────────
# Correction 3 : RandomOverSampler sur les classes < 5% pour compenser
# ce que class_weight='balanced' ne peut pas corriger quand le texte manque.
# Oversampling issuetype seulement : les 5 classes sont a parité apres grouping
# mais New Feature (1896) et Other (2907) restent sous-représentés vs Bug (15037).
# On limite a 2x la classe mediane pour eviter l'overfitting par duplication excessive.
print("Oversampling issuetype (classes minoritaires) ...")
median_count = int(pd.Series(train["issuetype"]).value_counts().median())
sampling_strategy = {cls: max(cnt, median_count)
                     for cls, cnt in pd.Series(train["issuetype"]).value_counts().items()}
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train_it_os, y_train_it_os = ros.fit_resample(X_train, train["issuetype"])
print(f"  Avant: {dict(pd.Series(train['issuetype']).value_counts())}")
print(f"  Apres: {dict(pd.Series(y_train_it_os).value_counts())}")

# Resolution : pas d'oversampling — Cannot Reproduce n'a que 528 exemples,
# dupliquer 55x cause de l'overfitting severe. class_weight='balanced' suffit.
print("Entrainement LogisticRegression (issuetype) ...")
t0 = time.time()
clf_it = LogisticRegression(
    max_iter=1000, C=1.0, solver="lbfgs", n_jobs=1
)
clf_it.fit(X_train_it_os, y_train_it_os)
print(f"  Done in {time.time()-t0:.0f}s")

print("Entrainement LogisticRegression (resolution) ...")
t0 = time.time()
clf_res = LogisticRegression(
    class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs", n_jobs=1
)
clf_res.fit(X_train_res, train["resolution"])
print(f"  Done in {time.time()-t0:.0f}s")

# ── 4b. Prédictions + confiances ─────────────────────────────────────────────
pred_issuetype  = clf_it.predict(X_val).tolist()
pred_resolution = clf_res.predict(X_val_res).tolist()

prob_it  = clf_it.predict_proba(X_val).max(axis=1).round(4).tolist()
prob_res = clf_res.predict_proba(X_val_res).max(axis=1).round(4).tolist()

# ── 4c. Build predictions dataframe ──────────────────────────────────────────
preds = val[["key", "issuetype", "resolution"]].copy()
preds.columns = ["key", "true_issuetype", "true_resolution"]
preds["pred_issuetype"]   = pred_issuetype
preds["pred_resolution"]  = pred_resolution
preds["conf_issuetype"]   = prob_it
preds["conf_resolution"]  = prob_res
preds["method"]           = "LR_BALANCED"
preds["fix_summary"]      = ""

# ── 5. Evaluate ─────────────────────────────────────────────────────────────
def evaluate(true, pred, label):
    acc = (np.array(true) == np.array(pred)).mean()
    f1  = f1_score(true, pred, average="macro", zero_division=0)
    print(f"\n  [{label}]  accuracy={acc:.4f}  macro-F1={f1:.4f}")
    return acc, f1

print("\n=== Evaluation ===")
acc_it,  f1_it  = evaluate(preds["true_issuetype"],  preds["pred_issuetype"],  "issuetype")
acc_res, f1_res = evaluate(preds["true_resolution"],  preds["pred_resolution"], "resolution")

# Per-class F1
it_labels = sorted(preds["true_issuetype"].unique())
it_f1_per_class = f1_score(preds["true_issuetype"], preds["pred_issuetype"],
                            labels=it_labels, average=None, zero_division=0)

res_labels = sorted(preds["true_resolution"].unique())
res_f1_per_class = f1_score(preds["true_resolution"], preds["pred_resolution"],
                             labels=res_labels, average=None, zero_division=0)

# Confusion matrices
cm_it  = confusion_matrix(preds["true_issuetype"],  preds["pred_issuetype"],  labels=it_labels)
cm_res = confusion_matrix(preds["true_resolution"],  preds["pred_resolution"], labels=res_labels)

# ── 6. Save results locally ─────────────────────────────────────────────────
print("\n=== Saving results ===")

# Predictions CSV
preds_path = RESULTS / "mart_predictions.csv"
preds.to_csv(preds_path, index=False, encoding="utf-8")
print(f"  mart_predictions.csv  ({len(preds):,} rows)")

# Confusion matrices
pd.DataFrame(cm_it,  index=it_labels,  columns=it_labels).to_csv(
    RESULTS / "confusion_matrix_issuetype.csv")
pd.DataFrame(cm_res, index=res_labels, columns=res_labels).to_csv(
    RESULTS / "confusion_matrix_resolution.csv")
print("  confusion_matrix_issuetype.csv")
print("  confusion_matrix_resolution.csv")

# Evaluation report
report_lines = [
    "=== PFE Spark Triage — Evaluation Report ===",
    f"Model : sentence-transformers/{model_name} + LogisticRegression(balanced) + {len(TABULAR_FEATURES)} tabular features",
    f"Train  : {len(train):,} tickets",
    f"Val    : {len(val):,} tickets",
    "",
    "--- issuetype ---",
    f"Accuracy : {acc_it:.4f}",
    f"Macro-F1 : {f1_it:.4f}",
    "Per-class F1:",
]
for lbl, score in zip(it_labels, it_f1_per_class):
    report_lines.append(f"  {lbl:<20} {score:.4f}")

report_lines += [
    "",
    "--- resolution ---",
    f"Accuracy : {acc_res:.4f}",
    f"Macro-F1 : {f1_res:.4f}",
    "Per-class F1:",
]
for lbl, score in zip(res_labels, res_f1_per_class):
    report_lines.append(f"  {lbl:<20} {score:.4f}")

report_lines += [
    "",
    f"Method: LR_BALANCED={len(preds):,}  (embeddings {train_emb_it.shape[1]}d + {len(TABULAR_FEATURES)} tabular features)",
]

report_text = "\n".join(report_lines)
(RESULTS / "evaluation_report.txt").write_text(report_text, encoding="utf-8")
print("  evaluation_report.txt")
print()
print(report_text)

# ── 7. Upload predictions to Snowflake ──────────────────────────────────────
# Reconnexion fraiche : la session initiale peut expirer apres de longs embeddings
print("\nUploading PREDICTIONS.MART_PREDICTIONS to Snowflake ...")
conn.close()
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH",
    database="PFE_SPARK",
    schema="PREDICTIONS",
)
cur = conn.cursor()
cur.execute("USE DATABASE PFE_SPARK")
cur.execute("USE SCHEMA PREDICTIONS")
cur.execute("""
    CREATE OR REPLACE TABLE PREDICTIONS.MART_PREDICTIONS (
        key               VARCHAR,
        true_issuetype    VARCHAR,
        true_resolution   VARCHAR,
        pred_issuetype    VARCHAR,
        pred_resolution   VARCHAR,
        conf_issuetype    FLOAT,
        conf_resolution   FLOAT,
        method            VARCHAR,
        fix_summary       VARCHAR
    )
""")

upload_df = preds.copy()
upload_df.columns = [c.upper() for c in upload_df.columns]
success, n_chunks, n_rows, _ = write_pandas(conn, upload_df, "MART_PREDICTIONS",
                                             schema="PREDICTIONS", database="PFE_SPARK")
print(f"  Uploaded {n_rows:,} rows  (success={success})")

conn.close()
print("\nPipeline complete. All results saved to results/")
