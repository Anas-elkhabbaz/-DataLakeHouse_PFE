"""
Deploie les notebooks dans Snowflake Notebooks (Snowsight).

Notebooks deployes :
  1. sklearn_resolution_training  — entrainement resolution + Model Registry
  2. deberta_fine_tuning_v3       — notebook Kaggle DeBERTa v3 (reference)

Usage : python deploy_notebooks.py
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from snowflake.snowpark import Session

ROOT         = Path(__file__).parent
KAGGLE_NB    = ROOT / "results" / "notebooks" / "deberta_fine_tuning_v3.ipynb"
NB_OUT_DIR   = ROOT / "results" / "notebooks"
NB_OUT_DIR.mkdir(parents=True, exist_ok=True)

session = Session.builder.configs({
    "account":   os.environ["SNOWFLAKE_ACCOUNT"],
    "user":      os.environ["SNOWFLAKE_USER"],
    "password":  os.environ["SNOWFLAKE_PASSWORD"],
    "role":      os.environ["SNOWFLAKE_ROLE"],
    "warehouse": "PFE_WH",
    "database":  "PFE_SPARK",
    "schema":    "ML_MODELS",
}).create()

# ── Helpers ──────────────────────────────────────────────────────────────────
def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }

def make_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }

def build_ipynb(cells):
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }

# ── Notebook 1 : sklearn resolution training ─────────────────────────────────
print("Creation notebook sklearn_resolution_training.ipynb...")

cells_sklearn = [
    make_md_cell([
        "# PFE Spark — Entraînement de la résolution (modèle hybride)\n",
        "\n",
        "Entraîne la **régression logistique hybride** décrite dans le rapport pour prédire la **résolution** des tickets Apache Spark.\n",
        "\n",
        "Approche : embeddings sémantiques `all-mpnet-base-v2` (768 d) sur `text_for_res`, **concaténés** aux caractéristiques tabulaires de `MARTS_ML.MART_ML`, puis `LogisticRegression(class_weight='balanced', C=3.0)`.\n",
        "\n",
        "Ce notebook est l'équivalent reproductible de `load/train_save_resolution_model.py` (qui produit le `.pkl` déployé et `results/evaluation_report.txt`).",
    ]),
    make_md_cell([
        "> **Note de cohérence rapport ↔ code (17 vs 19 features).**\n",
        "> Ce notebook entraîne le modèle **tel qu'il a réellement tourné** : embeddings (768 d) + **19** caractéristiques tabulaires (incluant `resolution_days` et `n_resolution_changes`), soit un vecteur de 787 dimensions, d'où les métriques **81,3 % d'accuracy / 26,9 % de macro-F1**.\n",
        "> Le **rapport documente 17 caractéristiques** : il exclut volontairement de la *spécification* ces deux variables, identifiées comme une **fuite directe de la cible** (jamais disponibles au triage d'un nouveau ticket). L'écart 17 (rapport) vs 19 (code) est donc **assumé et documenté** ; un ré-entraînement strict à 17 features constitue l'étape suivante si l'on souhaite des métriques 17-features.",
    ]),
    make_code_cell([
        "import pandas as pd\n",
        "import numpy as np\n",
        "from snowflake.snowpark.context import get_active_session\n",
        "from sentence_transformers import SentenceTransformer  # ajouter 'sentence-transformers' aux packages du notebook\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import f1_score, classification_report\n",
        "import json\n",
        "\n",
        "session = get_active_session()\n",
        "print('Session Snowflake active :', session.get_current_database())",
    ]),
    make_md_cell("## 1. Chargement de MART_ML (texte + 19 features tabulaires)"),
    make_code_cell([
        "# 19 caractéristiques tabulaires (telles qu'utilisées par train_save_resolution_model.py).\n",
        "# resolution_days et n_resolution_changes sont incluses ici (modèle tel qu'il a tourné) ;\n",
        "# le rapport documente les 17 features hors fuite de cible — voir note ci-dessus.\n",
        "TABULAR_FEATURES = [\n",
        "    'n_total_changes', 'n_status_changes', 'n_priority_changes',\n",
        "    'n_assignee_changes', 'n_resolution_changes', 'was_escalated',\n",
        "    'n_people_involved', 'n_links_total', 'n_duplicates', 'n_blocks',\n",
        "    'n_blocked_by', 'n_relates', 'n_comments', 'n_commenters',\n",
        "    'resolution_days', 'summary_length', 'description_length',\n",
        "    'n_container', 'has_parent',\n",
        "]\n",
        "feat_str = ', '.join(TABULAR_FEATURES)\n",
        "\n",
        "df = session.sql(f\"\"\"\n",
        "    SELECT key, split, issuetype, resolution, text_for_res, {feat_str}\n",
        "    FROM PFE_SPARK.MARTS_ML.MART_ML\n",
        "    WHERE split IN ('train', 'validation')\n",
        "    ORDER BY key\n",
        "\"\"\").to_pandas()\n",
        "df.columns = [c.lower() for c in df.columns]\n",
        "print(f'Chargé {len(df):,} tickets')\n",
        "print(df['split'].value_counts())",
    ]),
    make_md_cell("## 2. Branche textuelle — embeddings all-mpnet-base-v2 (768 d)"),
    make_code_cell([
        "train = df[df['split'] == 'train'].reset_index(drop=True)\n",
        "val   = df[df['split'] == 'validation'].reset_index(drop=True)\n",
        "\n",
        "model = SentenceTransformer('all-mpnet-base-v2')\n",
        "\n",
        "def embed(texts):\n",
        "    return model.encode(texts, batch_size=256, show_progress_bar=True,\n",
        "                        normalize_embeddings=True)\n",
        "\n",
        "train_emb_res = embed(train['text_for_res'].fillna('').tolist())\n",
        "val_emb_res   = embed(val['text_for_res'].fillna('').tolist())\n",
        "print('Embeddings :', train_emb_res.shape, val_emb_res.shape)",
    ]),
    make_md_cell("## 3. Branche tabulaire + concaténation (vecteur 768 + 19 = 787 d)"),
    make_code_cell([
        "scaler    = StandardScaler()\n",
        "train_tab = scaler.fit_transform(train[TABULAR_FEATURES].fillna(0).clip(upper=1e6))\n",
        "val_tab   = scaler.transform(val[TABULAR_FEATURES].fillna(0).clip(upper=1e6))\n",
        "\n",
        "X_train_res = np.hstack([train_emb_res, train_tab])\n",
        "X_val_res   = np.hstack([val_emb_res,   val_tab])\n",
        "print('X_train_res :', X_train_res.shape, '| X_val_res :', X_val_res.shape)",
    ]),
    make_md_cell("## 4. Entraînement LogisticRegression (résolution)"),
    make_code_cell([
        "clf_res = LogisticRegression(\n",
        "    class_weight='balanced', max_iter=1000, C=3.0, solver='lbfgs', n_jobs=-1\n",
        ")\n",
        "clf_res.fit(X_train_res, train['resolution'])\n",
        "\n",
        "pred = clf_res.predict(X_val_res)\n",
        "f1   = f1_score(val['resolution'], pred, average='macro', zero_division=0)\n",
        "acc  = (pred == val['resolution'].values).mean()\n",
        "print(f'Résolution — accuracy={acc:.4f}  macro-F1={f1:.4f}')\n",
        "print(classification_report(val['resolution'], pred, zero_division=0))",
    ]),
    make_md_cell("## 5. Sauvegarde dans le Snowflake Model Registry"),
    make_code_cell([
        "from snowflake.ml.registry import Registry\n",
        "\n",
        "reg = Registry(session=session)\n",
        "reg.log_model(\n",
        "    clf_res,\n",
        "    model_name='resolution_classifier',\n",
        "    version_name='v2_hybrid_mpnet_768_plus_19',\n",
        "    conda_dependencies=['scikit-learn', 'pandas', 'numpy'],\n",
        "    comment=f'LR balanced C=3.0, embeddings all-mpnet-base-v2 (768d) + 19 features tabulaires. accuracy={acc:.4f} macro-F1={f1:.4f}'\n",
        ")\n",
        "print('Modèle hybride sauvegardé dans le Model Registry.')",
    ]),
]

nb_sklearn = build_ipynb(cells_sklearn)
sklearn_path = NB_OUT_DIR / "sklearn_resolution_training.ipynb"
sklearn_path.write_text(json.dumps(nb_sklearn, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  Créé : {sklearn_path}")

# ── Notebook 2 : DeBERTa fine-tuning v3 (notebook Kaggle tel quel) ───────────
print("Preparation notebook deberta_fine_tuning_v3.ipynb...")

if KAGGLE_NB.exists():
    import shutil
    deberta_path = NB_OUT_DIR / "deberta_fine_tuning_v3.ipynb"
    if KAGGLE_NB.resolve() != deberta_path.resolve():
        shutil.copy2(KAGGLE_NB, deberta_path)
        print(f"  Copié : {deberta_path}")
    else:
        print(f"  Déjà en place : {deberta_path}")
else:
    print(f"  ATTENTION : {KAGGLE_NB} introuvable, skip.")
    deberta_path = None

# ── Upload vers le stage Snowflake ───────────────────────────────────────────
print("\nCreation stage notebooks_stage si absent...")
session.sql("CREATE SCHEMA IF NOT EXISTS PFE_SPARK.ML_MODELS").collect()
session.sql("""
    CREATE STAGE IF NOT EXISTS PFE_SPARK.ML_MODELS.notebooks_stage
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    DIRECTORY  = (ENABLE = TRUE)
    COMMENT    = 'Notebooks PFE Spark'
""").collect()
print("  OK")

print("\nUpload notebooks -> @ML_MODELS.notebooks_stage...")
for nb_file in NB_OUT_DIR.glob("*.ipynb"):
    session.file.put(
        str(nb_file),
        "@PFE_SPARK.ML_MODELS.notebooks_stage/",
        auto_compress=False, overwrite=True,
    )
    size_kb = nb_file.stat().st_size // 1024
    print(f"  {nb_file.name}  ({size_kb} KB)  OK")

# ── Création des Snowflake Notebooks via SQL ─────────────────────────────────
print("\nCreation des Snowflake Notebooks...")

notebooks = [
    ("sklearn_resolution_training",  "sklearn_resolution_training.ipynb",
     "PFE Spark — Entraînement sklearn résolution + Model Registry"),
    ("deberta_fine_tuning_v3",        "deberta_fine_tuning_v3.ipynb",
     "PFE Spark — DeBERTa v3 fine-tuning (reference Kaggle)"),
]

created = []
for nb_name, nb_file, comment in notebooks:
    if not (NB_OUT_DIR / nb_file).exists():
        print(f"  SKIP {nb_name} (fichier absent)")
        continue
    try:
        session.sql(f"""
            CREATE OR REPLACE NOTEBOOK PFE_SPARK.ML_MODELS.{nb_name}
            FROM '@PFE_SPARK.ML_MODELS.notebooks_stage'
            MAIN_FILE = '{nb_file}'
            QUERY_WAREHOUSE = 'PFE_WH'
            COMMENT = '{comment}'
        """).collect()
        print(f"  OK {nb_name}")
        created.append(nb_name)
    except Exception as e:
        print(f"  ERR {nb_name} -- {str(e)[:80]}")

# ── Instructions manuelles si SQL ne fonctionne pas ─────────────────────────
if len(created) < len([n for n,f,_ in notebooks if (NB_OUT_DIR/f).exists()]):
    print("\n" + "="*60)
    print("IMPORT MANUEL dans Snowsight (si SQL a echoue) :")
    print("="*60)
    print("  1. Snowsight → Projects → Notebooks → + Notebook")
    print("  2. Choisir 'Upload .ipynb file'")
    for nb_name, nb_file, _ in notebooks:
        p = NB_OUT_DIR / nb_file
        if p.exists():
            print(f"  3. Importer : {p}")
    print("  4. Assigner warehouse : PFE_WH")

# ── Résumé stage ─────────────────────────────────────────────────────────────
print("\nContenu @notebooks_stage :")
rows = session.sql("LIST @PFE_SPARK.ML_MODELS.notebooks_stage").collect()
for r in rows:
    print(f"  {r[0]}  ({int(r[1])//1024} KB)")

session.close()
print("\nTerminé.")
