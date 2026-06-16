"""
Deploiement complet de l'app Streamlit in Snowflake.
Gere : schema, stage, models (sklearn + DeBERTa), environment.yml, code.

Usage :
  1. python load/train_save_resolution_model.py   # genere results/sklearn_models/
  2. python deploy_streamlit_snowflake.py         # deploie tout dans Snowflake
"""
import os, zipfile, shutil, tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from snowflake.snowpark import Session

ROOT     = Path(__file__).parent
APP_DIR  = ROOT / "apps" / "inference"
RESULTS  = ROOT / "results"
SKLEARN_DIR  = RESULTS / "sklearn_models"
DEBERTA_ZIP  = RESULTS / "deberta_v3_parent.zip"

session = Session.builder.configs({
    "account":   os.environ["SNOWFLAKE_ACCOUNT"],
    "user":      os.environ["SNOWFLAKE_USER"],
    "password":  os.environ["SNOWFLAKE_PASSWORD"],
    "role":      os.environ["SNOWFLAKE_ROLE"],
    "warehouse": "PFE_WH",
    "database":  "PFE_SPARK",
    "schema":    "ML_MODELS",
}).create()

# ── 0. Schema + Stage ────────────────────────────────────────────────────────
print("Creation schema ML_MODELS si absent...")
session.sql("CREATE SCHEMA IF NOT EXISTS PFE_SPARK.ML_MODELS").collect()

print("Creation stage app_stage si absent...")
session.sql("""
    CREATE STAGE IF NOT EXISTS PFE_SPARK.ML_MODELS.app_stage
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    DIRECTORY  = (ENABLE = TRUE)
    COMMENT    = 'Streamlit app + modeles ML'
""").collect()
print("  OK\n")

# ── 1. Upload code + environment.yml ─────────────────────────────────────────
print("Upload streamlit_in_snowflake.py...")
session.file.put(
    str(APP_DIR / "streamlit_in_snowflake.py"),
    "@PFE_SPARK.ML_MODELS.app_stage/",
    auto_compress=False, overwrite=True
)
print("  OK")

print("Upload similar_reference_utils.py...")
session.file.put(
    str(APP_DIR / "similar_reference_utils.py"),
    "@PFE_SPARK.ML_MODELS.app_stage/",
    auto_compress=False, overwrite=True
)
print("  OK")

print("Upload environment.yml...")
session.file.put(
    str(APP_DIR / "environment.yml"),
    "@PFE_SPARK.ML_MODELS.app_stage/",
    auto_compress=False, overwrite=True
)
print("  OK\n")

# ── 2. Upload modeles sklearn ─────────────────────────────────────────────────
if SKLEARN_DIR.exists():
    print("Upload modeles sklearn -> stage/models/...")
    for f in SKLEARN_DIR.iterdir():
        if f.suffix in (".pkl", ".json"):
            session.file.put(
                str(f),
                "@PFE_SPARK.ML_MODELS.app_stage/models/",
                auto_compress=False, overwrite=True
            )
            print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
    print("  OK\n")
else:
    print("ATTENTION : results/sklearn_models/ absent.")
    print("  Lancer d'abord : python load/train_save_resolution_model.py\n")

# ── 3. Upload modele DeBERTa (extraction depuis le zip) ──────────────────────
if DEBERTA_ZIP.exists():
    print("Extraction + upload DeBERTa -> stage/deberta/...")
    tmp_dir = Path(tempfile.mkdtemp()) / "deberta_extracted"

    with zipfile.ZipFile(DEBERTA_ZIP, "r") as z:
        z.extractall(tmp_dir)

    uploaded = 0
    for f in tmp_dir.rglob("*"):
        if f.is_file():
            session.file.put(
                str(f),
                "@PFE_SPARK.ML_MODELS.app_stage/deberta/",
                auto_compress=False, overwrite=True
            )
            print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
            uploaded += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  {uploaded} fichiers uploades\n")
else:
    print("ATTENTION : results/deberta_v3_parent.zip absent — DeBERTa non disponible.\n")

# ── 4. Creer / remplacer l'app Streamlit ─────────────────────────────────────
print("Creation / remplacement de l'app Streamlit in Snowflake...")
try:
    session.sql("""
        CREATE OR REPLACE STREAMLIT PFE_SPARK.ML_MODELS.spark_triage_app
        ROOT_LOCATION = '@PFE_SPARK.ML_MODELS.app_stage'
        MAIN_FILE     = 'streamlit_in_snowflake.py'
        QUERY_WAREHOUSE = 'PFE_WH'
        COMMENT = 'PFE 2026 — Apache Spark Ticket Triage (DeBERTa + sklearn)'
    """).collect()
    print("  App creee !")
except Exception as e:
    print(f"  Erreur CREATE STREAMLIT: {e}")

# ── 5. Verification finale ────────────────────────────────────────────────────
print("\nContenu du stage:")
rows = session.sql("LIST @PFE_SPARK.ML_MODELS.app_stage").collect()
for r in rows:
    name = r[0]
    size = int(r[1]) / 1e6
    print(f"  {name:<60}  {size:.1f} MB")

print("\nOuvrir dans Snowsight : Projects -> Streamlit -> spark_triage_app")
session.close()
