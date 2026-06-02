"""
Migration locale -> Snowflake en 3 etapes :
  Step 1 : Charger spark_parent_keys.csv -> RAW.SPARK_PARENT_KEYS
  Step 2 : Rafraichir MART_ML via dbt (select mart_ml)
  Step 3 : Verifier le resultat

Usage : python migrate_to_snowflake.py
"""
import os, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

ROOT        = Path(__file__).parent
CSV_PATH    = ROOT / "results" / "spark_parent_keys.csv"
DBT_DIR     = ROOT / "dbt_project"
DBT_BIN     = Path(r"C:\Users\aelkhabbaz\AppData\Local\anaconda3\Scripts\dbt.exe")

# ── Step 1 : Charger spark_parent_keys.csv ───────────────────────────────────
print("=" * 60)
print("STEP 1 — Upload spark_parent_keys.csv -> RAW.SPARK_PARENT_KEYS")
print("=" * 60)

if not CSV_PATH.exists():
    print(f"ERREUR : {CSV_PATH} introuvable.")
    print("Lancer d'abord : python fetch_parent_keys.py")
    sys.exit(1)

conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH",
    database="PFE_SPARK",
    schema="RAW",
)
cur = conn.cursor()

df = pd.read_csv(CSV_PATH)
df.columns = [c.upper() for c in df.columns]
df["HAS_PARENT"] = df["HAS_PARENT"].fillna(0).clip(0, 1).astype(int)
print(f"  CSV : {len(df):,} lignes — has_parent=1 : {df['HAS_PARENT'].sum():,}")

cur.execute("""
    CREATE OR REPLACE TABLE RAW.SPARK_PARENT_KEYS (
        KEY        VARCHAR,
        PARENT_KEY VARCHAR,
        HAS_PARENT INTEGER
    )
""")
success, _, n_rows, _ = write_pandas(
    conn, df, "SPARK_PARENT_KEYS", database="PFE_SPARK", schema="RAW"
)
print(f"  Upload : {n_rows:,} lignes  success={success}")
conn.close()
print()

# ── Step 2 : Rafraichir MART_ML via dbt ─────────────────────────────────────
print("=" * 60)
print("STEP 2 — dbt run --select mart_ml")
print("=" * 60)

if not DBT_BIN.exists():
    print(f"ERREUR : dbt introuvable dans {DBT_BIN}")
    print("Verifier que le venv dbt est installe dans dbt_project/.venv/")
    sys.exit(1)

result = subprocess.run(
    [str(DBT_BIN), "run", "--select", "mart_ml"],
    cwd=str(DBT_DIR),
    capture_output=False,
)
if result.returncode != 0:
    print("ERREUR : dbt run a echoue.")
    sys.exit(1)
print()

# ── Step 3 : Verification ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Verification MART_ML")
print("=" * 60)

conn2 = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH",
    database="PFE_SPARK",
    schema="MARTS_ML",
)
cur2 = conn2.cursor()
cur2.execute("""
    SELECT
        COUNT(*)                                  AS total,
        COUNT_IF(split = 'train')                 AS train,
        COUNT_IF(split = 'validation')            AS validation,
        COUNT_IF(has_parent = 1)                  AS avec_parent,
        COUNT_IF(issuetype = 'Sub-task' AND has_parent = 1) AS subtask_avec_parent
    FROM MARTS_ML.MART_ML
""")
row = cur2.fetchone()
conn2.close()

total, train, val, avec_parent, subtask_parent = row
print(f"  Total         : {total:>7,}")
print(f"  Train         : {train:>7,}")
print(f"  Validation    : {val:>7,}")
print(f"  has_parent=1  : {avec_parent:>7,}  ({avec_parent/total*100:.1f}%)")
print(f"  Sub-task OK   : {subtask_parent:>7,}  (attendu ~7 277)")
print()

if avec_parent > 0:
    print("MIGRATION MART_ML REUSSIE")
else:
    print("ATTENTION : has_parent=0 partout — verifier RAW.SPARK_PARENT_KEYS")
