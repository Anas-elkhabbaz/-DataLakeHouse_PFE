"""
Récupère le champ parent_key pour tous les tickets SPARK depuis l'API Apache JIRA.
L'API est publique, aucune authentification requise.

Durée : ~15-20 min (42K tickets, 10 threads parallèles)
Résultat : results/spark_parent_keys.csv  (key, parent_key, has_parent)
"""
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "spark_parent_keys.csv"

# ── Charger les clés depuis Snowflake ────────────────────────────────────
print("Chargement des clés depuis Snowflake...")
import snowflake.connector
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH", database="PFE_SPARK",
)
cur = conn.cursor()
cur.execute("SELECT DISTINCT key FROM PFE_SPARK.MARTS_ML.MART_ML ORDER BY key")
all_keys = [r[0] for r in cur.fetchall()]
conn.close()
print(f"  {len(all_keys):,} tickets à traiter")

# Reprendre depuis où on s'était arrêté si le fichier existe déjà
if OUT.exists():
    existing = pd.read_csv(OUT)
    done_keys = set(existing["key"].tolist())
    all_keys  = [k for k in all_keys if k not in done_keys]
    print(f"  {len(done_keys):,} déjà traités, {len(all_keys):,} restants")
else:
    existing = pd.DataFrame(columns=["key", "parent_key", "has_parent"])

# ── Fonction d'appel API ─────────────────────────────────────────────────
BASE_URL = "https://issues.apache.org/jira/rest/api/2/issue"

def fetch_parent(key: str) -> dict:
    """Retourne {"key": key, "parent_key": "SPARK-XXXX" ou None, "has_parent": 0/1}"""
    for attempt in range(3):
        try:
            url = f"{BASE_URL}/{key}?fields=parent,issuetype"
            r   = requests.get(url, timeout=10)
            if r.status_code == 200:
                fields = r.json().get("fields", {})
                parent = fields.get("parent")
                pk     = parent["key"] if parent else None
                return {"key": key, "parent_key": pk, "has_parent": int(pk is not None)}
            elif r.status_code == 429:   # rate limit
                time.sleep(2 * (attempt + 1))
            elif r.status_code == 404:
                return {"key": key, "parent_key": None, "has_parent": 0}
        except Exception:
            time.sleep(1)
    return {"key": key, "parent_key": None, "has_parent": -1}  # -1 = échec

# ── Exécution parallèle ──────────────────────────────────────────────────
WORKERS    = 10    # 10 threads parallèles
BATCH_SIZE = 500   # sauvegarder toutes les 500 clés

results = []
t0      = time.time()
processed = 0

print(f"Démarrage avec {WORKERS} threads parallèles...")

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(fetch_parent, k): k for k in all_keys}

    for future in as_completed(futures):
        row = future.result()
        results.append(row)
        processed += 1

        if processed % 100 == 0:
            elapsed = time.time() - t0
            rate    = processed / elapsed
            eta     = (len(all_keys) - processed) / rate / 60
            print(f"  {processed:>5}/{len(all_keys)}  "
                  f"({processed/len(all_keys)*100:.1f}%)  "
                  f"rate={rate:.1f} req/s  ETA={eta:.1f} min")

        # Sauvegarde intermédiaire toutes les 500 requêtes
        if len(results) >= BATCH_SIZE:
            chunk = pd.DataFrame(results)
            existing = pd.concat([existing, chunk], ignore_index=True)
            existing.to_csv(OUT, index=False)
            results = []
            print(f"  Sauvegarde intermédiaire -> {OUT}  ({len(existing):,} lignes)")

# Sauvegarder le reste
if results:
    chunk = pd.DataFrame(results)
    existing = pd.concat([existing, chunk], ignore_index=True)
    existing.to_csv(OUT, index=False)

print(f"\nTerminé en {(time.time()-t0)/60:.1f} min")
print(f"Fichier : {OUT}  ({len(existing):,} lignes)")
print(f"\nDistribution has_parent :")
print(existing["has_parent"].value_counts().to_string())

# Stats par rapport à issuetype dans MART_ML
print("\nCorrection : % de Sub-tasks avec has_parent=1 ?")
import snowflake.connector
conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ["SNOWFLAKE_ROLE"],
    warehouse="PFE_WH", database="PFE_SPARK",
)
df_mart = pd.read_sql(
    "SELECT key, issuetype FROM PFE_SPARK.MARTS_ML.MART_ML",
    conn
)
conn.close()

df_merged = df_mart.merge(existing[["key","has_parent"]], on="key", how="left")
df_merged["has_parent"] = df_merged["has_parent"].fillna(0)

print(f"\n  % has_parent=1 par issuetype :")
stats = df_merged.groupby("issuetype")["has_parent"].mean() * 100
print(stats.sort_values(ascending=False).to_string())
