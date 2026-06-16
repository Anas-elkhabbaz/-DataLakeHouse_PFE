"""
Références similaires de secours (fallback).

Utilisé par l'application Streamlit de triage lorsque la requête vers
PFE_SPARK.MARTS_ML.MART_ML n'est pas disponible (Snowflake non configuré,
table absente, ou erreur réseau). Retourne un petit jeu de tickets de
référence représentatifs du type d'incident prédit, clairement marqués
comme données de secours (`source = "fallback"`).

API publique :
    build_similar_reference_rows(issue_type, summary="", description="", limit=3)
        -> pandas.DataFrame
        colonnes : key, summary_clean, issuetype, resolution, similarity, source
"""

import pandas as pd

# Exemples représentatifs par type d'incident (libellés génériques, non liés
# à un ticket réel — affichés uniquement en mode dégradé).
_REFERENCE_LIBRARY = {
    "Bug": [
        ("SPARK-REF-BUG-1", "Memory leak in TaskSetManager during long-running jobs", "Fixed"),
        ("SPARK-REF-BUG-2", "NullPointerException thrown when reading empty Parquet file", "Fixed"),
        ("SPARK-REF-BUG-3", "Executor crashes with OutOfMemoryError under high shuffle load", "Fixed"),
    ],
    "Improvement": [
        ("SPARK-REF-IMP-1", "Improve performance of DataFrame groupBy aggregation", "Fixed"),
        ("SPARK-REF-IMP-2", "Reduce memory footprint of broadcast joins", "Fixed"),
        ("SPARK-REF-IMP-3", "Optimize shuffle partition handling for skewed data", "Fixed"),
    ],
    "New Feature": [
        ("SPARK-REF-NF-1", "Add support for new file format in DataSource API", "Fixed"),
        ("SPARK-REF-NF-2", "Expose configuration option for adaptive query execution", "Fixed"),
        ("SPARK-REF-NF-3", "Provide built-in function for array transformations", "Incomplete"),
    ],
    "Sub-task": [
        ("SPARK-REF-ST-1", "Sub-task: implement unit tests for the new optimizer rule", "Fixed"),
        ("SPARK-REF-ST-2", "Sub-task: update documentation for the parent feature", "Fixed"),
        ("SPARK-REF-ST-3", "Sub-task: refactor helper used by the parent ticket", "Fixed"),
    ],
    "Task": [
        ("SPARK-REF-TSK-1", "Upgrade dependency to the latest stable version", "Fixed"),
        ("SPARK-REF-TSK-2", "Clean up deprecated configuration keys", "Fixed"),
        ("SPARK-REF-TSK-3", "Migrate build scripts to the new tooling", "Fixed"),
    ],
}

# Repli générique quand le type n'est pas reconnu.
_GENERIC = [
    ("SPARK-REF-GEN-1", "Investigate reported behaviour and gather reproduction steps", "Incomplete"),
    ("SPARK-REF-GEN-2", "Clarify expected behaviour with the reporter", "Not A Problem"),
    ("SPARK-REF-GEN-3", "Check whether the issue duplicates an existing ticket", "Duplicate"),
]

_COLUMNS = ["key", "summary_clean", "issuetype", "resolution", "similarity", "source"]


def build_similar_reference_rows(issue_type, summary="", description="", limit=3):
    """Construit un DataFrame de références de secours pour un type d'incident.

    Les lignes sont marquées `source = "fallback"`. La colonne `similarity`
    porte des valeurs décroissantes purement indicatives (aucun calcul réel
    n'est effectué en mode dégradé).
    """
    key = str(issue_type).strip()
    # Normalisation tolérante (casse / variantes proches).
    lookup = {k.lower(): k for k in _REFERENCE_LIBRARY}
    examples = _REFERENCE_LIBRARY.get(lookup.get(key.lower(), key), _GENERIC)

    rows = []
    for i, (ref_key, ref_summary, ref_resolution) in enumerate(examples[:max(1, int(limit))]):
        rows.append({
            "key": ref_key,
            "summary_clean": ref_summary,
            "issuetype": key or "Other",
            "resolution": ref_resolution,
            "similarity": round(0.9 - 0.1 * i, 3),
            "source": "fallback",
        })

    return pd.DataFrame(rows, columns=_COLUMNS)
