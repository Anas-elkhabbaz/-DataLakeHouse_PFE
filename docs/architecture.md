# Architecture de la plateforme — PFE Spark Triage

## Vue d'ensemble

La plateforme adopte une **architecture en médaillon** (Bronze → Argent → Or) hébergée sur
Snowflake et orchestrée par dbt-snowflake. L'inférence est exposée via une application
**Streamlit-in-Snowflake** (exécutée nativement dans l'entrepôt), et l'analytique via un tableau de
bord **Power BI** connecté nativement à Snowflake. Le pipeline d'inférence utilise des embeddings de
phrases locaux (`sentence-transformers`) car les fonctions Snowflake Cortex sont bloquées sur le
compte d'essai utilisé pour la reproduction.

---

## Schéma d'architecture

```
Fichiers CSV source (Kaggle, mars 2025)
  issues.csv · comments.csv · changelog.csv · issuelinks.csv
        │
        ▼  python load/run_phase1.py   →  Snowflake: PFE_SPARK + 7 schemas + PFE_WH
        ▼  python load/03_put_files.py →  PUT vers @RAW.CSV_STAGE
        ▼  python load/run_phase4.py   →  COPY INTO tables brutes
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Snowflake — PFE_SPARK                                                              │
│                                                                                    │
│  BRONZE (RAW)                                                                      │
│  RAW.ISSUES (1 149 321) · RAW.COMMENTS (5 047 714)                                │
│  RAW.CHANGELOG (9 653 526) · RAW.ISSUELINKS (390 063)                             │
│  Tout en VARCHAR — aucune transformation de type                                   │
│       ▼  dbt run — staging/ (vues)                                                 │
│  ARGENT — STAGING                                                                  │
│  STG_ISSUES (49 832 tickets SPARK) · STG_COMMENTS · STG_CHANGELOG · STG_ISSUELINKS │
│       ▼  dbt run — intermediate/ (tables)                                          │
│  ARGENT — INTERMEDIATE                                                             │
│  INT_ISSUES_CLEANED · INT_COMMENTS_AGGREGATED · INT_CHANGELOG_FEATURES            │
│  INT_ISSUELINKS_FEATURES · INT_ISSUES_ANALYTICS · INT_STATUS_TRANSITIONS · ...    │
│       ▼  dbt run — marts/ (tables)                                                 │
│  OR                                                                                │
│  MARTS_ML.MART_ML (42 083)                                                         │
│  MARTS_ANALYTICS.* (7 tables : ops, workload, links, transitions, versions, ...)  │
│  PREDICTIONS.MART_PREDICTIONS (3 809)                                              │
└───────────────────────────────────────────────────────────────────────────────────┘
        │                                          │
        ▼ Chemin 2 — Power BI                      ▼ Chemin 1 — Pipeline ML (Python)
  Connecteur natif Snowflake                 python load/run_ml_pipeline.py
  Tables MARTS_ANALYTICS.*                    1. Fetch MART_ML + spark_parent_keys.csv
  (volumes, résolution, charge,              2. Issuetype : DeBERTa-v3-base fine-tuné
   liens, versions, calendrier)                 text_for_it + [HAS-PARENT] → 4 classes
                                             3. Résolution : LogisticRegression
                                                all-mpnet-base-v2 (768d) + features tabulaires
                                             4. Confiance + explication « Que faire ? » (Gemini)
                                             5. Évaluation + upload PREDICTIONS.MART_PREDICTIONS
                                                          │
                                                          ▼
                                          Streamlit-in-Snowflake (SPARK_TRIAGE_APP)
                                          déployé via deploy_streamlit_snowflake.py
```

---

## Couche Bronze (RAW)

**Rôle :** Ingestion fidèle des CSV Kaggle. Zéro transformation métier.

- Tables entièrement en VARCHAR pour absorber toute variation de format CSV
- Chargement via `COPY INTO` avec mapping positionnel `$N` documenté dans `load/04_copy_into_raw.sql`
- Script `inspect_headers.py` affiche les en-têtes réels pour vérifier le mapping avant chargement
- `run_phase4.py` exécute le COPY INTO et vérifie les comptes attendus

| Table | Lignes chargées |
|-------|----------------|
| RAW.ISSUES | 1 149 321 |
| RAW.COMMENTS | 5 047 714 |
| RAW.CHANGELOG | 9 653 526 |
| RAW.ISSUELINKS | 390 063 |
| dont issues SPARK | 49 832 |

---

## Couche Argent (STAGING + INTERMEDIATE)

### Staging — vues dbt

Transformations mécaniques uniquement (sans logique métier) :

- Filtre `project_key = 'SPARK'` (STG_ISSUES)
- Renommage des colonnes
- Cast des horodatages en `TIMESTAMP_TZ` via `TRY_TO_TIMESTAMP_TZ`
- `QUALIFY ROW_NUMBER() OVER (PARTITION BY key ...) = 1` pour dédupliquer les clés en double

### Intermediate — tables dbt

Logique métier et feature engineering :

**INT_ISSUES_CLEANED (45 043 lignes)**
- Nettoyage NLP en 6 étapes via macro `clean_jira_text` (HTML, blocs `{code}`/`{noformat}`, mentions, URLs, espaces)
- Consolidation des labels via `seeds/issuetype_mapping.csv` (9 classes) et `seeds/resolution_mapping.csv` (7 classes)
- Split temporel : `train` (<2023) / `validation` (2023) / `excluded` (≥2024) ; `resolution_days` plafonné à 5 000

**INT_COMMENTS_AGGREGATED (41 986)** — nettoyage + LISTAGG par ticket (tronqué à 3 000 caractères)
**INT_CHANGELOG_FEATURES (29 937)** — `was_escalated`, `n_status_changes`, `n_priority_changes`, `n_assignee_changes`, `n_people_involved`, ...
**INT_ISSUELINKS_FEATURES (11 179)** — `n_links_total`, `n_duplicates`, `n_blocks`, `n_blocked_by`, `n_relates`, `n_container`
**INT_ISSUES_ANALYTICS / INT_STATUS_TRANSITIONS / INT_DAILY_ACTIVITY / INT_FIX_VERSIONS** — sources des marts analytiques

---

## Couche Or (MARTS)

### MARTS_ML.MART_ML — 42 083 lignes

Table de contrat pour le pipeline d'inférence. Jointure large des tables intermédiaires + signal
`has_parent`. Filtrée sur `split IN ('train', 'validation')`.

| Partition | Lignes |
|-----------|--------|
| train | 38 274 |
| validation | 3 809 |

### MARTS_ANALYTICS (7 tables — source Power BI)

`MART_ANALYTICS_OPS`, `MART_ANALYTICS_WORKLOAD`, `MART_ANALYTICS_LINKS`,
`MART_ANALYTICS_TRANSITIONS_SANKEY`, `MART_ANALYTICS_TRANSITIONS_TIME`,
`MART_ANALYTICS_VERSIONS`, `MART_ANALYTICS_CALENDAR`. Incluent l'ensemble des tickets (y compris ouverts).

### PREDICTIONS.MART_PREDICTIONS — 3 809 lignes

Résultats d'évaluation du pipeline sur la validation (peuplée par `load/run_ml_pipeline.py`).

---

## Pipeline d'inférence

### Modèle issuetype — DeBERTa-v3-base fine-tuné (v3)

| Propriété | Valeur |
|-----------|--------|
| Modèle de base | `microsoft/deberta-v3-base` |
| Tokenizer | SentencePiece (max_length=256) |
| Classes | 4 : Bug, Improvement, Sub-task, Other |
| Epochs | 5 (best à epoch 3) |
| Optimizer | AdamW lr=2e-5, weight_decay=0,01 |
| Scheduler | Linear warmup (10 % des steps) |
| Infrastructure | Kaggle, GPU Tesla T4 |
| Sortie | `results/deberta_v3_parent.zip` (~599 MB) |

**Feature clé — `has_parent` :** récupéré via l'API JIRA (`spark_parent_keys.csv`), corrélation
quasi-parfaite avec Sub-task (100 % has_parent=1 pour Sub-task, 0 % pour Bug/Improvement). Injecté
en tête de séquence sous la forme `[HAS-PARENT]` / `[NO-PARENT]`, plus signaux lexicaux optionnels
`[BUG-SIGNAL]` / `[IMPROVEMENT-SIGNAL]`.

**Class weights (balanced) :** Bug 0,64 · Improvement 0,86 · Sub-task 1,31 · Other 1,99.

### Modèle résolution — LogisticRegression hybride

| Propriété | Valeur |
|-----------|--------|
| Modèle | `LogisticRegression(class_weight='balanced', C=3.0)` (scikit-learn) |
| Embeddings | `all-mpnet-base-v2` (sentence-transformers, 768d) sur `text_for_res` |
| Features tabulaires | **17** documentées (les 19 du code moins `resolution_days` et `n_resolution_changes`, écartées car fuite de cible) |
| Entrée | embeddings (768) ⊕ features tabulaires standardisées |
| Cache embeddings | `results/embeddings_cache.npz` (Git LFS) |

> Le modèle entraîné (`results/sklearn_models/`) utilise les 19 features ; le rapport documente le
> jeu leakage-free de 17. Voir `docs/decisions_log.md`.

### Explication « Que faire ? »

Générée par l'**API Google Gemini** (`gemini-2.0-flash`) ; un **moteur de règles contextuel** sert
de repli déterministe si la clé `GEMINI_API_KEY` est absente ou l'API indisponible.

### Résultats (validation, 3 809 tickets)

| Cible | Modèle | Accuracy | Macro-F1 |
|-------|--------|----------|----------|
| issuetype (4 classes) | DeBERTa-v3-base fine-tuné | **79,6 %** | **73,63 %** |
| résolution (7 classes) | LogisticRegression (all-mpnet-base-v2) | 81,3 % | 26,9 % |

> Sub-task F1 = 1,00 grâce au signal `has_parent`. La résolution est un **résultat partiel assumé**
> (macro-F1 faible dû au déséquilibre extrême, *Fixed* ≈ 90 % de la validation).

---

## Tests dbt

Suite de **66 tests de données** (unicité des clés, non-nullité, `accepted_values`, `relationships`),
exécutée nativement sur Snowflake via l'intégration Git des Workspaces : **PASS=66, WARN=0, ERROR=0**.
(En local, l'apostrophe de « Won't Fix » générait 2 warnings sur les tests `accepted_values` ;
l'exécution native sur Snowflake les lève.)

---

## Déploiement (Streamlit-in-Snowflake)

L'application d'inférence s'exécute nativement dans Snowflake. Déploiement via
`deploy_streamlit_snowflake.py` :

1. Crée le schéma `ML_MODELS` et le stage `APP_STAGE`
2. Uploade `apps/inference/streamlit_in_snowflake.py`, `similar_reference_utils.py`, `environment.yml`
3. Uploade les modèles sklearn (`results/sklearn_models/`) et le DeBERTa (`results/deberta_v3_parent.zip`)
4. Crée/replace l'app Streamlit `SPARK_TRIAGE_APP`

Les deux notebooks (`deberta_fine_tuning_v3`, `sklearn_resolution_training`) sont déployés via
`deploy_notebooks.py`. Aucun conteneur Docker n'est utilisé.
