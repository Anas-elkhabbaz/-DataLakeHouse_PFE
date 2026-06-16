# Spark Issue Triage Platform

**Projet de Fin d'Études (PFE)** — Filière Big Data & IA, UIR Rabat
**Étudiant :** Anas Elkhabbaz | **Encadrant entreprise :** SQLI Rabat
**Dépôt :** https://github.com/Anas-elkhabbaz/-DataLakeHouse_PFE

---

## Description

Plateforme complète de triage automatique des incidents du projet Apache Spark, construite sur
une architecture en médaillon (Bronze → Argent → Or) hébergée sur Snowflake et orchestrée par dbt.
Le dataset source est le dump public JIRA Apache Spark de Kaggle (mars 2025, ~49 832 tickets SPARK).

> **Contexte / confidentialité :** la solution a été conçue pour un client de SQLI pendant le stage.
> Les données client étant confidentielles, **tous les résultats de ce dépôt sont une reproduction sur
> données publiques Apache Spark**, exécutée dans un **compte Snowflake d'essai personnel** — ce qui
> explique l'indisponibilité de Snowflake Cortex (cf. ci-dessous).

Deux chemins de consommation sont exposés :

- **Chemin 1 — Inférence IA** : application **Streamlit-in-Snowflake** qui, pour un ticket donné,
  prédit le **type d'incident** (modèle `microsoft/deberta-v3-base` fine-tuné, 4 classes : Bug,
  Improvement, Sub-task, Other — texte enrichi du signal `has_parent`) et la **résolution probable**
  (régression logistique hybride : embeddings `all-mpnet-base-v2` + features tabulaires). Elle produit
  une explication corrective « Que faire ? » générée par l'**API Google Gemini** (`gemini-2.0-flash`),
  avec repli sur un moteur de règles contextuel si l'API est indisponible.
- **Chemin 2 — Tableau de bord analytique Power BI** : dashboard connecté nativement aux tables
  `MARTS_ANALYTICS.*` de Snowflake (volumes mensuels, dynamique de résolution, charge des
  assignataires, liens entre tickets, versions).

---

## Résultats obtenus (jeu de validation, 3 809 tickets)

| Métrique | Modèle | Valeur | Seuil cible |
|----------|--------|--------|-------------|
| Accuracy issuetype (4 classes) | DeBERTa-v3-base fine-tuné | **79,6 %** | > 70 % |
| Macro-F1 issuetype (4 classes) | DeBERTa-v3-base fine-tuné | **73,63 %** | — |
| Accuracy résolution (7 classes) | LogisticRegression (all-mpnet-base-v2 + features) | 81,3 % | > 75 % |
| Macro-F1 résolution (7 classes) | LogisticRegression (all-mpnet-base-v2 + features) | 26,9 % | — |

> **Lecture honnête de la résolution :** le seuil d'accuracy de 75 % est franchi (81,3 %), mais sur
> une cible aussi déséquilibrée (la classe *Fixed* ≈ 90 % de la validation) l'accuracy est trompeuse.
> La métrique honnête est le **macro-F1 (26,9 %)** ; ce volet est donc présenté comme un **résultat
> partiel assumé**, pas comme une réussite.

Résultats par classe issuetype :

| Classe | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Bug | 0,72 | 0,69 | 0,71 | 671 |
| Improvement | 0,66 | 0,72 | 0,69 | 1 012 |
| Other | 0,57 | 0,49 | 0,53 | 572 |
| Sub-task | 0,99 | 1,00 | **1,00** | 1 554 |

> Le Sub-task atteint F1 = 1,00 grâce au signal **`has_parent`** (100 % des Sub-tasks ont un ticket
> parent JIRA, 0 % pour Bug/Improvement). Signal récupéré via l'API JIRA (`spark_parent_keys.csv`) et
> préfixé dans le texte sous la forme `[HAS-PARENT]`. C'est un signal quasi-étiquette : l'essentiel de
> la compréhension textuelle se mesure sur Bug/Improvement/Other (F1 0,53–0,71).

> **Note sur les features de résolution :** le modèle déployé a été entraîné sur 19 features
> tabulaires ; le rapport en documente **17**, après exclusion de deux variables constituant une fuite
> de la cible (`resolution_days`, `n_resolution_changes`). Cf. `docs/decisions_log.md`.

**Jeu de données :**

| Table | Lignes |
|-------|--------|
| RAW.ISSUES | 1 149 321 |
| RAW.COMMENTS | 5 047 714 |
| RAW.CHANGELOG | 9 653 526 |
| RAW.ISSUELINKS | 390 063 |
| Tickets SPARK filtrés | 49 832 |
| MARTS_ML.MART_ML (train + val) | 42 083 |
| Tickets d'entraînement / validation | 38 274 / 3 809 |

**Pipeline dbt :** 20 modèles, 2 seeds — **66 tests de données : PASS=66, WARN=0, ERROR=0**
(exécutés nativement sur Snowflake via l'intégration Git des Workspaces).

---

## Prérequis

- Python 3.11+ (scripts de chargement et pipeline ML) ; Python 3.12 pour le venv dbt
- Un compte Snowflake (trial ou payant — voir note Cortex)
- dbt-snowflake 1.11+ (installé via `uv` dans `dbt_project/.venv/`)
- Les 4 fichiers CSV source dans `data/` (issues, comments, changelog, issuelinks)
- *(optionnel)* une clé `GEMINI_API_KEY` pour l'explication générative « Que faire ? »

> **Note Snowflake Cortex :** les fonctions `SNOWFLAKE.CORTEX.COMPLETE` et `EMBED_TEXT_1024` sont
> bloquées sur les comptes trial. Le pipeline utilise donc `sentence-transformers` (gratuit, local)
> et le DeBERTa v3 fine-tuné ; l'explication « Que faire ? » passe par l'API Gemini.

---

## Mise en route

L'installation pas-à-pas complète (infrastructure Snowflake, chargement Bronze, dbt, pipeline ML,
déploiement) est détaillée dans **[SETUP.md](SETUP.md)**. En résumé :

```bash
git clone https://github.com/Anas-elkhabbaz/-DataLakeHouse_PFE.git
cd -DataLakeHouse_PFE
cp .env.example .env          # renseigner les identifiants SNOWFLAKE_* (+ GEMINI_API_KEY optionnel)

python load/write_profiles.py # profil dbt
python load/run_phase1.py     # DB PFE_SPARK + 7 schémas + warehouse PFE_WH
python load/03_put_files.py   # PUT des CSV vers @RAW.CSV_STAGE
python load/run_phase4.py     # COPY INTO tables brutes

cd dbt_project
dbt deps && dbt seed && dbt run && dbt test   # 20 modèles, 66 tests

python fetch_parent_keys.py        # has_parent via l'API JIRA -> spark_parent_keys.csv
python migrate_to_snowflake.py     # upload + reconstruction de mart_ml
python load/run_ml_pipeline.py     # embeddings + prédictions + évaluation -> PREDICTIONS.MART_PREDICTIONS
```

### Déploiement (Streamlit-in-Snowflake)

L'application d'inférence s'exécute **nativement dans Snowflake** (Streamlit-in-Snowflake) :

```bash
python load/train_save_resolution_model.py   # génère results/sklearn_models/
python deploy_streamlit_snowflake.py          # déploie l'app + les modèles dans Snowflake
python deploy_notebooks.py                    # déploie les 2 notebooks dans Snowflake
```

Accès dans Snowsight : **Streamlit → SPARK_TRIAGE_APP** et **Projects → Notebooks**.

---

## Configuration optionnelle — Explication générative (Gemini)

Pour activer l'explication « Que faire ? » générée par IA dans l'application d'inférence, ajouter dans `.env` :

```
GEMINI_API_KEY=...
```

Obtenir une clé sur [Google AI Studio](https://aistudio.google.com/app/apikey). Sans cette clé,
l'application bascule automatiquement sur un **moteur de règles contextuel** qui produit une
explication déterministe équivalente.

---

## Structure du projet

```
DataLakeHouse_PFE/
├── .env.example  .gitignore  .gitattributes
├── pyproject.toml  uv.lock
├── README.md  SETUP.md
├── fetch_parent_keys.py          # has_parent via l'API JIRA Apache
├── migrate_to_snowflake.py       # upload parent keys + reconstruction mart_ml
├── deploy_streamlit_snowflake.py # déploie l'app Streamlit-in-Snowflake + modèles
├── deploy_notebooks.py           # déploie les 2 notebooks dans Snowflake
│
├── data/                         # CSVs source (non versionnés, ~8 GB)
│
├── load/                         # Chargement Bronze + pipeline ML
│   ├── run_phase1.py             # crée la base + schémas + warehouse
│   ├── write_profiles.py         # génère ~/.dbt/profiles.yml
│   ├── inspect_headers.py        # vérifie les positions CSV
│   ├── 03_put_files.py           # PUT vers le stage
│   ├── run_phase4.py             # COPY INTO + vérification des comptes
│   ├── run_ml_pipeline.py        # embeddings + prédictions + évaluation
│   ├── train_save_resolution_model.py  # entraîne/sauve le modèle résolution (.pkl)
│   ├── upload_predictions.py
│   └── 01_..05_*.sql             # DDL bronze + mapping $N
│
├── dbt_project/                  # Transformations Argent + Or
│   ├── models/
│   │   ├── staging/              # 4 vues (1:1 avec les sources)
│   │   ├── intermediate/         # tables NLP, features, split, analytics
│   │   └── marts/
│   │       ├── ml/               # MART_ML (42 083 lignes)
│   │       └── analytics/        # 7 tables MART_ANALYTICS_* (source Power BI)
│   ├── seeds/                    # issuetype_mapping.csv, resolution_mapping.csv
│   └── macros/                   # clean_jira_text (consolidate_label), generate_schema_name
│
├── apps/
│   └── inference/                # Application de triage (Streamlit-in-Snowflake)
│       ├── streamlit_in_snowflake.py   # app canonique (Gemini + repli règles)
│       ├── similar_reference_utils.py  # références historiques similaires
│       ├── environment.yml             # env conda pour Streamlit-in-Snowflake
│       └── requirements.txt            # deps pour usage local des scripts
│
├── results/                      # Artefacts du pipeline
│   ├── embeddings_cache.npz      # embeddings pré-calculés (versionné, Git LFS)
│   ├── spark_parent_keys.csv     # signal has_parent (API JIRA)
│   ├── deberta_v3_parent.zip     # modèle DeBERTa fine-tuné (599 MB)
│   ├── sklearn_models/           # clf_resolution.pkl, scaler.pkl, meta.json
│   ├── notebooks/                # deberta_fine_tuning_v3.ipynb, sklearn_resolution_training.ipynb
│   └── generate_*.py             # scripts de génération des figures du rapport
│
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    └── decisions_log.md
```

---

## Documentation

| Document | Contenu |
|----------|---------|
| [SETUP.md](SETUP.md) | Guide d'installation pas-à-pas (Snowflake → dbt → ML → déploiement) |
| [docs/architecture.md](docs/architecture.md) | Schéma médaillon et description des couches |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Description de chaque table/colonne (MART_ML, PREDICTIONS, analytics) |
| [docs/decisions_log.md](docs/decisions_log.md) | Choix architecturaux figés et leur justification |

---

## Tableau de bord Power BI

Le dashboard analytique (tables `MARTS_ANALYTICS.*`) est développé sous Power BI Desktop et connecté
nativement à Snowflake. Fichier `.pbix` disponible sur SharePoint :

**[Ouvrir PB-PFE.pbix](https://sqli468-my.sharepoint.com/:u:/r/personal/sdriham_sqli_com/Documents/PB-PFE.pbix?csf=1&web=1&e=n6MAsp)**

> Connecté aux tables `MARTS_ANALYTICS.*` dans Snowflake (`PFE_SPARK`). Requiert Power BI Desktop.
