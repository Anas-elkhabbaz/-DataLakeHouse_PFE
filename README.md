# Spark Issue Triage Platform

**Projet de Fin d'Études (PFE)** — Filière Big Data & IA, UIR Rabat  
**Étudiant :** Anas Elkhabbaz | **Encadrant entreprise :** SQLI Rabat  
**Soutenance :** 24 juin 2026  
**Dépôt :** https://github.com/Anas-elkhabbaz/-DataLakeHouse_PFE

---

## Description

Plateforme complète de triage automatique des incidents du projet Apache Spark, construite sur
une architecture en médaillon (Bronze → Argent → Or) hébergée sur Snowflake et orchestrée par dbt.
Le dataset source est le dump public JIRA Apache Spark de Kaggle (mars 2025, ~49 832 tickets SPARK).

Deux chemins de consommation sont exposés :

- **Chemin 1 — Inférence IA** : pipeline basé sur un modèle `microsoft/deberta-v3-base`
  fine-tuné sur 38 274 tickets Spark (5 epochs, Kaggle Tesla T4). Classifie le type d'incident
  en 4 classes (Bug, Improvement, Sub-task, Other) à partir du texte enrichi avec le signal
  `has_parent` (ticket lié à un parent JIRA). Génère une analyse textuelle (via Anthropic
  claude-haiku si une clé API est configurée, sinon un template structuré).
- **Chemin 2 — Tableau de bord analytique** : dashboard Streamlit 5 pages explorant les
  volumes mensuels, la dynamique de résolution, la charge des assignataires et les liens entre tickets.

---

## Résultats obtenus

| Métrique | Modèle | Valeur | Seuil cible |
|----------|--------|--------|-------------|
| Accuracy issuetype (4 classes) | DeBERTa-v3-base fine-tuné | **79,6 %** | > 70 % |
| Macro-F1 issuetype (4 classes) | DeBERTa-v3-base fine-tuné | **73,63 %** | — |
| Accuracy résolution | LogisticRegression (all-mpnet-base-v2 + features tabulaires) | 91,52 % | > 75 % |

Résultats par classe issuetype (jeu de validation, 3 809 tickets) :

| Classe | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| Bug | 0,72 | 0,69 | 0,71 | 671 |
| Improvement | 0,66 | 0,72 | 0,69 | 1 012 |
| Other | 0,57 | 0,49 | 0,53 | 572 |
| Sub-task | 0,99 | 1,00 | **1,00** | 1 554 |

> Le Sub-task atteint 100% de recall grâce au signal `has_parent` — 100% des Sub-tasks ont un
> ticket parent JIRA, contre 0% pour Bug/Improvement/Other. Ce signal est récupéré via l'API
> JIRA (fichier `spark_parent_keys.csv`) et préfixé dans le texte sous la forme `[HAS-PARENT]`.

**Jeu de données :**

| Table | Lignes |
|-------|--------|
| RAW.ISSUES | 1 149 321 |
| RAW.COMMENTS | 5 047 714 |
| RAW.CHANGELOG | 9 653 526 |
| RAW.ISSUELINKS | 390 063 |
| Tickets SPARK filtrés | 49 832 |
| MARTS_ML.MART_ML (train + val) | 42 083 |
| Tickets d'entraînement | 38 274 |
| Tickets de validation | 3 809 |

**Tests dbt :** PASS=44 WARN=2 ERROR=0 (les 2 warnings concernent les valeurs "Won't Fix"
dont l'apostrophe génère un warning SQL dans les tests `accepted_values`).

---

## Prérequis

- Python 3.12 (le dbt venv) + Python 3.11+ (scripts de chargement)
- Un compte Snowflake (trial ou payant — voir note ci-dessous)
- dbt-snowflake 1.11+ installé via `uv` dans `dbt_project/.venv/`
- Les 4 fichiers CSV dans `data/` (issues, comments, changelog, issuelinks)
- Docker Desktop (optionnel, pour lancer les apps via docker-compose)

> **Note Snowflake Cortex :** Les fonctions `SNOWFLAKE.CORTEX.COMPLETE` et
> `SNOWFLAKE.CORTEX.EMBED_TEXT_1024` sont bloquées sur les comptes trial. Le pipeline
> d'inférence utilise `sentence-transformers` (gratuit, ~80 MB) et DeBERTa v3 fine-tuné.

---

## Installation rapide (Docker)

La façon la plus simple de lancer les applications :

```bash
git clone https://github.com/Anas-elkhabbaz/-DataLakeHouse_PFE.git
cd -DataLakeHouse_PFE

# Configurer les identifiants Snowflake
cp .env.example .env
# Éditer .env avec vos valeurs SNOWFLAKE_*

# Lancer les deux applications
docker-compose up --build
```

- Application d'inférence : http://localhost:8501
- Tableau de bord analytique : http://localhost:8502

La première construction télécharge le modèle `microsoft/deberta-v3-base` (~370 MB) et
charge le modèle fine-tuné `deberta_v3_parent/` (~599 MB). Les lancements suivants démarrent
en quelques secondes grâce au cache Docker.

---

## Installation manuelle

```bash
git clone https://github.com/Anas-elkhabbaz/-DataLakeHouse_PFE.git
cd -DataLakeHouse_PFE

# Installer les dépendances de l'app d'inférence
pip install -r apps/inference/requirements.txt

# Installer les dépendances du tableau de bord
pip install -r apps/analytics/requirements.txt

# Configurer les identifiants
cp .env.example .env
# Éditer .env

# Lancer les apps
streamlit run apps/inference/inference_app.py   # port 8501
streamlit run apps/analytics/analytics_app.py  # port 8502
```

---

## Configuration Snowflake (chargement initial des données)

### 1. Écrire le profil dbt

```bash
python load/write_profiles.py
```

Ce script lit `.env` et écrit `~/.dbt/profiles.yml` en gérant correctement l'encodage Unicode.

### 2. Créer la base de données, les schémas et le warehouse

```bash
python load/run_phase1.py
```

Crée : DATABASE `PFE_SPARK`, 6 schémas (RAW, STAGING, INTERMEDIATE, MARTS_ML,
MARTS_ANALYTICS, CORTEX), warehouse `PFE_WH`, stage interne `RAW.CSV_STAGE`.

### 3. Inspecter les en-têtes CSV

```bash
python load/inspect_headers.py
```

Affiche les positions exactes des colonnes dans chaque CSV. Vérifier que le mapping
dans `load/04_copy_into_raw.sql` correspond avant de passer à l'étape suivante.

### 4. Charger les CSV vers le stage puis dans les tables brutes

```bash
python load/03_put_files.py    # PUT vers @RAW.CSV_STAGE (~30 min selon la connexion)
python load/run_phase4.py      # COPY INTO + vérification des comptes
```

### 5. Pipeline dbt

```bash
cd dbt_project
.venv\Scripts\dbt deps         # Installer dbt-utils
.venv\Scripts\dbt seed         # Tables de mapping labels
.venv\Scripts\dbt run          # 11 modèles
.venv\Scripts\dbt test         # 46 tests (attendu : PASS=44, WARN=2)
```

### 6. Pipeline ML (embeddings + prédictions + évaluation)

```bash
python load/run_ml_pipeline.py
```

Lance le pipeline d'inférence (DeBERTa v3 fine-tuné pour issuetype + LogisticRegression pour résolution),
fait les prédictions sur les 3 809 tickets de validation, évalue les performances et sauvegarde
les résultats dans `results/` et dans `CORTEX.MART_PREDICTIONS` sur Snowflake.

> **Prérequis :** le répertoire `deberta_v3_parent/` doit être présent localement. Il est
> généré par le notebook `notebook6d3e788d00.ipynb` (entraîné sur Kaggle avec GPU Tesla T4)
> ou téléchargeable depuis le dépôt (599 MB zippé).

---

## Configuration optionnelle — Analyse LLM (Anthropic)

Pour activer l'analyse textuelle générée par IA dans l'application d'inférence,
ajouter la clé suivante dans `.env` :

```
ANTHROPIC_API_KEY=sk-ant-...
```

Obtenir une clé sur [console.anthropic.com](https://console.anthropic.com).
Le modèle utilisé est `claude-haiku-4-5` (~$0,25 / 1M tokens — coût négligeable par prédiction).
Sans cette clé, l'application génère un rapport structuré automatique.

---

## Structure du projet

```
DataLakeHouse_PFE/
├── .env.example                 # Template de configuration
├── .gitignore
├── .dockerignore
├── docker-compose.yml           # Lance inference + analytics
├── pyproject.toml
├── README.md
│
├── data/                        # CSVs source (non versionnés, ~8 GB total)
│   ├── issues.csv
│   ├── comments.csv
│   ├── changelog.csv
│   └── issuelinks.csv
│
├── load/                        # Scripts de chargement Bronze
│   ├── run_phase1.py            # Crée la base Snowflake
│   ├── inspect_headers.py       # Vérifie les positions CSV
│   ├── 03_put_files.py          # PUT vers le stage
│   ├── run_phase4.py            # COPY INTO + vérification
│   ├── run_ml_pipeline.py       # Embeddings + prédictions + évaluation
│   ├── write_profiles.py        # Génère ~/.dbt/profiles.yml
│   └── 04_copy_into_raw.sql     # DDL tables brutes + mapping $N
│
├── dbt_project/                 # Transformations Silver + Or
│   ├── models/
│   │   ├── staging/             # 4 vues (1:1 avec les sources)
│   │   ├── intermediate/        # 4 tables (NLP, features, split)
│   │   └── marts/
│   │       ├── ml/              # MART_ML (42 083 lignes)
│   │       └── analytics/       # MART_ANALYTICS_OPS + DEPS
│   ├── seeds/                   # issuetype_mapping.csv, resolution_mapping.csv
│   ├── macros/                  # clean_jira_text, generate_schema_name
│   └── .venv/                   # Python 3.12 venv pour dbt-snowflake
│
├── apps/
│   ├── inference/               # Application de triage (Streamlit)
│   │   ├── Dockerfile
│   │   ├── inference_app.py     # UI professionnelle + inférence temps réel
│   │   └── requirements.txt
│   └── analytics/               # Tableau de bord 5 pages (Streamlit)
│       ├── Dockerfile
│       ├── analytics_app.py
│       ├── pages/
│       │   ├── 1_overview.py
│       │   ├── 2_resolution_dynamics.py
│       │   ├── 3_workload.py
│       │   └── 4_relationships.py
│       └── requirements.txt
│
├── results/                     # Artefacts de l'évaluation
│   ├── embeddings_cache.npz     # Embeddings pré-calculés (57 MB, versionné)
│   └── ...                      # Prédictions et métriques (non versionnées)
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
| [docs/architecture.md](docs/architecture.md) | Schéma medallion et description des couches |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Description de chaque colonne de mart_ml |
| [docs/decisions_log.md](docs/decisions_log.md) | Choix architecturaux figés et leur justification |

---

## Tableau de bord Power BI

Le dashboard analytique (visualisation des données `MARTS_ANALYTICS`) est disponible sur SharePoint :

**[Ouvrir PB-PFE.pbix](https://sqli468-my.sharepoint.com/:u:/r/personal/sdriham_sqli_com/Documents/PB-PFE.pbix?csf=1&web=1&e=n6MAsp)**

> Développé par l'équipe analytics. Connecté aux tables `MARTS_ANALYTICS.*` dans Snowflake (`PFE_SPARK`).
> Requiert Power BI Desktop pour ouvrir le fichier `.pbix`.
