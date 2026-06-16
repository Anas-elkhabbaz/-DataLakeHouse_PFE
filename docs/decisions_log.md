# Journal des décisions de conception — PFE Spark Triage

Ce document liste les décisions architecturales prises durant le projet, leur justification,
et les adaptations apportées lors de l'implémentation réelle par rapport à la spécification initiale.

---

## D-01 : Cibles de classification

**Décision :** Deux cibles de classification et une analyse textuelle.

| Cible | Nature | Modèle | Justification |
|-------|--------|--------|---------------|
| `issuetype` | Classification (**4 classes** modèle ML, 9 dans dbt) | DeBERTa-v3-base fine-tuné | Discrimine le type de travail et oriente le routage |
| `resolution` | Classification (7 classes) | LogisticRegression hybride (all-mpnet-base-v2 + features) | Prédit l'issue probable, oriente la priorité |
| Analyse textuelle | Génération (2-3 phrases) | API Google Gemini / repli moteur de règles | Explication actionnable pour l'ingénieur |

**Raison :** Correspond à la question métier réelle d'un responsable qualité :
*Quel type de problème est-ce ? Comment va-t-il être résolu ? Que faut-il faire ?*

> Les 9 classes dbt sont conservées dans MART_ML pour la traçabilité. Le modèle ML réduit
> à 4 classes au niveau du notebook (voir D-02 mis à jour).

---

## D-02 : Vocabulaire issuetype consolidé

**Deux niveaux de consolidation :**

### Niveau dbt (MART_ML) — 9 classes

| Classe | Valeurs brutes regroupées |
|--------|--------------------------|
| Bug | Bug |
| Improvement | Improvement |
| Sub-task | Sub-task |
| New Feature | New Feature |
| Task | Task, Technical task |
| Test | Test |
| Documentation | Documentation |
| Question | Question |
| Other | Umbrella, Wish, Story, Dependency upgrade, Epic, et toute valeur inconnue |

### Niveau ML (notebook fine-tuning v3) — 4 classes

| Classe ML | Classes dbt regroupées |
|-----------|------------------------|
| Bug | Bug |
| Improvement | Improvement |
| Sub-task | Sub-task |
| Other | New Feature, Task, Documentation, Test, Question, Other |

**Raison :** Les 5 classes intermédiaires (New Feature, Task, Documentation, Test, Question)
représentent chacune <5% du dataset. Le fine-tuning DeBERTa sur 4 classes bien séparées permet
d'obtenir un macro-F1 de 73,63% vs 33,77% sur 9 classes avec KNN, tout en conservant les
classes métier utiles pour le routage. La fusion est faite à l'entrée du notebook, pas dans dbt.

---

## D-03 : Vocabulaire resolution consolidé

**7 classes finales :**

| Classe | Valeurs brutes regroupées |
|--------|--------------------------|
| Fixed | Fixed, Done, Resolved, Implemented |
| Won't Fix | Won't Fix, Won't Do, Later, Abandoned |
| Not A Problem | Not A Problem, Not A Bug, Works for Me |
| Incomplete | Incomplete |
| Duplicate | Duplicate |
| Invalid | Invalid |
| Cannot Reproduce | Cannot Reproduce |

**Valeurs supprimées :** Auto Closed, Workaround, Information Provided, NULL (issues ouvertes).

**Raison :** Les issues ouvertes (résolution NULL) ne peuvent pas servir d'exemples
d'entraînement car leur résolution finale est inconnue. Après filtrage, il reste 42 083 tickets.

---

## D-04 : Split temporel

| Partition | Filtre | Lignes réelles | Usage |
|-----------|--------|----------------|-------|
| `train` | `created_at < '2023-01-01'` | 38 274 | Index de récupération KNN |
| `validation` | `2023-01-01 ≤ created_at < '2024-01-01'` | 3 809 | Évaluation hors-échantillon |
| `excluded` | `created_at ≥ '2024-01-01'` | ~3 700 | Année partielle, exclue |

**Raison :** Le split temporel est plus réaliste qu'un split aléatoire — en production, on
prédit sur des tickets futurs, jamais sur du passé mélangé avec l'entraînement.

---

## D-05 : Périmètre projet = SPARK uniquement

**Décision :** Seules les issues avec `project_key = 'SPARK'` sont traitées.

**Raison :** La spécialisation garantit la cohérence du vocabulaire technique. Le dataset
complet contient 1 149 321 issues toutes projets confondus ; SPARK en représente 49 832
(4,3 %), un volume suffisant pour entraîner et évaluer le pipeline.

---

## D-06 : Déduplication à la source

**Décision :** `QUALIFY ROW_NUMBER() OVER (PARTITION BY key ORDER BY id) = 1` dans `stg_issues`.

**Raison :** La source RAW.ISSUES contient 4 clés JIRA en double (bug de l'export Kaggle).
Sans déduplication, les tests `unique` dbt échouaient en cascade jusqu'à `mart_ml`.
La déduplication par `id` le plus petit garantit l'idempotence et le choix de la ligne originale.

---

## D-07 : Pipeline d'inférence — Évolution vers DeBERTa fine-tuning

**Décision initiale (spécification) :** Pipeline entièrement dans Snowflake Cortex.
- Embeddings : `SNOWFLAKE.CORTEX.EMBED_TEXT_1024('voyage-multilingual-2', ...)` (1024d)
- Résumé RCA : `SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', ...)`
- Arbitrage LLM : `SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', ...)` pour les cas incertains
- Génération fix_summary : `SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', ...)`

**Raison de l'adaptation initiale (v1–v2) :** Les fonctions `CORTEX.COMPLETE` et
`CORTEX.EMBED_TEXT_1024` sont bloquées sur les comptes Snowflake trial (erreur 399258).
Le fallback Python KNN (`all-MiniLM-L6-v2`, k=15) a été implémenté en premier.

**Décision finale implémentée (v3) — deux modèles distincts :**

| Cible | Modèle | Justification |
|-------|--------|---------------|
| issuetype | `microsoft/deberta-v3-base` fine-tuné (5 epochs, Kaggle Tesla T4) | DeBERTa > KNN : +40pt macro-F1 grâce à la compréhension contextuelle et au signal has_parent |
| résolution | `LogisticRegression` + `all-mpnet-base-v2` (768d) + 17 features tabulaires (hors `resolution_days`/`n_resolution_changes`, fuite de cible) | 81,3% accuracy (macro-F1 26,9%), au-dessus du seuil de 75% |
| Analyse textuelle | API **Google Gemini** (`gemini-2.0-flash`, optionnel) avec repli sur moteur de règles | Génère l'explication actionnable « Que faire ? » pour l'ingénieur |

**Chemin de fine-tuning :**
1. Export `MART_ML` → `mart_ml_export.csv` (Kaggle dataset)
2. Jointure avec `spark_parent_keys.csv` (API JIRA) → feature `has_parent`
3. Remapping 9 → 4 classes issuetype
4. Fine-tuning DeBERTa sur Kaggle (Tesla T4, ~2h) → `deberta_v3_parent/` (599 MB)
5. Résultats : 79,6% accuracy, 73,63% macro-F1

Le pipeline Snowflake Cortex (`CORTEX.COMPLETE` / `EMBED_TEXT_1024`) reste l'**implémentation de
référence** envisageable sur un compte payant, mais n'est pas inclus dans le dépôt (compte d'essai).

---

## D-08 : Seuils du gate de confiance

Implémenté dans `load/run_ml_pipeline.py` et l'application `apps/inference/streamlit_in_snowflake.py` :

| Niveau | Seuil confiance | Couleur UI | Comportement |
|--------|-----------------|------------|--------------|
| High | ≥ 65 % | Vert | Prédiction directe, affichée sans avertissement |
| Medium | 45–64 % | Ambre | Prédiction affichée, consultation des tickets similaires conseillée |
| Low | < 45 % | Rouge | Avertissement explicite — révision manuelle requise |

**Raison :** Ces seuils ont été calibrés empiriquement sur la distribution des confiances
observées sur les 3 809 tickets de validation.

---

## D-09 : Gestion des tables brutes en VARCHAR

**Décision :** Toutes les colonnes de RAW.ISSUES, RAW.COMMENTS, RAW.CHANGELOG et
RAW.ISSUELINKS sont déclarées en VARCHAR.

**Raison :** Les CSV Kaggle contiennent des valeurs mal formées (timestamps invalides,
nombres en notation scientifique). Typer les colonnes en COPY INTO provoquerait des erreurs
et des lignes ignorées silencieusement. Le cast est effectué en staging via `TRY_TO_*` qui
retourne NULL plutôt qu'une erreur.

---

## D-10 : Cache local des embeddings

**Décision :** Les embeddings d'entraînement sont calculés une seule fois et sauvegardés dans
`results/embeddings_cache.npz` (57 MB). Le fichier est versionné dans git.

**Raison :** Le calcul des embeddings pour 38 274 tickets prend ~10 min sur CPU. Versionner
le cache permet à tout collaborateur (ou à l'application déployée) de démarrer l'inférence
instantanément sans recalcul. La taille (57 MB) est sous la limite hard de GitHub (100 MB).

---

## D-11 : Déploiement — Streamlit-in-Snowflake (pas de Docker)

**Décision :** L'application d'inférence est déployée **nativement dans Snowflake**
(Streamlit-in-Snowflake) via `deploy_streamlit_snowflake.py`, et non plus dans des conteneurs Docker.
L'analytique est servie par **Power BI** connecté nativement à `MARTS_ANALYTICS.*`.

**Raison :** Exécuter l'app au sein de l'entrepôt place le calcul au plus près des données et des
modèles (sklearn + DeBERTa déployés dans le stage `ML_MODELS.APP_STAGE`), supprime la dépendance à
une infrastructure de conteneurs externe, et garantit un déploiement reproductible dans Snowflake.
Une ancienne variante Docker (deux apps Streamlit) a été retirée au profit de cette approche.

---

## D-12 : Feature `has_parent` — signal externe JIRA

**Décision :** Enrichir le dataset d'entraînement avec une feature binaire `has_parent`
récupérée via l'API JIRA Apache, stockée dans `spark_parent_keys.csv`.

**Implémentation :**
- Jointure LEFT sur `key` entre `mart_ml_export.csv` et `spark_parent_keys.csv`
- Valeurs manquantes → 0 (clip 0–1)
- Signal injecté dans le texte : `[HAS-PARENT]` ou `[NO-PARENT]` en préfixe

**Corrélation observée :**

| issuetype | has_parent moyen |
|-----------|-----------------|
| Bug | 0,0 % |
| Improvement | 0,0 % |
| Other | 0,1 % |
| Sub-task | **100,0 %** |

**Raison :** La définition JIRA d'un Sub-task implique structurellement l'existence d'un
ticket parent. Ce signal transforme un problème de classification textuelle difficile
(Sub-task vs Task lexicalement similaires) en une règle quasi-déterministe, portant le
F1 Sub-task à 1,00 et le macro-F1 global de 33,77% à 73,63%.

**Non intégré dans dbt** pour éviter une dépendance externe à l'API JIRA dans le pipeline
de transformation SQL. La jointure se fait uniquement au moment de l'entraînement.
