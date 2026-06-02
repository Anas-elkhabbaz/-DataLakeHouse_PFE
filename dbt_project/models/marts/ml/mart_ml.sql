-- Gold : table de données pour le pipeline ML
-- Deux représentations textuelles séparées pour éviter la fuite de label :
--   text_for_it  : sans issuetype (prédiction de issuetype)
--   text_for_res : avec issuetype (prédiction de resolution, type connu à l'arrivée)
-- STATUS retiré des deux : toujours "Resolved" en train → leakage temporel

SELECT
    i.key,
    i.created_at,
    i.split,

    -- Cibles de classification
    i.issuetype,
    i.resolution,

    -- Texte nettoyé
    i.summary       AS summary_clean,
    i.description   AS description_clean,
    COALESCE(c.all_comments, '') AS comments_concat,

    -- text_for_it : embedding pour prédire issuetype
    -- Prefixes [FLAG] encodes les signaux structurels comme texte :
    --   [NO-DESCRIPTION] → description absente (31.9% des Sub-tasks vs 6% des Bugs)
    --   [SHORT-DESC]     → description < 80 chars
    --   [NO-COMMENTS]    → aucun commentaire
    --   [CONTAINER-LINK] → lien CONTAINER = signal parent/enfant
    -- Ces flags permettent au modèle de les associer à des classes sans re-coder
    -- des features séparées, et bénéficient directement au fine-tuning Colab.
    LEFT(
        CONCAT(
            -- Flags structurels encodés en texte
            CASE WHEN LENGTH(COALESCE(i.description, '')) = 0        THEN '[NO-DESCRIPTION] ' ELSE '' END,
            CASE WHEN LENGTH(COALESCE(i.description, '')) BETWEEN 1 AND 79 THEN '[SHORT-DESC] '     ELSE '' END,
            CASE WHEN COALESCE(c.n_comments, 0) = 0                  THEN '[NO-COMMENTS] '    ELSE '' END,
            CASE WHEN COALESCE(il.n_container, 0) > 0                THEN '[CONTAINER-LINK] ' ELSE '' END,
            -- Contenu principal
            'TICKET: ', COALESCE(i.summary, ''), '\n',
            'PRI: ', COALESCE(i.priority, ''), '\n',
            'DESC: ', CASE
                WHEN LENGTH(COALESCE(i.description, '')) < 50
                THEN COALESCE(i.description, '') || ' ' || LEFT(COALESCE(c.all_comments, ''), 600)
                ELSE COALESCE(i.description, '')
            END
        ),
        2000
    ) AS text_for_it,

    -- text_for_res : embedding pour prédire resolution
    -- Inclut issuetype (connu à l'arrivée du ticket) + 400 chars de commentaires
    LEFT(
        CONCAT(
            'TICKET: ', COALESCE(i.summary, ''), '\n',
            'TYPE: ', COALESCE(i.issuetype, ''), ' | PRI: ', COALESCE(i.priority, ''), '\n',
            'DESC: ', LEFT(COALESCE(i.description, ''), 800), '\n',
            'COMMENTS: ', LEFT(COALESCE(c.all_comments, ''), 400)
        ),
        2000
    ) AS text_for_res,

    -- Conservé pour compatibilité avec inference_app.py (ne pas supprimer)
    LEFT(
        CONCAT(
            'TICKET: ', COALESCE(i.summary, ''), '\n',
            'TYPE: ', COALESCE(i.issuetype, ''), ' | PRI: ', COALESCE(i.priority, ''), '\n',
            'DESC: ', LEFT(COALESCE(i.description, ''), 800)
        ),
        2000
    ) AS text_noco,

    -- Métadonnées pour le boost de récupération (§9.4 Step 3)
    i.priority,
    i.status,
    i.reporter,
    i.assignee,

    -- Features changelog (contribution originale du PFE)
    COALESCE(cl.n_total_changes,      0) AS n_total_changes,
    COALESCE(cl.n_status_changes,     0) AS n_status_changes,
    COALESCE(cl.n_priority_changes,   0) AS n_priority_changes,
    COALESCE(cl.n_assignee_changes,   0) AS n_assignee_changes,
    COALESCE(cl.n_resolution_changes, 0) AS n_resolution_changes,
    COALESCE(cl.was_escalated,        0) AS was_escalated,
    COALESCE(cl.n_people_involved,    0) AS n_people_involved,
    cl.first_assignee,

    -- Features de liens
    COALESCE(il.n_links_total,  0) AS n_links_total,
    COALESCE(il.n_duplicates,   0) AS n_duplicates,
    COALESCE(il.n_blocks,       0) AS n_blocks,
    COALESCE(il.n_blocked_by,   0) AS n_blocked_by,
    COALESCE(il.n_relates,      0) AS n_relates,
    COALESCE(il.n_container,    0) AS n_container,

    -- Features commentaires
    COALESCE(c.n_comments,   0) AS n_comments,
    COALESCE(c.n_commenters, 0) AS n_commenters,

    -- Métriques de timing et de longueur
    i.resolution_days,
    i.summary_length,
    i.description_length,

    -- has_parent : signal déterminant pour Sub-task (récupéré via API JIRA Apache)
    -- 1 = ticket enfant d'un parent = Sub-task avec quasi-certitude
    COALESCE(pk.has_parent, 0) AS has_parent,
    pk.parent_key

FROM {{ ref('int_issues_cleaned') }} i
LEFT JOIN {{ ref('int_comments_aggregated') }}  c  USING (key)
LEFT JOIN {{ ref('int_changelog_features') }}   cl USING (key)
LEFT JOIN {{ ref('int_issuelinks_features') }}  il USING (key)
LEFT JOIN PFE_SPARK.RAW.SPARK_PARENT_KEYS       pk USING (key)

-- On exclut les tickets 2024+ du mart ML (tagués 'excluded' dans int_issues_cleaned)
WHERE i.split IN ('train', 'validation')
