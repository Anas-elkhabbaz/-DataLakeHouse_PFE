-- Intermédiaire analytics : toutes les issues SPARK y compris ouvertes/non résolues
-- Différence avec int_issues_cleaned : pas de filtre WHERE resolution IS NOT NULL
-- Utilisé par les marts analytics (Power BI) — NE PAS utiliser pour le ML

WITH base AS (
    SELECT * FROM {{ ref('stg_issues') }}
),

with_labels AS (
    SELECT
        b.key,
        b.issue_id,
        b.created_at,
        b.resolved_at,
        COALESCE(NULLIF(TRIM(b.priority), ''), 'Unknown') AS priority,
        b.status,
        b.project,
        b.assignee,
        b.reporter,
        b.votes,
        b.watches,
        COALESCE(im.issuetype, 'Other') AS issuetype,
        rm.resolution                   AS resolution,
        LEAST(DATEDIFF('day', b.created_at, COALESCE(b.resolved_at, CURRENT_TIMESTAMP())), 5000) AS age_days,
        CASE
            WHEN b.resolved_at IS NOT NULL
            THEN LEAST(DATEDIFF('day', b.created_at, b.resolved_at), 5000)
        END AS resolution_days,
        CASE
            WHEN b.resolved_at IS NOT NULL THEN 1 ELSE 0
        END AS is_resolved,
        DATE_TRUNC('month', b.created_at)  AS created_month,
        DATE_TRUNC('year',  b.created_at)  AS created_year
    FROM base b
    LEFT JOIN {{ ref('issuetype_mapping') }} im ON b.issuetype_raw = im.issuetype_raw
    LEFT JOIN {{ ref('resolution_mapping') }} rm ON b.resolution_raw  = rm.resolution_raw
)

SELECT *
FROM with_labels
