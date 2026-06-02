-- Gold : charge de travail par assignataire (Path 2 — tableau de bord)
-- Source : int_issues_analytics (inclut les tickets ouverts pour le backlog individuel)

SELECT
    assignee,
    COUNT(*)                                                              AS n_assigned,
    SUM(is_resolved)                                                      AS n_fixed,
    COUNT(*) - SUM(is_resolved)                                           AS n_open_backlog,
    ROUND(AVG(CASE WHEN is_resolved = 1 THEN resolution_days END), 1)     AS avg_resolution_days,
    ROUND(MEDIAN(CASE WHEN is_resolved = 1 THEN resolution_days END), 1)  AS median_resolution_days,
    ROUND(100.0 * SUM(is_resolved) / NULLIF(COUNT(*), 0), 2)              AS resolution_rate_pct,
    COUNT(CASE WHEN issuetype = 'Bug' THEN 1 END)                         AS n_bugs,
    COUNT(CASE WHEN issuetype = 'Improvement' THEN 1 END)                 AS n_improvements,
    COUNT(CASE WHEN issuetype = 'New Feature' THEN 1 END)                 AS n_new_features,
    MODE(issuetype)                                                        AS top_issuetype,
    COUNT(DISTINCT issuetype)                                              AS n_distinct_issuetypes,
    MIN(created_at)                                                        AS first_assignment,
    MAX(created_at)                                                        AS last_assignment
FROM {{ ref('int_issues_analytics') }}
WHERE assignee IS NOT NULL
  AND assignee != 'Unassigned'
GROUP BY assignee
