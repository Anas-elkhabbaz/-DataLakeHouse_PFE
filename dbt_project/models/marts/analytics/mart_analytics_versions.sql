-- Gold : métriques par version Spark cible (1.x / 2.x / 3.x / 4.x)
-- Granularité : version_major × version_minor × issuetype
-- Source : int_fix_versions (une issue peut apparaître plusieurs fois si multi-version)

WITH by_version AS (
    SELECT
        version_major,
        version_minor,
        issuetype,
        COUNT(DISTINCT key)                                                    AS n_issues,
        COUNT(DISTINCT CASE WHEN issuetype = 'Bug'         THEN key END)       AS n_bugs,
        COUNT(DISTINCT CASE WHEN issuetype = 'Improvement' THEN key END)       AS n_improvements,
        COUNT(DISTINCT CASE WHEN issuetype = 'New Feature' THEN key END)       AS n_new_features,
        COUNT(DISTINCT CASE WHEN is_resolved = 1           THEN key END)       AS n_resolved,
        ROUND(AVG(CASE WHEN is_resolved = 1 THEN resolution_days END), 1)      AS avg_resolution_days,
        ROUND(MEDIAN(CASE WHEN is_resolved = 1 THEN resolution_days END), 1)   AS median_resolution_days,
        -- Ratio bugs : indicateur de qualité de la release
        ROUND(
            100.0 * COUNT(DISTINCT CASE WHEN issuetype = 'Bug' THEN key END)
            / NULLIF(COUNT(DISTINCT key), 0), 2
        ) AS bug_ratio_pct
    FROM {{ ref('int_fix_versions') }}
    WHERE version_major != 'Other'
    GROUP BY version_major, version_minor, issuetype
)

SELECT *
FROM by_version
ORDER BY version_major, version_minor, issuetype
