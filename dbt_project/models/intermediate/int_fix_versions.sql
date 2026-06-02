-- Intermédiaire : versions cibles par issue, normalisées en version majeure Spark
-- Source : changelog champ 'FIX VERSION' / 'FIX VERSION/S'
-- Granularité : 1 ligne par (key, version_assigned) — une issue peut avoir plusieurs versions

WITH raw_versions AS (
    SELECT
        key,
        to_string AS version_raw,
        created   AS assigned_at
    FROM {{ ref('stg_changelog') }}
    WHERE UPPER(field) IN ('FIX VERSION', 'FIX VERSION/S')
      AND to_string IS NOT NULL
      AND TRIM(to_string) != ''
),

normalized AS (
    SELECT
        key,
        version_raw,
        assigned_at,
        -- Version majeure : extrait X.Y de "X.Y.Z" ou "X.Y.Z-rcN"
        REGEXP_SUBSTR(version_raw, '^[0-9]+\\.[0-9]+') AS version_minor,
        -- Branche majeure : 1.x, 2.x, 3.x, 4.x
        CASE
            WHEN version_raw LIKE '1.%' THEN '1.x'
            WHEN version_raw LIKE '2.%' THEN '2.x'
            WHEN version_raw LIKE '3.%' THEN '3.x'
            WHEN version_raw LIKE '4.%' THEN '4.x'
            ELSE 'Other'
        END AS version_major,
        -- Indicateur release candidate
        CASE WHEN version_raw ILIKE '%-rc%' THEN 1 ELSE 0 END AS is_rc
    FROM raw_versions
)

SELECT
    n.key,
    n.version_raw,
    n.version_minor,
    n.version_major,
    n.is_rc,
    n.assigned_at,
    i.issuetype,
    i.resolution,
    i.priority,
    i.resolution_days,
    i.is_resolved
FROM normalized n
INNER JOIN {{ ref('int_issues_analytics') }} i USING (key)
