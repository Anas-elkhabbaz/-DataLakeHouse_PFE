-- Intermédiaire : features de liens entre tickets par issue

WITH base AS (
    SELECT * FROM {{ ref('stg_issuelinks') }}
)

SELECT
    key,
    COUNT(*)                                                           AS n_links_total,
    COUNT(CASE WHEN UPPER(type_name) = 'DUPLICATE'   THEN 1 END)      AS n_duplicates,
    COUNT(CASE WHEN UPPER(type_name) = 'BLOCKER'     THEN 1 END)      AS n_blocks,
    COUNT(CASE WHEN UPPER(type_name) = 'BLOCKED'     THEN 1 END)      AS n_blocked_by,
    COUNT(CASE WHEN UPPER(type_name) = 'REFERENCE'   THEN 1 END)      AS n_relates,
    COUNT(CASE WHEN UPPER(type_name) = 'CLONERS'     THEN 1 END)      AS n_clones,
    COUNT(CASE WHEN UPPER(type_name) = 'CONTAINER'   THEN 1 END)      AS n_container

FROM base
WHERE key IN (SELECT key FROM {{ ref('stg_issues') }})
GROUP BY key
