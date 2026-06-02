-- Intermédiaire : transitions de statut individuelles avec durée dans l'état précédent
-- Granularité : 1 ligne par transition — source du Sankey et du temps-par-statut
-- ~52 000 transitions pour les issues SPARK

WITH raw_transitions AS (
    SELECT
        key,
        COALESCE(NULLIF(TRIM(from_string), ''), 'Open') AS from_status,
        to_string                                        AS to_status,
        created                                          AS transition_at,
        LAG(created) OVER (PARTITION BY key ORDER BY created) AS prev_transition_at
    FROM {{ ref('stg_changelog') }}
    WHERE UPPER(field) = 'STATUS'
      AND to_string IS NOT NULL
),

with_duration AS (
    SELECT
        key,
        from_status,
        to_status,
        transition_at,
        -- Durée en heures dans l'état from_status avant cette transition
        CASE
            WHEN prev_transition_at IS NOT NULL
            THEN GREATEST(DATEDIFF('hour', prev_transition_at, transition_at), 0)
        END AS hours_in_from_status,
        -- Indicateur de réouverture (Resolved/Closed → Reopened)
        CASE
            WHEN UPPER(to_status) = 'REOPENED'
             AND UPPER(from_status) IN ('RESOLVED', 'CLOSED')
            THEN 1 ELSE 0
        END AS is_reopen_event
    FROM raw_transitions
)

SELECT *
FROM with_duration
