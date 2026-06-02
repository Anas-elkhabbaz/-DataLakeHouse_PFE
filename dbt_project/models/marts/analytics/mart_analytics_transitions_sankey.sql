-- Gold : transitions de statut agrégées pour le Sankey diagram (Page 3)
-- Granularité : 1 ligne par paire (from_status × to_status) — ~12 lignes

SELECT
    from_status,
    to_status,
    COUNT(*)                                  AS n_transitions,
    ROUND(AVG(hours_in_from_status), 1)       AS avg_hours_before_transition,
    ROUND(MEDIAN(hours_in_from_status), 1)    AS median_hours_before_transition,
    SUM(is_reopen_event)                       AS n_reopen_events
FROM {{ ref('int_status_transitions') }}
GROUP BY from_status, to_status
ORDER BY n_transitions DESC
