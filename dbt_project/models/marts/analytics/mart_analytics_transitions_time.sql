-- Gold : temps moyen passé dans chaque statut (Page 3 — heatmap durées)
-- Granularité : 1 ligne par statut source — ~6 lignes

SELECT
    from_status                                                              AS status,
    COUNT(*)                                                                 AS n_exits,
    ROUND(AVG(hours_in_from_status), 1)                                      AS avg_hours,
    ROUND(MEDIAN(hours_in_from_status), 1)                                   AS median_hours,
    ROUND(AVG(hours_in_from_status) / 24.0, 1)                               AS avg_days,
    ROUND(MEDIAN(hours_in_from_status) / 24.0, 1)                            AS median_days,
    -- Libellé formaté pour l'axe Power BI
    CONCAT(from_status, ' (moy. ', ROUND(AVG(hours_in_from_status)/24, 1)::VARCHAR, 'j)') AS status_label
FROM {{ ref('int_status_transitions') }}
WHERE hours_in_from_status IS NOT NULL
GROUP BY from_status
ORDER BY avg_hours DESC
