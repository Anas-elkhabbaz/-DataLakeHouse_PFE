-- Gold : activité quotidienne enrichie pour heatmap calendrier Power BI
-- Granularité : 1 ligne par jour
-- Ajoute une moyenne mobile 7 jours pour lisser la courbe de vélocité

WITH daily AS (
    SELECT * FROM {{ ref('int_daily_activity') }}
),

with_rolling AS (
    SELECT
        activity_date,
        year,
        week_of_year,
        day_of_week,
        issues_created,
        bugs_created,
        issues_resolved,
        changelog_events,
        active_issues,
        active_contributors,
        daily_backlog_delta,

        -- Cumul du backlog depuis le début (pour courbe de dette)
        SUM(daily_backlog_delta) OVER (ORDER BY activity_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_backlog,

        -- Moyenne mobile 7 jours : lisse les pics weekend
        ROUND(AVG(issues_created)  OVER (ORDER BY activity_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS issues_created_7d_avg,
        ROUND(AVG(issues_resolved) OVER (ORDER BY activity_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS issues_resolved_7d_avg,
        ROUND(AVG(active_contributors) OVER (ORDER BY activity_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS contributors_7d_avg,

        -- Intensité normalisée 0-100 pour la heatmap de couleur Power BI
        ROUND(
            100.0 * (issues_created + changelog_events)
            / NULLIF(MAX(issues_created + changelog_events) OVER (), 0),
        1) AS activity_intensity

    FROM daily
)

SELECT *
FROM with_rolling
ORDER BY activity_date
