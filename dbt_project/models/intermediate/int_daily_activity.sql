-- Intermédiaire : activité quotidienne — créations d'issues + événements changelog
-- Granularité : 1 ligne par jour — source de la heatmap calendrier Power BI

WITH issue_creations AS (
    SELECT
        DATE_TRUNC('day', created_at) AS activity_date,
        COUNT(*)                       AS issues_created,
        COUNT(CASE WHEN issuetype = 'Bug' THEN 1 END) AS bugs_created
    FROM {{ ref('int_issues_analytics') }}
    GROUP BY 1
),

changelog_activity AS (
    SELECT
        DATE_TRUNC('day', created) AS activity_date,
        COUNT(*)                    AS changelog_events,
        COUNT(DISTINCT key)         AS active_issues,
        COUNT(DISTINCT author)      AS active_contributors
    FROM {{ ref('stg_changelog') }}
    GROUP BY 1
),

issue_resolutions AS (
    SELECT
        DATE_TRUNC('day', resolved_at) AS activity_date,
        COUNT(*)                        AS issues_resolved
    FROM {{ ref('int_issues_analytics') }}
    WHERE resolved_at IS NOT NULL
    GROUP BY 1
)

SELECT
    COALESCE(ic.activity_date, ca.activity_date, ir.activity_date) AS activity_date,
    DAYOFWEEK(COALESCE(ic.activity_date, ca.activity_date, ir.activity_date)) AS day_of_week,
    WEEKOFYEAR(COALESCE(ic.activity_date, ca.activity_date, ir.activity_date)) AS week_of_year,
    YEAR(COALESCE(ic.activity_date, ca.activity_date, ir.activity_date))       AS year,
    COALESCE(ic.issues_created,   0) AS issues_created,
    COALESCE(ic.bugs_created,     0) AS bugs_created,
    COALESCE(ir.issues_resolved,  0) AS issues_resolved,
    COALESCE(ca.changelog_events, 0) AS changelog_events,
    COALESCE(ca.active_issues,    0) AS active_issues,
    COALESCE(ca.active_contributors, 0) AS active_contributors,
    -- Solde du jour : positif = accumulation, négatif = réduction du backlog
    COALESCE(ic.issues_created, 0) - COALESCE(ir.issues_resolved, 0) AS daily_backlog_delta
FROM issue_creations ic
FULL OUTER JOIN changelog_activity  ca USING (activity_date)
FULL OUTER JOIN issue_resolutions   ir USING (activity_date)
