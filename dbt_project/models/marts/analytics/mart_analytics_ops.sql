-- Gold : table au grain ticket — 1 ligne par issue SPARK (~49 833 lignes)
-- Les agrégations (mensuelles, par type, par priorité) sont faites en DAX dans Power BI
-- Source : int_issues_analytics (toutes issues) + int_comments_aggregated

SELECT
    i.key,
    i.issuetype,
    i.priority,
    i.status,
    i.assignee,
    i.reporter,
    i.resolution,
    i.created_at,
    i.resolved_at,
    i.resolution_days,
    i.age_days,
    i.is_resolved,
    i.created_month,
    i.created_year,
    i.votes,
    i.watches,
    COALESCE(c.n_comments,   0) AS n_comments,
    COALESCE(c.n_commenters, 0) AS n_commenters,
    c.first_comment_at,
    -- Délai en heures entre la création et le premier commentaire
    CASE
        WHEN c.first_comment_at IS NOT NULL
        THEN DATEDIFF('hour', i.created_at, c.first_comment_at)
    END AS hours_to_first_response

FROM {{ ref('int_issues_analytics') }} i
LEFT JOIN {{ ref('int_comments_aggregated') }} c USING (key)
