-- Macro de nettoyage de texte JIRA (6 étapes enchaînées).
-- Utilisé dans int_issues_cleaned et int_comments_aggregated.
-- Paramètre : col_expr — expression SQL contenant le texte brut (ex: summary, comment_body)
--
-- Étapes 2 et 3 : les blocs {code} et {noformat} ne sont PAS supprimés entièrement.
-- On conserve les 200 premiers chars du contenu (messages d'erreur, stack traces),
-- encadrés par [CODE:...] pour signaler au modèle que c'est du code.
-- Supprimer ces blocs entièrement détruisait le signal le plus fort des tickets Bug
-- (ex: "java.lang.NullPointerException at SparkContext.scala:847").
{% macro clean_jira_text(col_expr) %}
    TRIM(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(
                                COALESCE({{ col_expr }}, ''),
                                '<[^>]+>', ' '              -- 1. Balises HTML
                            ),
                            -- 2. Blocs {code} : garder les 200 premiers chars du contenu
                            '\\{code[^}]*\\}(.{0,200}).*?\\{code\\}', '[CODE:\\1]', 1, 0, 'si'
                        ),
                        -- 3. Blocs {noformat} : même traitement
                        '\\{noformat[^}]*\\}(.{0,200}).*?\\{noformat\\}', '[CODE:\\1]', 1, 0, 'si'
                    ),
                    '\\[~[^\\]]+\\]', ' '   -- 4. Mentions utilisateur [~user]
                ),
                'https?://\\S+', ' '        -- 5. URLs
            ),
            '[\\n\\r\\t ]{2,}', ' '         -- 6. Espaces/sauts de ligne multiples
        )
    )
{% endmacro %}
