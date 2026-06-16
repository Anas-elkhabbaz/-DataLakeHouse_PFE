"""
Génère les graphes d'analyse exploratoire (EDA) du rapport PFE à partir des
données réelles hébergées sur Snowflake (PFE_SPARK.MARTS_ANALYTICS.MART_ANALYTICS_OPS).

Requêtes en lecture seule (SELECT / GROUP BY). Les PNG sont écrits dans le dossier
images/ du rapport LaTeX. Le script affiche aussi les valeurs agrégées (pour la prose).

Usage : python results/generate_eda_charts.py
"""
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import snowflake.connector

# --- Sortie : dossier images du rapport ---
OUT = Path(__file__).resolve().parents[2] / "Rapport final" / "PFE_Anas_UIR" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# --- Palette cohérente avec le rapport ---
C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE = "#4c72b0", "#55a868", "#dd8452", "#c44e52", "#8172b3"
PALETTE = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, "#937860", "#da8bc3", "#8c8c8c", "#ccb974"]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load_env():
    env = {}
    for line in (Path(__file__).resolve().parents[1] / ".env").read_text(
            encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def connect(env):
    return snowflake.connector.connect(
        account=env["SNOWFLAKE_ACCOUNT"], user=env["SNOWFLAKE_USER"],
        password=env["SNOWFLAKE_PASSWORD"], role=env.get("SNOWFLAKE_ROLE"),
        warehouse=env.get("SNOWFLAKE_WAREHOUSE"), database=env.get("SNOWFLAKE_DATABASE"),
        login_timeout=30, network_timeout=30)


T = "PFE_SPARK.MARTS_ANALYTICS.MART_ANALYTICS_OPS"


def q(cur, sql):
    cur.execute(sql)
    return cur.fetchall()


def bar(labels, values, title, xlabel, fname, color=None, rotate=0, pct=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = color or [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_title(title)
    ax.set_ylabel("Nombre de tickets")
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " ")))
    total = sum(values)
    for b, v in zip(bars, values):
        lab = f"{v:,}".replace(",", " ") + (f"\n{100*v/total:.1f}%" if pct else "")
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), lab, ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ->", fname)


def main():
    env = load_env()
    con = connect(env)
    cur = con.cursor()
    summary = {}

    total = q(cur, f"SELECT COUNT(*) FROM {T}")[0][0]
    resolved = q(cur, f"SELECT COUNT(*) FROM {T} WHERE IS_RESOLVED = 1")[0][0]
    yr = q(cur, f"SELECT MIN(YEAR(CREATED_AT)), MAX(YEAR(CREATED_AT)) FROM {T} "
                f"WHERE CREATED_AT IS NOT NULL AND YEAR(CREATED_AT) BETWEEN 2010 AND 2025")[0]
    summary["total"] = total
    summary["resolved"] = resolved
    summary["year_range"] = list(yr)
    print(f"Total tickets={total:,}  resolved={resolved:,}  years={yr}")

    # 1. Type d'incident
    rows = q(cur, f"SELECT ISSUETYPE, COUNT(*) c FROM {T} WHERE ISSUETYPE IS NOT NULL "
                  f"GROUP BY 1 ORDER BY c DESC")
    bar([r[0] for r in rows], [r[1] for r in rows],
        "Répartition des tickets par type d'incident", "Type d'incident",
        "eda_issuetype.png", rotate=30, pct=True)
    summary["issuetype"] = [(r[0], r[1]) for r in rows]

    # 2. Résolution (tickets résolus)
    rows = q(cur, f"SELECT RESOLUTION, COUNT(*) c FROM {T} WHERE RESOLUTION IS NOT NULL "
                  f"GROUP BY 1 ORDER BY c DESC")
    bar([r[0] for r in rows], [r[1] for r in rows],
        "Répartition des tickets par résolution", "Résolution",
        "eda_resolution.png", rotate=30, pct=True)
    summary["resolution"] = [(r[0], r[1]) for r in rows]

    # 3. Priorité
    order = ["Blocker", "Critical", "Major", "Minor", "Trivial"]
    rows = q(cur, f"SELECT PRIORITY, COUNT(*) c FROM {T} WHERE PRIORITY IS NOT NULL GROUP BY 1")
    d = {r[0]: r[1] for r in rows}
    labels = [p for p in order if p in d] + [k for k in d if k not in order]
    vals = [d[k] for k in labels]
    bar(labels, vals, "Répartition des tickets par priorité", "Priorité",
        "eda_priority.png", color=[C_RED, C_ORANGE, C_BLUE, C_GREEN, "#8c8c8c"][:len(labels)], pct=True)
    summary["priority"] = list(zip(labels, vals))

    # 4. Évolution annuelle des créations
    rows = q(cur, f"SELECT YEAR(CREATED_AT) y, COUNT(*) c FROM {T} "
                  f"WHERE CREATED_AT IS NOT NULL AND YEAR(CREATED_AT) BETWEEN 2010 AND 2025 "
                  f"GROUP BY 1 ORDER BY 1")
    years = [int(r[0]) for r in rows]
    cnts = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(years, cnts, marker="o", color=C_BLUE, linewidth=2)
    ax.fill_between(years, cnts, alpha=0.12, color=C_BLUE)
    ax.set_title("Évolution annuelle du nombre de tickets créés")
    ax.set_xlabel("Année de création")
    ax.set_ylabel("Nombre de tickets créés")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    for x, y in zip(years, cnts):
        ax.annotate(f"{y:,}".replace(",", " "), (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(OUT / "eda_timeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> eda_timeline.png")
    summary["timeline"] = list(zip(years, cnts))

    cur.close()
    con.close()
    print("\n=== SUMMARY (pour la prose) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
