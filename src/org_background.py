"""球團截距 u_org 代表什麼：用球團層的變數去解釋三十隊的截距（不取樣）。

    python src/org_background.py data/ <season> outputs/ --promoted <tag> [--war <tag>]

每隊一列：u_org（同樣的人多升或少升多少）、θ_org 四種能力平均（人才）、農場深度（該隊打席 ≥ 100 球員的期望 WAR 平均）、
農場大小（人季數）、母隊當季團隊 WAR、薪資（有 payroll.parquet 才有）。
報：每個球團變數與 u_org 的 Spearman；全部一起的最小平方；殘差最大的球團（真正說不清的那部分）。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from misplaced_profile import add_club_strength, spearman
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    org = pd.read_csv(out / str(season) / tag_p / "org_M1.csv", index_col=0)
    org = org[org.index != "UNK"]
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_club_strength(m, data)
    club_col = m["org"].fillna("").astype(str).str.split("/").str[0]
    tab = pd.DataFrame(index=org.index)
    tab["u_org 升級截距"] = org["升級 u_org"]
    for f in FACTORS:
        tab[f"人才 {f}"] = org[f"能力平均 {f}"]
    tab["農場大小（人季）"] = club_col.value_counts().reindex(org.index)
    tab["母隊團隊 WAR"] = m.groupby(club_col)["club_war_t"].first().reindex(org.index)
    if tag_w:
        eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
        d = m.merge(eta_w[["key_mlbam", "score_war", "score_preach"]], on="key_mlbam", how="inner")
        d["exp_war"] = d["score_preach"] * d["score_war"]
        d = d[d["PA"] >= 100]
        tab["農場深度（期望 WAR 平均）"] = (
            d.groupby(d["org"].fillna("").astype(str).str.split("/").str[0])["exp_war"].mean().reindex(org.index)
        )
    pay = data / "payroll.parquet"
    if pay.exists():
        p = pd.read_parquet(pay)
        p = p[p["Season"] == season].set_index("Team")["payroll"]
        tab["薪資（百萬美元）"] = (p / 1e6).reindex(org.index)
    y = tab["u_org 升級截距"]
    rows = []
    for c in tab.columns[1:]:
        r, n = spearman(tab[c], y)
        rows.append({"球團變數": c, "與截距的 Spearman": r, "n": n})
    cor = pd.DataFrame(rows).round(2)
    X_cols = [c for c in tab.columns[1:] if tab[c].notna().sum() >= 20]
    d2 = tab.dropna(subset=X_cols + [tab.columns[0]])
    Z = (d2[X_cols] - d2[X_cols].mean()) / d2[X_cols].std()
    X = np.column_stack([np.ones(len(d2)), Z.to_numpy(dtype=float)])
    b, *_ = np.linalg.lstsq(X, d2.iloc[:, 0].to_numpy(dtype=float), rcond=None)
    fit = X @ b
    resid = d2.iloc[:, 0].to_numpy(dtype=float) - fit
    r2 = 1 - resid.var() / d2.iloc[:, 0].var()
    coef = pd.DataFrame({"球團變數": X_cols, "係數（標準化）": b[1:]}).round(3)
    left = pd.DataFrame(
        {
            "球團": d2.index,
            "截距": d2.iloc[:, 0].round(2),
            "球團變數解釋的": fit.round(2),
            "剩下的（說不清的）": resid.round(2),
        }
    ).sort_values("剩下的（說不清的）")
    md = [
        f"# {season} 球團截距代表什麼 · {tag_p}\n",
        "u_org = 同樣的球員，這隊比平均多升或少升多少（升級段的球團隨機效應）。把它用球團層的變數去解釋，解釋掉的有名字，剩下的是文化。\n",
        "## 每隊一列\n",
        tab.round(2).to_markdown(),
        "",
        "## 每個球團變數與截距的相關（Spearman，30 隊）\n",
        cor.to_markdown(index=False),
        "",
        f"## 全部一起解釋（最小平方，標準化係數；R² = {r2:.2f}，n = {len(d2)}）\n",
        coef.to_markdown(index=False),
        "",
        "## 解釋完剩下的：哪些隊升得特別慢或特別快，而且球團變數說不出為什麼\n",
        left.head(6).to_markdown(index=False),
        "",
        left.tail(6).to_markdown(index=False),
        "",
    ]
    name = f"org_background_{tag_p}.md"
    (out / str(season) / name).write_text("\n".join(md))
    return {"相關（Spearman）": (cor, None), "一起解釋（標準化係數）": (coef, None), "R²": (pd.DataFrame({"統計量": ["R²", "n"], "值": [round(r2, 3), len(d2)]}), None)}
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args("球團截距 u_org 代表什麼：用球團層的變數去解釋三十隊的截距（不取樣）。", need_war=False)
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
