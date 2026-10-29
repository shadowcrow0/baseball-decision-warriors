"""選球好的人跟選球差的人，各自是靠什麼升上去的？把球員依「選得到球」分三等，每等各自估一次係數，再比分布（不取樣）。

    python src/discipline_split.py data/ <season> outputs/ --promoted <tag> --war <tag>

四張表：
    1. 三個結果（升級、登上、WAR|登上）× 三等（選球差 / 中 / 好）的係數：打得到球、打得遠、跑得快、球探評價、非選秀。
       「選球好的人靠什麼上去」= 選球好那欄裡最大的係數；跟選球差那欄差兩倍標準誤標 ◆。
    2. 同樣三等，改用成績單上的原始數字（打擊率、保送率、三振率、全壘打率、年齡）估：球隊看選球好的人時，看的是哪個數字。
    3. 選球三等 × 打得到球三等：升級率、登上率、WAR ≥ 2 比例、登上者 WAR 中位。回答「選球只有配上接觸才有用？」
    4. 選球好的人裡，登上的 vs 沒登上的：每個候選的中位數與 P(登上 > 沒登上)；選球差的人同樣一張。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from slope_by_group import design, fit_by_group, table, logistic_irls, ols, X_NAMES
from misplaced_profile import build_candidates, standardize_within_level, prob_superiority
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]
STAT_NAMES = [
    ("打擊率", "rate_AVG"),
    ("保送率", "rate_BBpct"),
    ("三振率", "rate_Kpct"),
    ("全壘打率", "rate_HR_AB"),
    ("長打比", "rate_XBH_H"),
    ("盜壘嘗試率", "rate_SBA_OBT"),
    ("年齡", "age"),
    ("順位（log）", "log_pick"),
    ("非選秀", "undrafted"),
]


def tertile_within_level(x: pd.Series, level: pd.Series, labels: list) -> pd.Series:
    pct = x.groupby(level).rank(pct=True)
    return pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=labels, include_lowest=True)


def stat_design(d: pd.DataFrame, z: pd.DataFrame) -> tuple[np.ndarray, list]:
    """成績單版設計矩陣：常數 + 原始數字（層級內標準化）+ 層級 dummies。"""
    cols = [np.ones(len(d))] + [z[c].fillna(0.0).to_numpy(dtype=float) for _, c in STAT_NAMES]
    names = ["常數"] + [n for n, _ in STAT_NAMES]
    base = d["level_main"].mode().iloc[0]
    for lvl in sorted(d["level_main"].unique()):
        if lvl != base:
            cols.append((d["level_main"] == lvl).to_numpy(dtype=float))
            names.append(f"層級 {lvl}")
    return np.column_stack(cols), names


def fit_stats_by_group(d: pd.DataFrame, z: pd.DataFrame, y: np.ndarray, groups: dict, kind: str) -> dict:
    out = {}
    for gname, mask in groups.items():
        X, names = stat_design(d[mask], z[mask])
        yy = y[mask]
        if kind == "logit" and (yy.sum() < 10 or (1 - yy).sum() < 10):
            continue
        b, se = logistic_irls(X, yy) if kind == "logit" else ols(X, yy)
        out[gname] = {n: (b[i], se[i]) for i, n in enumerate(names) if n in [s for s, _ in STAT_NAMES]}
        out[gname]["n"] = (int(mask.sum()), 0)
    return out


def stat_table(fits: dict, pairs: list) -> pd.DataFrame:
    rows = []
    for name in [s for s, _ in STAT_NAMES] + ["n"]:
        row = {"係數": name}
        for g, f in fits.items():
            b, se = f.get(name, (np.nan, np.nan))
            row[g] = f"{b:.0f}" if name == "n" else f"{b:+.2f} ({se:.2f})"
        for a, c in pairs:
            if name != "n" and a in fits and c in fits:
                ba, sa = fits[a][name]
                bc, sc = fits[c][name]
                diff = ba - bc
                se = np.sqrt(sa**2 + sc**2)
                row[f"{a} − {c}"] = f"{diff:+.2f}" + (" ◆" if abs(diff) > 2 * se else "")
        rows.append(row)
    return pd.DataFrame(rows)


def grid(d: pd.DataFrame) -> pd.DataFrame:
    """選球三等 × 打得到球三等。"""
    rows = []
    obs = d["war_observable"].astype(bool)
    for (disc, con), g in d.groupby(["選球", "接觸"], observed=True):
        go = g[obs.loc[g.index]]
        r = go[go["reached_fwd"].astype(bool)]
        rows.append(
            {
                "選得到球": disc,
                "打得到球": con,
                "人數": len(g),
                "升級率": round(g["promoted"].mean(), 2),
                "登上率": round(go["reached_fwd"].mean(), 2) if len(go) else np.nan,
                "WAR ≥ 2 比例（全體）": round((go["war_fwd"] >= 2).mean(), 2) if len(go) else np.nan,
                "登上者 WAR 中位": round(r["war_fwd"].median(), 1) if len(r) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def reached_profile(d: pd.DataFrame, z: pd.DataFrame, names: list, mask: np.ndarray) -> pd.DataFrame:
    """某一群裡，登上的 vs 沒登上的：中位數（層級內標準化）與 P(登上 > 沒登上)，依離 0.5 最遠排序。"""
    obs = d["war_observable"].astype(bool).to_numpy() & mask
    r = d["reached_fwd"].astype(bool).to_numpy()
    rows = []
    for disp, col in names:
        a = z.loc[obs & r, col]
        b = z.loc[obs & ~r, col]
        if a.notna().sum() < 10 or b.notna().sum() < 10:
            continue
        rows.append(
            {
                "候選": disp,
                "登上者中位": round(a.median(), 2),
                "沒登上者中位": round(b.median(), 2),
                "P(登上 > 沒登上)": round(prob_superiority(a, b), 2),
                "n 登上 / 沒登上": f"{int(a.notna().sum())} / {int(b.notna().sum())}",
            }
        )
    out = pd.DataFrame(rows)
    out["_k"] = (out["P(登上 > 沒登上)"] - 0.5).abs()
    return out.sort_values("_k", ascending=False).drop(columns="_k").head(14)


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m, cand = build_candidates(m)
    eta_p = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
    sc = eta_p[["key_mlbam"] + [f"eta_{f}" for f in FACTORS] + (["zeta_scout"] if "zeta_scout" in eta_p else [])].merge(
        eta_w[["key_mlbam", "score_war", "score_preach"]], on="key_mlbam", how="inner"
    )
    d = m.merge(sc, on="key_mlbam", how="inner")
    if "zeta_scout" not in d:
        d["zeta_scout"] = 0.0
    d = d[d["PA"] >= 100].reset_index(drop=True)
    d["選球"] = tertile_within_level(d["eta_discipline"], d["level_main"], ["選球差", "選球中", "選球好"])
    d["接觸"] = tertile_within_level(d["eta_contact"], d["level_main"], ["接觸差", "接觸中", "接觸好"])
    names_all = cand + [(f"能力：{f}", f"eta_{f}") for f in FACTORS] + [("球探評價 ζ", "zeta_scout")]
    z = standardize_within_level(d, list(dict.fromkeys([c for _, c in names_all] + [c for _, c in STAT_NAMES])))
    groups = {
        "選球差": (d["選球"] == "選球差").to_numpy(),
        "選球中": (d["選球"] == "選球中").to_numpy(),
        "選球好": (d["選球"] == "選球好").to_numpy(),
    }
    pairs = [("選球好", "選球差")]
    obs = d["war_observable"].astype(bool).to_numpy()
    r = obs & d["reached_fwd"].astype(bool).to_numpy()
    dd = d[obs].reset_index(drop=True)
    dr = d[r].reset_index(drop=True)
    g2 = {k: v[obs] for k, v in groups.items()}
    g3 = {k: v[r] for k, v in groups.items()}
    zz = z[obs].reset_index(drop=True)
    zr = z[r].reset_index(drop=True)
    y_p = d["promoted"].astype(float).to_numpy()
    y_r = dd["reached_fwd"].astype(float).to_numpy()
    y_w = np.arcsinh(dr["war_fwd"].to_numpy(dtype=float))
    t_prom = table(fit_by_group(d, y_p, groups, "logit"), pairs)
    t_reach = table(fit_by_group(dd, y_r, g2, "logit"), pairs)
    t_war = table(fit_by_group(dr, y_w, g3, "ols"), pairs)
    s_prom = stat_table(fit_stats_by_group(d, z, y_p, groups, "logit"), pairs)
    s_reach = stat_table(fit_stats_by_group(dd, zz, y_r, g2, "logit"), pairs)
    s_war = stat_table(fit_stats_by_group(dr, zr, y_w, g3, "ols"), pairs)
    g_tab = grid(d)
    rp_good = reached_profile(d, z, names_all, groups["選球好"])
    rp_bad = reached_profile(d, z, names_all, groups["選球差"])
    md = [
        f"# {season} 選球好與選球差的人，各自靠什麼升上去 · {tag_p} / {tag_w}\n",
        "「選得到球」能力在層級內分三等。每等各自估一次係數（括號 = 標準誤；選球好 − 選球差 差超過兩倍標準誤標 ◆）。打席 ≥ 100。\n",
        "## 1. 用能力分數估：哪種能力把他們送上去\n",
        "### 升級：球團隔年升不升\n",
        t_prom.to_markdown(index=False),
        "",
        "### 登上：六季內登上與否\n",
        t_reach.to_markdown(index=False),
        "",
        "### WAR|登上：登上者的 asinh(六季 WAR)\n",
        t_war.to_markdown(index=False),
        "",
        "## 2. 用成績單上的數字估：球隊看選球好的人時看的是哪個數字\n",
        "### 升級\n",
        s_prom.to_markdown(index=False),
        "",
        "### 登上\n",
        s_reach.to_markdown(index=False),
        "",
        "### WAR|登上\n",
        s_war.to_markdown(index=False),
        "",
        "## 3. 選球 × 接觸：升級率、登上率、WAR 分布\n",
        "選球要配上接觸才有用的話：接觸好那三列裡選球好的登上率 / WAR ≥ 2 比例應該高於選球差；接觸差那三列應該沒差。\n",
        g_tab.to_markdown(index=False),
        "",
        "## 4. 選球好的人裡，登上的 vs 沒登上的\n",
        rp_good.to_markdown(index=False),
        "",
        "## 4b. 選球差的人裡，登上的 vs 沒登上的\n",
        rp_bad.to_markdown(index=False),
        "",
    ]
    name = f"discipline_split_{tag_p}_vs_{tag_w}.md"
    (out / str(season) / name).write_text("\n".join(md))
    # 給 all_seasons.py 跨季合併用：表名 → (表, 括號裡是什麼)
    return {
        "1 能力係數 · 升級": (t_prom, "se"),
        "1 能力係數 · 登上": (t_reach, "se"),
        "1 能力係數 · WAR|登上": (t_war, "se"),
        "2 成績單係數 · 升級": (s_prom, "se"),
        "2 成績單係數 · 登上": (s_reach, "se"),
        "2 成績單係數 · WAR|登上": (s_war, "se"),
        "3 選球 × 接觸九格": (g_tab, None),
        "4 選球好：登上 vs 沒登上": (rp_good, None),
        "4b 選球差：登上 vs 沒登上": (rp_bad, None),
    }
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args(
        "選球好的人跟選球差的人，各自是靠什麼升上去的？把球員依「選得到球」分三等，每等各自估一次係數，再比分布（不取樣）。"
    )
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
