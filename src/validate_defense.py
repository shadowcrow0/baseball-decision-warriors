"""守備直接量：驗證「球探看到而成績單沒有的是守備」（不取樣）。

    python src/validate_defense.py data/ <season> outputs/ --promoted <tag> --war <tag>

守備的直接量（都只有登上大聯盟之後才有，所以是驗證與結果拆解，不是預測變數）：
    FanGraphs  六季 Defense − Positional = 純守備分（守位調整拿掉）；Positional = 守位本身的價值
    Savant     OAA（比平均多幾個出局，2016 起，每年約 550 人）、fielding_runs_prevented；捕手 pop time（2015 起）；framing（只有 2015–2017）
三張表：
    一、四群人（錯放成功、升錯、球團獨具眼光、對照）登上後的守備：每 600 打席純守備分、OAA；中位數與 P(群 > 對照)
    二、捕手依小聯盟打擊能力分三等：登上後的純守備分、守位價值、pop time、framing（能拿到的年份）
    三、登上者：小聯盟成績單（打擊率、觸擊率、守位）與純守備分、OAA 的 Spearman；成績單裡有沒有任何東西看得到守備
輸出 outputs/<season>/validate_defense_<promoted tag>_vs_<war tag>.md。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from misplaced_profile import (
    build_candidates,
    add_position,
    add_club_strength,
    standardize_within_level,
    assign_groups,
    prob_superiority,
    spearman,
)
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]
WINDOW = 6


def fg_defense(data: Path, season: int) -> pd.DataFrame:
    fg = pd.read_parquet(data / "fg_batting.parquet", columns=["key_mlbam", "Season", "PA", "Defense", "Positional"])
    w = fg[(fg["Season"] > season) & (fg["Season"] <= season + WINDOW)]
    s = w.groupby("key_mlbam")[["PA", "Defense", "Positional"]].sum().rename(columns={"PA": "PA_mlb"})
    s["Fld"] = s["Defense"] - s["Positional"]
    pa600 = s["PA_mlb"].where(s["PA_mlb"] >= 100) / 600.0
    s["fld_rate"] = s["Fld"] / pa600
    s["pos_rate"] = s["Positional"] / pa600
    return s.reset_index()


def savant_defense(data: Path, season: int) -> pd.DataFrame:
    sv = pd.read_parquet(data / "savant_leaderboards.parquet")
    w = sv[(sv["season"] > season) & (sv["season"] <= season + WINDOW)]
    out = {}
    o = w[w["board"] == "oaa"]
    if len(o):
        out["oaa"] = o.groupby("key_mlbam")["outs_above_average"].sum()
        out["frp"] = o.groupby("key_mlbam")["fielding_runs_prevented"].sum()
    p = w[w["board"] == "poptime"]
    if len(p) and "pop_2b_sba" in p:
        out["pop_2b"] = p.groupby("key_mlbam")["pop_2b_sba"].mean()
    f = w[w["board"] == "framing"]
    if len(f) and "runs_extra_strikes" in f:
        out["framing_runs"] = f.groupby("key_mlbam")["runs_extra_strikes"].sum()
    return pd.DataFrame(out).reset_index()


def group_table(d: pd.DataFrame, group: pd.Series, cols: list) -> pd.DataFrame:
    order = ["錯放成功", "升錯", "球團獨具眼光", "對照"]
    rows = []
    for disp, c in cols:
        row = {"守備量": disp}
        ref = d.loc[group == "對照", c]
        for g in order:
            x = d.loc[group == g, c].dropna()
            row[f"{g} n"] = len(x)
            row[f"{g} 中位"] = x.median()
        for g in order[:3]:
            row[f"P({g} > 對照)"] = prob_superiority(d.loc[group == g, c], ref)
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_position(m, data)
    m = add_club_strength(m, data)
    m, names = build_candidates(m)
    eta_p = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
    sc = eta_p[["key_mlbam"] + [f"eta_{f}" for f in FACTORS]].merge(
        eta_w[["key_mlbam", "score_war", "score_preach"]], on="key_mlbam", how="inner"
    )
    d = m.merge(sc, on="key_mlbam", how="inner")
    d["exp_war"] = d["score_preach"] * d["score_war"]
    d = d[(d["PA"] >= 100) & d["exp_war"].notna()].reset_index(drop=True)
    d = d.merge(fg_defense(data, season), on="key_mlbam", how="left").merge(
        savant_defense(data, season), on="key_mlbam", how="left"
    )
    group = assign_groups(d, "exp_war")
    reached = d["reached_fwd"].astype(bool) & (d["PA_mlb"] >= 100)
    cols = [
        ("每 600 打席純守備分（FG Defense − Positional）", "fld_rate"),
        ("每 600 打席守位價值（FG Positional）", "pos_rate"),
        ("OAA 六季合計（Savant）", "oaa"),
        ("守備防止失分（Savant）", "frp"),
    ]
    md = [
        f"# {season} 守備直接量 · {tag_p} / {tag_w}\n",
        f"打席 ≥ 100：{len(d)} 人；登上且大聯盟打席 ≥ 100：{int(reached.sum())} 人。守備量都是登上之後六季的，驗證用。\n",
        "## 一、四群人登上後的守備（只看登上且打席 ≥ 100 的人；中位數與 P(群 > 對照)）\n",
        "球團獨具眼光（模型不看好、球團升了、後來成功）如果純守備分和 OAA 明顯高於對照組，「球探看到的是守備」就是直接量到的。\n",
        group_table(d[reached], group[reached], cols).to_markdown(index=False),
        "",
    ]
    # 二、捕手
    c = d[(d["pos_group"] == "捕手")].copy()
    bat = (c["eta_contact"] + c["eta_power"]) / 2
    c["bat_tier"] = bat.groupby(c["level_main"]).transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["低", "中", "高"])
    )
    rows = []
    for t in ["低", "中", "高"]:
        g = c[c["bat_tier"] == t]
        r = g[g["reached_fwd"].astype(bool) & (g["PA_mlb"] >= 100)]
        rows.append(
            {
                "捕手打擊能力": t,
                "人數": len(g),
                "登上且打席 ≥ 100": len(r),
                "純守備分中位": r["fld_rate"].median(),
                "守位價值中位": r["pos_rate"].median(),
                "六季 PA 中位": r["PA_mlb"].median(),
                "pop time 中位（秒）": r["pop_2b"].median() if "pop_2b" in r else np.nan,
                "framing 合計中位（2015–17）": r["framing_runs"].median() if "framing_runs" in r else np.nan,
                "六季 WAR ≥ 2 比例": (g["war_fwd"] >= 2).mean(),
            }
        )
    md += [
        "## 二、捕手依小聯盟打擊能力分三等：登上後的守備與守位價值\n",
        "會打的捕手守位價值中位數低 = 被移去一壘/DH；純守備分高 = 真的守得好。pop time 越小越好。\n",
        pd.DataFrame(rows).round(2).to_markdown(index=False),
        "",
    ]
    # 三、成績單看不看得到守備
    z = standardize_within_level(d, [k for _, k in names])
    rows = []
    for disp, k in names:
        r1, n1 = spearman(z.loc[reached, k], d.loc[reached, "fld_rate"])
        r2, n2 = spearman(z.loc[reached, k], d.loc[reached, "oaa"])
        rows.append({"小聯盟成績單": disp, "與純守備分的 Spearman": r1, "與 OAA 的 Spearman": r2, "n": n1})
    tab = pd.DataFrame(rows)
    tab["_k"] = tab[["與純守備分的 Spearman", "與 OAA 的 Spearman"]].abs().max(axis=1)
    md += [
        "## 三、小聯盟成績單裡，有沒有任何東西看得到登上後的守備（登上者；Spearman；2/√n ≈ "
        + f"{2/np.sqrt(max(int(reached.sum()),1)):.2f}）\n",
        tab.sort_values("_k", ascending=False).drop(columns="_k").head(15).round(3).to_markdown(index=False),
        "",
    ]
    name = f"validate_defense_{tag_p}_vs_{tag_w}.md"
    (out / str(season) / name).write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args("守備直接量：驗證「球探看到而成績單沒有的是守備」（不取樣）。")
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
