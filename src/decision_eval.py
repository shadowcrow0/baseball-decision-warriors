"""決策評估：三個問題的後處理（讀 fit_year 的 eta_M1.parquet 與 struct_M1.csv，不取樣）。

    python src/decision_eval.py data/ <season> outputs/ --promoted <tag> [--war <tag>]

問題一（能不能選得更好）：反事實選擇（球團實際升的人 vs 依 WAR 模型分數選同樣人數，後來的 WAR 總和）、錯放名單。
問題二（球探看的是什麼）：ζ 與四個 η 的相關；ζ 被 η 解釋的部分與剩下的部分；剩下的部分對 WAR 的相關。
問題三（被放掉的東西）：β_WAR − β_升級 表。
輸出 outputs/<season>/decision_<promoted tag>[_vs_<war tag>].md。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args

FACTORS = ["contact", "power", "speed", "discipline"]


def load_fit(out: Path, season: int, tag: str):
    d = out / str(season) / tag
    eta = pd.read_parquet(d / "eta_M1.parquet") if (d / "eta_M1.parquet").exists() else None
    st = pd.read_csv(d / "struct_M1.csv", index_col=0) if (d / "struct_M1.csv").exists() else None
    return eta, st


def counterfactual(eta_p: pd.DataFrame, eta_w: pd.DataFrame) -> pd.DataFrame:
    """同年同層級：球團實際升的人 vs 依 E[WAR] 分數選同樣人數，realized war_fwd 總和。"""
    if eta_w is None or "score_war" not in eta_w or "score_preach" not in eta_w:
        return None
    df = eta_p[["key_mlbam", "level_main", "promoted", "war_fwd", "war_observable", "PA"]].merge(
        eta_w[["key_mlbam", "score_war", "score_preach"]], on="key_mlbam", how="inner"
    )
    df = df[df["war_observable"].astype(bool) & df["score_war"].notna()]
    # 期望 WAR = P(六季內登上) × E[WAR | 登上]
    df["exp_war"] = df["score_preach"] * df["score_war"]
    rows = []
    for lvl, g in df.groupby("level_main"):
        k = int(g["promoted"].sum())
        if k == 0 or len(g) < 20:
            continue
        team = g[g["promoted"].astype(bool)]["war_fwd"].sum()
        model = g.nlargest(k, "exp_war")["war_fwd"].sum()
        overlap = len(set(g[g["promoted"].astype(bool)]["key_mlbam"]) & set(g.nlargest(k, "exp_war")["key_mlbam"]))
        rows.append(
            {
                "層級": lvl,
                "人數": len(g),
                "球團升了": k,
                "球團升的人後來 WAR": round(team, 1),
                "模型選同樣人數後來 WAR": round(model, 1),
                "差": round(model - team, 1),
                "重疊人數": overlap,
            }
        )
    return pd.DataFrame(rows)


def missed(eta_p: pd.DataFrame, eta_w: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """錯放名單：球團沒升（same/demoted/gone_*）但模型分數高的人，後來有沒有登上。"""
    score = "exp_war"
    df = eta_p.copy()
    if eta_w is not None and "score_war" in eta_w:
        df = df.merge(
            eta_w[["key_mlbam", "score_war", "score_preach", "score_debut_years"]], on="key_mlbam", how="left"
        )
        df["exp_war"] = df["score_preach"] * df["score_war"]
    else:
        score = "eta_contact"  # 沒有 WAR 模型時退而用接觸分數
    df = df[~df["promoted"].astype(bool) & (df["PA"] >= 100)]
    cols = [
        "key_mlbam",
        "level_main",
        "age",
        "PA",
        "pick_overall",
        "next_status",
        score,
        "score_debut_years",
        "reached_fwd",
        "war_fwd",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.nlargest(n, score)[cols].round(2)


def scout_decomposition(eta_p: pd.DataFrame) -> dict:
    """ζ 與 η：相關、被解釋的比例、剩下的部分與 WAR 的相關。"""
    if "zeta_scout" not in eta_p:
        return None
    d = eta_p.dropna(subset=["zeta_scout"])
    H = d[[f"eta_{f}" for f in FACTORS]].to_numpy()
    z = d["zeta_scout"].to_numpy()
    corr = pd.Series({f: np.corrcoef(z, d[f"eta_{f}"])[0, 1] for f in FACTORS}).round(3)
    X = np.column_stack([np.ones(len(z)), H])
    b = np.linalg.lstsq(X, z, rcond=None)[0]
    resid = z - X @ b
    r2 = 1 - resid.var() / z.var()
    out = {"corr": corr, "r2": round(float(r2), 3), "coef": pd.Series(b[1:], index=FACTORS).round(3)}
    if "war_observable" in d and d["war_observable"].astype(bool).any():
        dd = d[d["war_observable"].astype(bool)]
        rr = resid[d["war_observable"].astype(bool).to_numpy()]
        out["resid_vs_war"] = round(float(np.corrcoef(rr, dd["war_fwd"])[0, 1]), 3)
        out["eta_vs_war"] = pd.Series({f: np.corrcoef(dd[f"eta_{f}"], dd["war_fwd"])[0, 1] for f in FACTORS}).round(3)
        out["zeta_vs_war"] = round(float(np.corrcoef(dd["zeta_scout"], dd["war_fwd"])[0, 1]), 3)
    return out


def overlooked(st_p: pd.DataFrame, st_w: pd.DataFrame) -> pd.DataFrame:
    """β_WAR − β_升級。WAR 有兩段：幾年登上（t_，log 時間，負 = 更快，報告時取負號成「加速登上」）與 WAR|登上（w_）。"""
    if st_p is None or st_w is None:
        return None
    rows = []
    for name in [i for i in st_p.index if i.startswith("beta_")]:
        bp = st_p.loc[name, "mean"]
        bt = -st_w.loc["t_" + name, "mean"] if "t_" + name in st_w.index else np.nan  # 取負：正 = 更快登上
        bw = st_w.loc["w_" + name, "mean"] if "w_" + name in st_w.index else np.nan
        rows.append(
            {
                "係數": name.replace("beta_", "").rstrip("[0"),
                "升級決策 β": bp,
                "加速登上 −β_t": bt,
                "WAR|登上 β": bw,
                "被放掉（登上−升級）": bt - bp,
                "被放掉（WAR−升級）": bw - bp,
            }
        )
    return pd.DataFrame(rows).round(3)


def attach(eta: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """把建模表的結果與共變數欄位併進來（以建模表為準，覆蓋 fit 匯出的同名欄）。"""
    cols = [
        "key_mlbam",
        "promoted",
        "next_status",
        "war_fwd",
        "reached_fwd",
        "war_observable",
        "PA",
        "age",
        "org",
        "pick_overall",
    ]
    cols = [c for c in cols if c in m.columns]
    return eta.drop(columns=[c for c in cols if c != "key_mlbam" and c in eta.columns]).merge(
        m[cols], on="key_mlbam", how="left"
    )


def main(data_dir, season, out_dir, tag_p, tag_w):
    out = Path(out_dir)
    m = pd.read_parquet(Path(data_dir) / "model_bat.parquet")
    m = m[m["season"] == season]
    eta_p, st_p = load_fit(out, season, tag_p)
    eta_w, st_w = load_fit(out, season, tag_w) if tag_w else (None, None)
    eta_p = attach(eta_p, m)
    md = [f"# {season} 決策評估 · 升級 {tag_p}" + (f" · WAR {tag_w}" if tag_w else "") + "\n"]
    cf = counterfactual(eta_p, eta_w)
    md += [
        "## 問題一：反事實選擇（同層級、同人數）\n",
        cf.to_markdown(index=False) if cf is not None else "（需要 WAR 模型的分數：--war <tag>）",
        "",
    ]
    md += [
        "## 問題一：錯放名單（沒升、打席 ≥ 100、模型分數最高 20 人）\n",
        missed(eta_p, eta_w).to_markdown(index=False),
        "",
    ]
    sd = scout_decomposition(eta_p)
    md += ["## 問題二：球探先驗 ζ 與四種能力\n"]
    if sd is None:
        md += ["（需要 --scout 版本的 fit）", ""]
    else:
        md += [
            "ζ 與各 η 的相關：",
            sd["corr"].to_frame("corr").T.to_markdown(),
            f"\nζ 被四個 η 解釋的比例 R² = {sd['r2']}；迴歸係數：",
            sd["coef"].to_frame("coef").T.to_markdown(),
        ]
        if "resid_vs_war" in sd:
            md += [
                f"\nζ 剩下的部分（η 解釋不了的）與六季 WAR 的相關：**{sd['resid_vs_war']}**；ζ 本身與 WAR：{sd['zeta_vs_war']}；各 η 與 WAR：",
                sd["eta_vs_war"].to_frame("corr").T.to_markdown(),
            ]
        md += [""]
    ov = overlooked(st_p, st_w)
    md += [
        "## 問題三：被放掉的東西（β_WAR − β_升級）\n",
        ov.to_markdown(index=False) if ov is not None else "（需要兩個 fit 的 struct_M1.csv）",
        "",
    ]
    name = f"decision_{tag_p}" + (f"_vs_{tag_w}" if tag_w else "") + ".md"
    (out / str(season) / name).write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args("決策評估：三個問題的後處理（讀 fit_year 的 eta_M1.parquet 與 struct_M1.csv，不取樣）。")
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
