"""把「現有的」拆開再找：wOBA 的成分各自推動什麼；能力模型（M1）和綜合指標模型（Mh）意見不同的人，後來誰對（不取樣）。

    python src/disagree.py data/ <season> outputs/ --promoted <tag> [--war <tag>]

一、拆 wOBA：升級、登上、WAR|登上 各對四個成分（一壘安打率、二三壘安打率、全壘打率、保送率，層級內標準化）+ 層級 + log 打席 做迴歸。
二、意見不同的人：同層級內，M1 的升級分數排名 − Mh 的排名，分十等。每一等看實際升級率、登上率、六季 WAR ≥ 2 的比例、每百人 WAR 總和。
    M1 排得高、Mh 排得低的那一端，如果後來登上與 WAR 比較好 = 能力模型看到綜合指標看不到的；反之亦然。
三、意見最不同的兩成人裡，誰的排序更接近真相：M1 與 Mh 分數各自對「登上」「WAR ≥ 2」的 P(高分者 > 低分者)（AUC）。
四、意見不同的人長什麼樣：排名差與打席、四種能力、年齡、守位的 Spearman。
輸出 outputs/<season>/disagree_<promoted tag>.md。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from misplaced_profile import add_position, logistic_irls, spearman, prob_superiority
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]
COMPONENTS = [("一壘安打率", "r_1b"), ("二三壘安打率", "r_xbh"), ("全壘打率", "r_hr"), ("保送率", "r_bb")]


def components(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    pa = m["PA"].replace(0, np.nan)
    m["r_1b"] = (m["H"] - m["XBH"]) / pa
    m["r_xbh"] = (m["XBH"] - m["HR"]) / pa
    m["r_hr"] = m["HR"] / pa
    m["r_bb"] = m["BB"] / pa
    for _, c in COMPONENTS:
        m[c] = (
            m[c].groupby(m["level_main"]).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0).clip(-4, 4)
        )
    return m


def design(d: pd.DataFrame) -> tuple[np.ndarray, list]:
    cols = (
        [np.ones(len(d))]
        + [d[c].to_numpy(dtype=float) for _, c in COMPONENTS]
        + [np.log(d["PA"].to_numpy(dtype=float))]
    )
    names = ["常數"] + [n for n, _ in COMPONENTS] + ["log 打席"]
    base = d["level_main"].mode().iloc[0]
    for lvl in sorted(d["level_main"].unique()):
        if lvl != base:
            cols.append((d["level_main"] == lvl).to_numpy(dtype=float))
            names.append(f"層級 {lvl}")
    return np.column_stack(cols), names


def woba_split(d: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    X, names = design(d)
    b, se = logistic_irls(X, d["promoted"].astype(float).to_numpy())
    rows["升級（logit）"] = (b, se)
    obs = d["war_observable"].astype(bool).to_numpy()
    if obs.sum() > 50:  # 六季窗口還沒到的年份（2023）沒有登上與 WAR 可算
        Xo, _ = design(d[obs])
        b, se = logistic_irls(Xo, d.loc[obs, "reached_fwd"].astype(float).to_numpy())
        rows["登上（logit）"] = (b, se)
        r = obs & d["reached_fwd"].astype(bool).to_numpy()
        if r.sum() > 30:
            Xr, _ = design(d[r])
            y = np.arcsinh(d.loc[r, "war_fwd"].to_numpy(dtype=float))
            bb, *_ = np.linalg.lstsq(Xr, y, rcond=None)
            res = y - Xr @ bb
            s2 = res @ res / max(len(y) - Xr.shape[1], 1)
            rows["WAR|登上（asinh）"] = (bb, np.sqrt(np.diag(s2 * np.linalg.inv(Xr.T @ Xr))))
    out = []
    for n, _ in COMPONENTS:
        i = names.index(n)
        out.append({"wOBA 成分": n, **{k: f"{v[0][i]:+.2f} ({v[1][i]:.2f})" for k, v in rows.items()}})
    return pd.DataFrame(out)


def disagreement(d: pd.DataFrame, a: str, b: str) -> pd.Series:
    """同層級內兩個分數的排名差（百分位），正 = M1 排得比 Mh 高。"""
    ra = d[a].groupby(d["level_main"]).rank(pct=True)
    rb = d[b].groupby(d["level_main"]).rank(pct=True)
    return ra - rb


def decile_table(d: pd.DataFrame, diff: pd.Series) -> pd.DataFrame:
    dec = pd.qcut(diff.rank(method="first"), 10, labels=False)
    rows = []
    for k in range(10):
        g = d[dec == k]
        obs = g[g["war_observable"].astype(bool)]
        rows.append(
            {
                "十等（1 = Mh 高、M1 低）": k + 1,
                "人數": len(g),
                "排名差中位": diff[dec == k].median(),
                "實際升級率": g["promoted"].mean(),
                "登上率": obs["reached_fwd"].mean(),
                "六季 WAR ≥ 2 比例": (obs["war_fwd"] >= 2).mean(),
                "每百人 WAR 總和": obs["war_fwd"].sum() / max(len(obs), 1) * 100,
            }
        )
    return pd.DataFrame(rows).round(3)


def who_is_right(d: pd.DataFrame, diff: pd.Series, a: str, b: str) -> pd.DataFrame:
    """意見最不同的兩成人（|排名差| 最大）裡，各分數對結果的 AUC = P(有結果的人分數 > 沒結果的人分數)。"""
    top = diff.abs() >= diff.abs().quantile(0.8)
    g = d[top & d["war_observable"].astype(bool)]
    rows = []
    for label, y in [
        ("登上", g["reached_fwd"].astype(bool)),
        ("六季 WAR ≥ 2", g["war_fwd"] >= 2),
        ("球團升了", g["promoted"].astype(bool)),
    ]:
        rows.append(
            {
                "結果": label,
                "人數": len(g),
                "有結果的人數": int(y.sum()),
                "M1 的 AUC": prob_superiority(g.loc[y, a], g.loc[~y, a]),
                "Mh 的 AUC": prob_superiority(g.loc[y, b], g.loc[~y, b]),
            }
        )
    return pd.DataFrame(rows).round(3)


def profile(d: pd.DataFrame, diff: pd.Series) -> pd.DataFrame:
    cand = [
        ("打席", "PA"),
        ("年齡", "age"),
        ("打得到球", "eta_contact"),
        ("打得遠", "eta_power"),
        ("跑得快", "eta_speed"),
        ("選得到球", "eta_discipline"),
        ("能力不確定（打得到球 sd）", "eta_contact_sd"),
        ("捕手", "C"),
        ("游擊/二壘", "SS2B"),
        ("一壘/指定", "1BDH"),
        ("順位（有選秀）", "pick_overall"),
    ]
    rows = []
    for disp, c in cand:
        if c in d:
            r, n = spearman(d[c], diff)
            rows.append({"特徵": disp, "與排名差（M1 − Mh）的 Spearman": r, "n": n})
    return pd.DataFrame(rows).round(3)


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_position(m, data)
    m = components(m)
    eta = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    need = ["score_promoted", "score_promoted_Mh"]
    if not all(c in eta for c in need):
        raise SystemExit("eta_M1.parquet 沒有 score_promoted_Mh：用新版 fit_year.py --reuse 重新彙整一次")
    keep = ["key_mlbam"] + [c for c in eta.columns if c.startswith("score_") or c.startswith("eta_")]
    d = m.merge(eta[keep], on="key_mlbam", how="inner")
    d = d[(d["PA"] >= 100) & d["score_promoted"].notna()].reset_index(drop=True)
    diff = disagreement(d, "score_promoted", "score_promoted_Mh")
    note = (
        ""
        if d["war_observable"].astype(bool).sum() > 50
        else "（這一年的六季窗口還沒到：登上與 WAR 的欄位沒有資料，只看升級）"
    )
    md = [
        f"# {season} 拆開現有的：wOBA 成分、M1 與 Mh 意見不同的人 · {tag_p}\n",
        f"打席 ≥ 100：{len(d)} 人。{note}\n",
        "## 一、wOBA 的四個成分各自推動什麼（層級內標準化；括號 = 標準誤）\n",
        woba_split(d).to_markdown(index=False),
        "",
        "## 二、M1 與 Mh 的升級分數排名差，分十等\n",
        "排名差 = M1 的層級內百分位 − Mh 的。第 10 等 = 能力模型看好、綜合指標不看好；第 1 等相反。看兩端的登上率與 WAR 誰高。\n",
        decile_table(d, diff).to_markdown(index=False),
        "",
        "## 三、意見最不同的兩成人裡，誰的排序接近真相（AUC，0.5 = 隨機）\n",
        who_is_right(d, diff, "score_promoted", "score_promoted_Mh").to_markdown(index=False),
        "",
        "## 四、M1 比 Mh 看好的是什麼樣的人\n",
        profile(d, diff).to_markdown(index=False),
        "",
    ]
    if "score_war_asinh" in d and "score_war_asinh_Mh" in d:
        r = d[d["war_observable"].astype(bool) & d["score_war_asinh"].notna()].reset_index(drop=True)
        dw = disagreement(r, "score_war_asinh", "score_war_asinh_Mh")
        md += [
            "## 五、WAR 分數的排名差（M1 − Mh），分十等\n",
            decile_table(r, dw).to_markdown(index=False),
            "",
            "## 六、WAR 分數意見最不同的兩成人裡，誰對\n",
            who_is_right(r, dw, "score_war_asinh", "score_war_asinh_Mh").to_markdown(index=False),
            "",
        ]
    name = f"disagree_{tag_p}.md"
    (out / str(season) / name).write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args(
        "把「現有的」拆開再找：wOBA 的成分各自推動什麼；能力模型（M1）和綜合指標模型（Mh）意見不同的人，後來誰對（不取樣）。"
    )
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
