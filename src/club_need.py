"""個人層次：誰被塞住、誰被拉上去，跟母隊的狀況有沒有關？「缺什麼補什麼」是真的嗎？

    python src/club_need.py data/ <season> outputs/ --promoted <tag> --war <tag>

每個人季（打席 ≥ 100）配上他母隊的狀況：
    母隊該守位 WAR（當季 / 隔季扣掉自己）   低 = 這個守位缺人
    守位主力 WAR、守位主力年齡（當季）        主力強 = 卡位；主力老 = 位子快空
    母隊整體 WAR（當季）                     弱隊缺人、重建
    農場同守位排在前面的人數（分數比他高）    農場內塞車
    農場同守位層級比他高的人數
    農場同守位總人數
然後問四件事：
    1. 模型分數之外，這些母隊變數還能不能說明升不升、登不登、登上後 WAR、幾年後登上、登上那年打席（beyond-the-model）
    2. 缺人會不會把門檻降低：登上 ~ 模型 + 缺口 + 缺口 × 模型；缺 / 中 / 滿 × 模型分數三等 的登上率與 WAR 分布
    3. 守位缺口 vs 整體弱隊：兩個一起放，誰留下
    4. 四群（錯放成功 / 升錯 / 球團獨具眼光 / 對照）在這些變數上的分布：被塞住的人是被誰塞住的
需要 data/fg_positions.parquet（fetch_fg_positions.py）；fg_batting 若已有 position 欄就直接用。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from misplaced_profile import standardize_within_level, prob_superiority, assign_groups, residual_table, logistic_irls
from club_features import add_position, add_club_strength, load_mlb_positions, club_position_table
from war_decompose import six_season_components, COMPONENTS, RUNS_PER_WIN
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]
NEED_VARS = [
    ("母隊該守位 WAR（當季）", "pos_war_t"),
    ("母隊該守位 WAR（隔季，扣掉自己）", "pos_war_t1"),
    ("守位主力 WAR（當季）", "top_war_t"),
    ("守位主力年齡（當季）", "top_age_t"),
    ("母隊整體 WAR（當季）", "club_war_t"),
    ("母隊整體 WAR（隔季）", "club_war_t1"),
    ("農場同守位分數比他高的人數", "queue_ahead"),
    ("農場同守位層級比他高的人數", "queue_above"),
    ("農場同守位總人數", "queue_total"),
]


def attach_need(df: pd.DataFrame, fg: pd.DataFrame, season: int) -> pd.DataFrame:
    """把母隊狀況接到每個人季。"""
    cp = club_position_table(fg)
    df = df.copy()
    df["club"] = df["org"].fillna("").astype(str).str.split("/").str[0].replace({"ATH": "OAK"})
    k_t = pd.DataFrame({"Team": df["club"], "Season": season, "posg": df["pos_group"]})
    k_t1 = pd.DataFrame({"Team": df["club"], "Season": season + 1, "posg": df["pos_group"]})
    a = k_t.merge(cp, on=["Team", "Season", "posg"], how="left")
    b = k_t1.merge(cp, on=["Team", "Season", "posg"], how="left")
    df["pos_war_t"] = a["pos_war"].to_numpy()
    df["top_war_t"] = a["top_war"].to_numpy()
    df["top_age_t"] = a["top_age"].to_numpy()
    own = fg[fg["Season"] == season + 1][["key_mlbam", "Team", "posg", "WAR"]].rename(columns={"WAR": "own_war_t1"})
    own = own.merge(
        pd.DataFrame({"key_mlbam": df["key_mlbam"], "Team": df["club"], "posg": df["pos_group"]}),
        on=["key_mlbam", "Team", "posg"],
        how="inner",
    )
    own_map = own.drop_duplicates("key_mlbam").set_index("key_mlbam")["own_war_t1"]
    df["pos_war_t1"] = b["pos_war"].to_numpy() - df["key_mlbam"].map(own_map).fillna(0.0).to_numpy()
    # 農場內的塞車：同球團、同守位群、同季
    lvl = (
        df["level_rank_main"]
        if "level_rank_main" in df
        else df["level_main"].map({"DSL": 0, "R/CPX": 1, "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6})
    )
    df["_lvl"] = lvl.to_numpy()
    ahead, above, total = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for _, idx in df.groupby(["club", "pos_group"]).groups.items():
        s = df.loc[idx, "score_promoted"].to_numpy()
        l = df.loc[idx, "_lvl"].to_numpy()
        for j, i in enumerate(idx):
            ahead[df.index.get_loc(i)] = (s > s[j]).sum()
            above[df.index.get_loc(i)] = (l > l[j]).sum()
            total[df.index.get_loc(i)] = len(idx) - 1
    df["queue_ahead"] = ahead
    df["queue_above"] = above
    df["queue_total"] = total
    return df.drop(columns="_lvl")


def beyond_playing_time(reached: pd.DataFrame, z: pd.DataFrame, names: list) -> pd.DataFrame:
    """登上的人：幾年後登上、登上那年打席、每季平均打席、六季 WAR 率 ~ 期望 WAR + 登上機率 + 候選。係數 = 候選高一個標準差多多少。"""
    rows = []
    for disp, col in names:
        row = {"候選": disp}
        for ydisp, ycol in [
            ("幾年後登上", "years_to_debut"),
            ("登上那年打席", "first_season_pa"),
            ("每季平均打席", "pa_per_season"),
            ("每 600 打席 WAR（分）", "war_rate"),
        ]:
            cols = pd.DataFrame(
                {"a": reached["exp_war"], "b": reached["score_preach"], "x": z[col], "y": reached[ycol]}
            ).astype(float)
            ok = np.isfinite(cols).all(axis=1)  # 任一欄 NaN / inf 的人季不用（inf 會讓 SVD 不收斂）
            if ok.sum() < 30 or cols.loc[ok, "x"].std() == 0:
                row[ydisp] = np.nan
                continue
            X = np.column_stack([np.ones(ok.sum()), cols.loc[ok, "a"], cols.loc[ok, "b"], cols.loc[ok, "x"]])
            y = cols.loc[ok, "y"].to_numpy(dtype=float)
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ b
            se = np.sqrt(np.diag(r @ r / max(len(y) - 4, 1) * np.linalg.inv(X.T @ X)))
            row[ydisp] = f"{b[3]:+.2f}{'*' if abs(b[3] / se[3]) > 2 else ''}"
        row["n"] = int((z[col].notna() & reached["years_to_debut"].notna()).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def bar_test(df: pd.DataFrame, need: pd.Series, outcome: str, model_logit: pd.Series) -> dict:
    """結果 ~ 模型 logit + 缺口 + 缺口 × 模型 logit。交互作用為負 = 缺人時能力沒那麼重要（門檻降低）。"""
    ok = need.notna() & model_logit.notna() & df[outcome].notna()
    X = np.column_stack([np.ones(ok.sum()), model_logit[ok], need[ok], need[ok] * model_logit[ok]])
    b, se = logistic_irls(X, df.loc[ok, outcome].astype(float).to_numpy())
    return {
        "模型": f"{b[1]:+.2f} ({b[1] / se[1]:+.1f})",
        "缺口": f"{b[2]:+.2f} ({b[2] / se[2]:+.1f})",
        "缺口 × 模型": f"{b[3]:+.2f} ({b[3] / se[3]:+.1f})",
        "n": int(ok.sum()),
    }


def tertile(x: pd.Series, labels: list) -> pd.Series:
    return pd.qcut(x.rank(method="first"), 3, labels=labels)


def need_by_score_table(d: pd.DataFrame) -> pd.DataFrame:
    """缺 / 中 / 滿（母隊該守位 WAR 三等）× 模型分數三等：人數、登上率、登上者 WAR 中位、WAR ≥ 2 比例。"""
    d = d[d["pos_war_t"].notna()].copy()
    d["缺口"] = tertile(-d["pos_war_t"], ["滿（守位 WAR 高）", "中", "缺（守位 WAR 低）"])
    pct = d["exp_war"].groupby(d["level_main"]).rank(pct=True)  # 層級內的百分位
    d["模型"] = pd.cut(pct, [0, 1 / 3, 2 / 3, 1], labels=["低", "中", "高"], include_lowest=True)
    rows = []
    for (need, sc), g in d.groupby(["缺口", "模型"], observed=True):
        r = g[g["reached_fwd"].astype(bool)]
        rows.append(
            {
                "母隊該守位": need,
                "模型分數": sc,
                "人數": len(g),
                "登上率": round(g["reached_fwd"].mean(), 2),
                "登上者 WAR 中位": round(r["war_fwd"].median(), 1) if len(r) else np.nan,
                "WAR ≥ 2 比例（全體）": round((g["war_fwd"] >= 2).mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def group_need_table(df: pd.DataFrame, group: pd.Series, names: list) -> pd.DataFrame:
    """四群在母隊變數上的中位數，與 P(群 > 對照)。"""
    rows = []
    ctrl = df[group == "對照"]
    for disp, col in names:
        row = {"變數": disp}
        for gname in ["錯放成功", "升錯", "球團獨具眼光", "對照"]:
            g = df[group == gname]
            row[f"{gname} 中位"] = round(g[col].median(), 2)
            if gname != "對照":
                row[f"P({gname} > 對照)"] = round(prob_superiority(g[col], ctrl[col]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_position(m, data)
    m = add_club_strength(m, data)
    eta_p = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
    scores = eta_p[["key_mlbam", "score_promoted"] + [f"eta_{f}" for f in FACTORS]].merge(
        eta_w[["key_mlbam", "score_war", "score_war_asinh", "score_preach"]], on="key_mlbam", how="inner"
    )
    df = m.merge(scores, on="key_mlbam", how="inner")
    df["exp_war"] = df["score_preach"] * df["score_war"]
    df = df[(df["PA"] >= 100) & (df["pos_group"] != "無資料")].reset_index(drop=True)
    fgpos = load_mlb_positions(data)
    df = attach_need(df, fgpos, season)
    fg = pd.read_parquet(
        data / "fg_batting.parquet",
        columns=["key_mlbam", "Season", "PA", "G", "WAR", "Bats"] + [c for _, c in COMPONENTS],
    )
    comp = six_season_components(fg, season)
    df = df.merge(
        comp[["key_mlbam", "years_to_debut", "first_season_pa", "pa_per_season", "PA_mlb"]], on="key_mlbam", how="left"
    )
    df["war_rate"] = (df["war_fwd"] * RUNS_PER_WIN / (df["PA_mlb"].where(df["PA_mlb"] >= 100) / 600.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    z = standardize_within_level(df, [c for _, c in NEED_VARS])
    cover = pd.DataFrame(
        {
            "變數": [d for d, _ in NEED_VARS],
            "有值的人季": [int(df[c].notna().sum()) for _, c in NEED_VARS],
            "中位": [round(df[c].median(), 2) for _, c in NEED_VARS],
        }
    )
    rt = residual_table(df, z, NEED_VARS)
    obs = df["war_observable"].astype(bool)
    reached = df[obs & df["reached_fwd"].astype(bool)].reset_index(drop=True)
    z_r = z.loc[obs & df["reached_fwd"].astype(bool)].reset_index(drop=True)
    pt = beyond_playing_time(reached, z_r, NEED_VARS)
    # 2. 門檻會不會降低
    need = -z["pos_war_t"]
    pr = df["score_preach"].clip(1e-4, 1 - 1e-4)
    reach_logit = np.log(pr / (1 - pr))
    bar = pd.DataFrame(
        [
            {"結果": "隔年升不升", **bar_test(df, need, "promoted", df["score_promoted"])},
            {
                "結果": "六季內登上",
                **bar_test(
                    df[obs].reset_index(drop=True),
                    need[obs].reset_index(drop=True),
                    "reached_fwd",
                    reach_logit[obs].reset_index(drop=True),
                ),
            },
        ]
    )
    bar_club = pd.DataFrame(
        [
            {"結果": "隔年升不升", **bar_test(df, -z["club_war_t"], "promoted", df["score_promoted"])},
            {
                "結果": "六季內登上",
                **bar_test(
                    df[obs].reset_index(drop=True),
                    -z["club_war_t"][obs].reset_index(drop=True),
                    "reached_fwd",
                    reach_logit[obs].reset_index(drop=True),
                ),
            },
        ]
    )
    nbs = need_by_score_table(df[obs])
    # 3. 守位缺口 vs 整體弱隊，一起放
    rows = []
    for name, ycol, ml, sub in [
        ("隔年升不升", "promoted", df["score_promoted"], pd.Series(True, index=df.index)),
        ("六季內登上", "reached_fwd", reach_logit, obs),
    ]:
        ok = (
            sub
            & z["pos_war_t"].notna()
            & z["club_war_t"].notna()
            & z["queue_ahead"].notna()
            & z["top_war_t"].notna()
            & ml.notna()
            & df[ycol].notna()
        )  # 模型分數只在結構遮罩內有值
        X = np.column_stack(
            [
                np.ones(ok.sum()),
                ml[ok],
                z.loc[ok, "pos_war_t"],
                z.loc[ok, "club_war_t"],
                z.loc[ok, "top_war_t"],
                z.loc[ok, "queue_ahead"],
            ]
        )
        b, se = logistic_irls(X, df.loc[ok, ycol].astype(float).to_numpy())
        rows.append(
            {
                "結果": name,
                "母隊該守位 WAR": f"{b[2]:+.2f} ({b[2] / se[2]:+.1f})",
                "母隊整體 WAR": f"{b[3]:+.2f} ({b[3] / se[3]:+.1f})",
                "守位主力 WAR": f"{b[4]:+.2f} ({b[4] / se[4]:+.1f})",
                "農場同守位排前面人數": f"{b[5]:+.2f} ({b[5] / se[5]:+.1f})",
                "n": int(ok.sum()),
            }
        )
    joint = pd.DataFrame(rows)
    # 4. 四群
    group = assign_groups(df, "exp_war")
    gt = group_need_table(df, group, NEED_VARS)
    # 被模型看好卻沒升的人：是被誰塞住？
    md = [
        f"# {season} 個人層次：誰被塞住、誰被拉上去，跟母隊的狀況有關嗎 · {tag_p} / {tag_w}\n",
        "母隊 = 人季所屬球團（多球團取第一個）。守位 = 小聯盟 positionDB 的守位群；母隊該守位 = 大聯盟主守位同群的人加總。打席 ≥ 100、有守位資料。\n",
        "## 變數覆蓋\n",
        cover.to_markdown(index=False),
        "",
        "## 1. 模型分數之外，母隊的狀況還說明了什麼（beyond-the-model，係數 = 高一個標準差多多少；* = |z| > 2）\n",
        "「缺什麼補什麼」若是真的：母隊該守位 WAR 對升級 / 登上是負號（守位越滿越不升）；守位主力 WAR 負號；農場排前面人數負號。\n",
        rt.to_markdown(index=False),
        "",
        "### 登上的人：母隊狀況對上場機會與產出\n",
        pt.to_markdown(index=False),
        "",
        "## 2. 缺人會不會把門檻降低\n",
        "結果 ~ 模型 + 缺口 + 缺口 × 模型（缺口 = 母隊該守位 WAR 取負、層級內標準化）。缺口為正 = 缺人就多升；交互作用為負 = 缺人時模型分數沒那麼重要（門檻降低）。括號是 z。\n",
        bar.to_markdown(index=False),
        "",
        "同一個檢定，缺口改成「母隊整體弱」：\n",
        bar_club.to_markdown(index=False),
        "",
        "### 缺 / 中 / 滿 × 模型分數三等（六季窗口可觀察的人）\n",
        "看登上率有沒有隨缺口變；登上者 WAR 中位看被補上去的人有沒有比較差。\n",
        nbs.to_markdown(index=False),
        "",
        "## 3. 守位缺口 vs 整體弱隊 vs 卡位 vs 農場塞車：一起放，誰留下\n",
        joint.to_markdown(index=False),
        "",
        "## 4. 四群在母隊變數上的分布（中位數與 P(群 > 對照)；0.5 = 沒差）\n",
        "錯放成功（模型看好、球隊沒升、後來成功）若中位守位主力 WAR 高、排前面人數多 → 他們是被卡住的。球團獨具眼光（模型不看好、球隊升了、成功）若守位 WAR 低 → 是缺人補上去的。\n",
        gt.to_markdown(index=False),
        "",
    ]
    name = f"club_need_{tag_p}_vs_{tag_w}.md"
    (out / str(season) / name).write_text("\n".join(md))
    return {
        "1 模型之外母隊狀況": (rt, "z"),
        "1b 登上者：上場機會與產出": (pt, "z"),
        "2 缺人降門檻（守位缺口）": (bar, "z"),
        "2 缺人降門檻（整體弱隊）": (bar_club, "z"),
        "2b 缺／中／滿 × 分數三等": (nbs, None),
        "3 四個一起放": (joint, "z"),
        "4 四群的母隊變數": (gt, None),
    }
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args("個人層次：誰被塞住、誰被拉上去，跟母隊的狀況有沒有關？「缺什麼補什麼」是真的嗎？")
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
