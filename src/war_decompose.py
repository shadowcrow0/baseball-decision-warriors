"""用排除法找被放掉的東西：把六季 WAR 拆成四塊，看模型（打擊成績單 + 球探評價）各解釋多少、剩下的是哪一塊。

    python src/war_decompose.py data/ <season> outputs/ --promoted <tag> --war <tag>

FanGraphs 的 WAR = (打擊 Batting + 跑壘 wBsR + 守備 Defense[含守位調整] + 替補基準 Replacement + 聯盟調整) ÷ 約 9.8 分/勝。
Replacement 跟打席數成正比，就是「上場機會」。六季加總後：
    打擊的那塊    成績單應該看得到 → 模型解釋不了的部分 = 打擊裡成績單沒有的（擊球品質之類）
    守備的那塊    成績單完全沒有   → 球探看得到、模型看不到
    上場機會那塊  一半是能力、一半是球隊給不給（順位、健康）
    跑壘的那塊    很小
對每一塊：佔六季 WAR 變異的比例、模型（期望 WAR、登上機率、四種能力、球探評價）的 R²、剩下的殘差跟候選欄位的相關。
輸出 outputs/<season>/war_decompose_<promoted tag>_vs_<war tag>.md。
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
    spearman,
    prob_superiority,
)
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]
COMPONENTS = [
    ("打擊", "Batting"),
    ("跑壘", "wBsR"),
    ("守備（含守位）", "Defense"),
    ("上場機會（替補基準）", "Replacement"),
]
RUNS_PER_WIN = 9.8
WINDOW = 6


def six_season_components(fg: pd.DataFrame, season: int) -> pd.DataFrame:
    """每個人在 season+1 到 season+6 的大聯盟成分加總（分），加 PA、G；沒出現的人之後補 0。"""
    w = fg[(fg["Season"] > season) & (fg["Season"] <= season + WINDOW)]
    cols = [c for _, c in COMPONENTS] + ["PA", "G", "WAR"]
    out = w.groupby("key_mlbam")[cols].sum().rename(columns={"PA": "PA_mlb", "G": "G_mlb"})  # 跟小聯盟的 PA 分開
    out["Bats_L"] = (w.groupby("key_mlbam")["Bats"].first() == "L").astype(float)
    out["Bats_B"] = (w.groupby("key_mlbam")["Bats"].first() == "B").astype(float)
    # 上場機會的零件：幾年後登上、待了幾季、每季平均打席、登上那年的打席
    played = w[w["PA"] > 0].sort_values(["key_mlbam", "Season"])
    g = played.groupby("key_mlbam")
    out["years_to_debut"] = g["Season"].min() - season
    out["n_seasons"] = g["Season"].nunique()
    out["pa_per_season"] = out["PA_mlb"] / out["n_seasons"].replace(0, np.nan)
    out["first_season_pa"] = g["PA"].first()
    out["gap_seasons"] = (g["Season"].max() - g["Season"].min() + 1) - out["n_seasons"]  # 中間消失的季數（下放或受傷）
    return out.reset_index()


def r_squared(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    """y 對 X（含常數項）的 OLS R² 與殘差。"""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    return 1.0 - resid.var() / y.var(), resid


def variance_shares(c: pd.DataFrame) -> pd.DataFrame:
    """每一塊佔六季 WAR（分）變異的比例：Var 分解含共變，所以用 Cov(塊, 總) / Var(總)，四塊加起來 = 1（扣掉聯盟調整的小尾巴）。"""
    total = c["WAR"] * RUNS_PER_WIN
    rows = []
    for disp, col in COMPONENTS:
        rows.append(
            {
                "成分": disp,
                "六季平均（分）": c[col].mean(),
                "佔 WAR 變異": np.cov(c[col], total)[0, 1] / total.var(),
                "與 WAR 的相關": np.corrcoef(c[col], total)[0, 1],
            }
        )
    return pd.DataFrame(rows).round(3)


def explained_table(c: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """每一塊被三組預測變數解釋的 R²：(a) 期望 WAR + 登上機率；(b) 四種能力 + 球探評價；(c) 兩組合併。回傳表與 (c) 的殘差。"""
    one = np.ones(len(c))
    Xa = np.column_stack([one, c["exp_war"], c["score_preach"]])
    eta_cols = [f"eta_{f}" for f in FACTORS] + (
        ["zeta_scout", "undrafted"] if "zeta_scout" in c else []
    )  # 非選秀的 ζ 補 0，另加 dummy
    Xb = np.column_stack([one] + [c[k] for k in eta_cols])
    Xc = np.column_stack([Xa, Xb[:, 1:]])
    rows, resid = [], {}
    for disp, col in COMPONENTS + [("六季 WAR（分）", "WAR_runs"), ("六季打席數", "PA_mlb")]:
        y = c[col].to_numpy(dtype=float)
        ra, _ = r_squared(y, Xa)
        rb, _ = r_squared(y, Xb)
        rc, res = r_squared(y, Xc)
        rows.append(
            {"成分": disp, "R²：模型的期望 WAR": ra, "R²：能力 + 球探評價": rb, "R²：合併": rc, "沒解釋的比例": 1 - rc}
        )
        resid[col] = res
    return pd.DataFrame(rows).round(3), resid


PT_PARTS = [
    ("幾年後登上", "years_to_debut"),
    ("待了幾季", "n_seasons"),
    ("每季平均打席", "pa_per_season"),
    ("登上那年的打席", "first_season_pa"),
    ("中間消失的季數", "gap_seasons"),
    ("六季打席數（log）", "log_pa_mlb"),
]


def playing_time_table(reached: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """上場機會再拆：每個零件被模型解釋的 R²。log 打席 = log(待了幾季) + log(每季平均打席)，兩塊的 R² 可以直接比。"""
    c = reached.copy()
    c["log_pa_mlb"] = np.log(c["PA_mlb"].clip(lower=1))
    one = np.ones(len(c))
    eta_cols = [f"eta_{f}" for f in FACTORS] + (["zeta_scout", "undrafted"] if "zeta_scout" in c else [])
    Xa = np.column_stack([one, c["exp_war"], c["score_preach"]])
    Xc = np.column_stack([Xa] + [c[k] for k in eta_cols])
    rows, resid = [], {}
    for disp, col in PT_PARTS:
        y = c[col].to_numpy(dtype=float)
        ok = np.isfinite(y)
        ra, _ = r_squared(y[ok], Xa[ok])
        rc, res = r_squared(y[ok], Xc[ok])
        full = np.full(len(c), np.nan)
        full[ok] = res
        rows.append(
            {"零件": disp, "平均": y[ok].mean(), "R²：模型的期望 WAR": ra, "R²：合併": rc, "沒解釋的比例": 1 - rc}
        )
        resid[col] = full
    return pd.DataFrame(rows).round(3), resid


RATE_PARTS = [
    ("每 600 打席的 WAR（分）", "war_rate"),
    ("每 600 打席的打擊分", "bat_rate"),
    ("每 600 打席的守備分", "def_rate"),
    ("六季打席數（log）", "log_pa_mlb"),
]


def rate_table(reached: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """把「每打席的產出」（球員的事）和「打席數」（球隊的事）分開：各自被模型解釋的 R²。"""
    c = reached.copy()
    c["log_pa_mlb"] = np.log(c["PA_mlb"].clip(lower=1))
    one = np.ones(len(c))
    eta_cols = [f"eta_{f}" for f in FACTORS] + (["zeta_scout", "undrafted"] if "zeta_scout" in c else [])
    Xc = np.column_stack([one, c["exp_war"], c["score_preach"]] + [c[k] for k in eta_cols])
    rows, resid = [], {}
    for disp, col in RATE_PARTS:
        y = c[col].to_numpy(dtype=float)
        ok = np.isfinite(y)
        rc, res = r_squared(y[ok], Xc[ok])
        full = np.full(len(c), np.nan)
        full[ok] = res
        rows.append({"量": disp, "人數": int(ok.sum()), "平均": y[ok].mean(), "R²：合併": rc, "沒解釋的比例": 1 - rc})
        resid[col] = full
    return pd.DataFrame(rows).round(3), resid


def cdf_by_avg(reached: pd.DataFrame, z: pd.DataFrame, resid_pt: dict) -> pd.DataFrame:
    """每個上場機會零件的殘差，依小聯盟打擊率（層級內三等）分三組，各給 25/50/75 分位。
    三組是整條平移（三個分位差距一樣）→ 同一個「機會傾向」在推；只有某一端動 → 打擊率只影響該環節的一部分。"""
    tier = (
        z["rate_AVG"]
        .groupby(reached["level_main"])
        .transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["低", "中", "高"]))
    )
    rows = []
    for disp, col in PT_PARTS:
        if col == "log_pa_mlb":
            continue
        r = pd.Series(resid_pt[col])
        for t in ["低", "中", "高"]:
            v = r[(tier == t).to_numpy()].dropna()
            rows.append(
                {
                    "零件": disp,
                    "打擊率": t,
                    "n": len(v),
                    "25%": v.quantile(0.25),
                    "50%": v.median(),
                    "75%": v.quantile(0.75),
                }
            )
    return pd.DataFrame(rows).round(2)


def stage_residual_corr(resid_pt: dict) -> pd.DataFrame:
    """五個零件殘差的兩兩相關：高 → 同一個隱藏的機會傾向；低 → 各環節各有各的隨機。"""
    d = pd.DataFrame({disp: resid_pt[col] for disp, col in PT_PARTS if col != "log_pa_mlb"})
    return d.corr().round(2)


def club_farm_grid(c: pd.DataFrame, reached: pd.DataFrame, resid_pt: dict, resid_rate: dict) -> pd.DataFrame:
    """母隊強弱 × 農場強弱的四格。母隊強度 = 當季大聯盟團隊 WAR；農場強度 = 該球團打席 ≥ 100 球員的期望 WAR 平均。
    各以球團中位數切兩半。每格：人數、登上比例、以及登上者在三個殘差上的平均（幾年後登上、登上那年的打席、每 600 打席的 WAR）。
    預期：母隊弱 + 農場弱 那格登上快、打席多、但每打席產出低 = 「機會不是因為你好，是因為沒人了」。"""
    org = c["org"].fillna("").astype(str).str.split("/").str[0]
    farm = c.groupby(org)["exp_war"].mean()
    club = c.groupby(org)["club_war_t"].first()
    strong_farm = farm > farm.median()
    strong_club = club > club.median()
    c = c.assign(
        _org=org.to_numpy(),
        農場=np.where(org.map(strong_farm).fillna(False), "農場強", "農場弱"),
        母隊=np.where(org.map(strong_club).fillna(False), "母隊強", "母隊弱"),
    )
    r = reached.assign(_org=reached["org"].fillna("").astype(str).str.split("/").str[0].to_numpy())
    r["r_debut"] = resid_pt["years_to_debut"]
    r["r_first_pa"] = resid_pt["first_season_pa"]
    r["r_war_rate"] = resid_rate["war_rate"]
    r["農場"] = np.where(r["_org"].map(strong_farm).fillna(False), "農場強", "農場弱")
    r["母隊"] = np.where(r["_org"].map(strong_club).fillna(False), "母隊強", "母隊弱")
    rows = []
    for (cl, fa), g in c.groupby(["母隊", "農場"]):
        gr = r[(r["母隊"] == cl) & (r["農場"] == fa)]
        rows.append(
            {
                "母隊": cl,
                "農場": fa,
                "球團數": g["_org"].nunique(),
                "人數": len(g),
                "登上比例": g["reached_fwd"].mean(),
                "幾年後登上 殘差中位": gr["r_debut"].median(),
                "登上那年打席 殘差中位": gr["r_first_pa"].median(),
                "登上那年打席 殘差>0 比例": (gr["r_first_pa"] > 0).mean(),
                "每600打席 WAR 殘差中位": gr["r_war_rate"].median(),
                "每600打席 WAR 殘差>0 比例": (gr["r_war_rate"] > 0).mean(),
                "P(每600打席 WAR 殘差 > 其他三格)": prob_superiority(
                    gr["r_war_rate"], r.loc[~((r["母隊"] == cl) & (r["農場"] == fa)), "r_war_rate"]
                ),
                "登上人數": len(gr),
            }
        )
    return pd.DataFrame(rows).round(2)


def residual_correlates(
    c: pd.DataFrame, z: pd.DataFrame, resid: dict, names: list, parts: list, top: int = 12
) -> pd.DataFrame:
    """每一塊的殘差跟候選欄位的相關，各列最強的 top 個；* = |r| > 2/√n。"""
    rows = []
    for disp, col in parts:
        r = resid[col]
        cors = []
        for cand_disp, cand in names:
            rr, nn = spearman(z[cand], pd.Series(r))
            if nn < 30 or not np.isfinite(rr):
                continue
            cors.append((cand_disp, rr, nn))
        cors.sort(key=lambda t: -abs(t[1]))
        for cand_disp, rr, nn in cors[:top]:
            rows.append(
                {
                    "成分": disp,
                    "候選": cand_disp,
                    "與殘差的相關": round(rr, 3),
                    "": "*" if abs(rr) > 2 / np.sqrt(nn) else "",
                    "n": nn,
                }
            )
    return pd.DataFrame(rows)


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_position(m, data)
    m = add_club_strength(m, data)
    m, names = build_candidates(m)
    fg = pd.read_parquet(
        data / "fg_batting.parquet",
        columns=["key_mlbam", "Season", "PA", "G", "WAR", "Bats"] + [c for _, c in COMPONENTS],
    )
    comp = six_season_components(fg, season)
    eta_p = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
    scores = eta_p[
        ["key_mlbam"] + [f"eta_{f}" for f in FACTORS] + (["zeta_scout"] if "zeta_scout" in eta_p else [])
    ].merge(eta_w[["key_mlbam", "score_war", "score_preach"]], on="key_mlbam", how="inner")
    df = m.merge(scores, on="key_mlbam", how="inner")
    df["exp_war"] = df["score_preach"] * df["score_war"]
    if "zeta_scout" in df:
        df["zeta_scout"] = df["zeta_scout"].fillna(0.0)
    df = df[(df["PA"] >= 100) & df["war_observable"].astype(bool) & df["exp_war"].notna()].reset_index(drop=True)
    names = names + [("左打", "Bats_L"), ("左右開弓", "Bats_B")]
    # 登上的人：成分加總；沒登上的人：全部 0（他們的 WAR 就是 0）
    c = df.merge(comp, on="key_mlbam", how="left")
    for col in [c2 for _, c2 in COMPONENTS] + [
        "PA_mlb",
        "G_mlb",
        "WAR",
        "Bats_L",
        "Bats_B",
        "n_seasons",
        "gap_seasons",
    ]:
        c[col] = c[col].fillna(0.0)
    c["WAR_runs"] = c["WAR"] * RUNS_PER_WIN
    pa600 = c["PA_mlb"].where(c["PA_mlb"] >= 100) / 600.0  # 比率只算打席 ≥ 100 的人，不然幾個打席就能把比率炸掉
    c["war_rate"] = c["WAR_runs"] / pa600  # 每 600 打席的 WAR（分）
    c["bat_rate"] = c["Batting"] / pa600  # 每 600 打席的打擊分
    c["def_rate"] = c["Defense"] / pa600
    reached = c[c["reached_fwd"].astype(bool)].reset_index(drop=True)
    z_all = standardize_within_level(c, [k for _, k in names])
    z_reached = z_all.loc[c["reached_fwd"].astype(bool)].reset_index(drop=True)
    shares = variance_shares(reached)
    expl_all, resid_all = explained_table(c)
    expl_r, resid_r = explained_table(reached)
    cor_r = residual_correlates(reached, z_reached, resid_r, names, COMPONENTS)
    pt, resid_pt = playing_time_table(reached)
    cor_pt = residual_correlates(
        reached, z_reached, resid_pt, names, [p for p in PT_PARTS if p[1] != "log_pa_mlb"], top=8
    )
    rt, resid_rate = rate_table(reached)
    cor_rate = residual_correlates(reached, z_reached, resid_rate, names, RATE_PARTS[:3], top=8)
    cdf = cdf_by_avg(reached, z_reached, resid_pt)
    scorr = stage_residual_corr(resid_pt)
    grid = club_farm_grid(c, reached, resid_pt, resid_rate)
    md = [
        f"# {season} 六季 WAR 拆解 · 升級 {tag_p} · WAR {tag_w}\n",
        f"打席 ≥ 100、六季窗口可觀察：{len(c)} 人，其中登上 {len(reached)} 人。成分單位是「分」，約 {RUNS_PER_WIN} 分 = 1 勝。\n",
        "## 一、登上的人：六季 WAR 由哪幾塊組成\n",
        "佔 WAR 變異 = Cov(成分, WAR) / Var(WAR)，四塊加起來約 1。這是「成功的人之間的差距來自哪裡」。\n",
        shares.to_markdown(index=False),
        "",
        "## 二、模型解釋了每一塊的多少（所有人，沒登上的人各塊為 0）\n",
        expl_all.to_markdown(index=False),
        "",
        "## 三、模型解釋了每一塊的多少（只看登上的人）\n",
        "排除法在這裡：成績單看得到的是打擊那塊，守備和上場機會是成績單沒有的。模型對打擊那塊的 R² 高、對守備那塊的 R² 低，就表示被放掉的在守備。\n",
        expl_r.to_markdown(index=False),
        "",
        "## 四、每一塊的殘差跟候選欄位的相關（登上的人；Spearman；* = |r| > 2/√n）\n",
        cor_r.to_markdown(index=False),
        "",
        "## 五、上場機會再拆（登上的人）\n",
        "六季打席數 = 幾年後登上 × 待了幾季 × 每季平均打席。幾年後登上和登上那年的打席是球隊的決定；每季平均打席是角色（先發或板凳）；中間消失的季數是下放或受傷。模型預測哪一個零件，就是在預測誰的行為。\n",
        pt.to_markdown(index=False),
        "",
        "## 六、上場機會零件的殘差跟候選欄位的相關（登上的人）\n",
        cor_pt.to_markdown(index=False),
        "",
        "## 七、每打席的產出 vs 打席數（登上且大聯盟打席 ≥ 100 的人）\n",
        "WAR = 每打席比替補好多少 × 打席數。前者是球員的事、後者是球隊的事。模型對哪個有解釋力，打擊率的殘差跟哪個有關，就把「門票」和「產出」分開了。\n",
        rt.to_markdown(index=False),
        "",
        cor_rate.to_markdown(index=False),
        "",
        "## 八、每個零件的殘差分布，依小聯盟打擊率分三組（CDF 的三個分位）\n",
        "三組的 25/50/75 分位差距一樣 = 整條曲線平移，同一個機會傾向在推；只有一端動 = 打擊率只影響該環節的一部分人。\n",
        cdf.to_markdown(index=False),
        "",
        "## 九、五個零件殘差的兩兩相關\n",
        "高 = 同一個隱藏的機會傾向推著五個環節（第二階段可以共用一個潛在變數）；低 = 各環節各有各的隨機。\n",
        scorr.to_markdown(),
        "",
        "## 十、母隊強弱 × 農場強弱（球團中位數切半）\n",
        "母隊強度 = 當季大聯盟團隊 WAR；農場強度 = 該球團打席 ≥ 100 球員的期望 WAR 平均。殘差皆已扣掉模型預期；用中位數與「殘差為正的比例」，不用平均。母隊弱 + 農場弱 那格若登上快、打席多、每打席產出低，就是「機會不是因為你好，是因為沒人了」。\n",
        grid.to_markdown(index=False),
        "",
    ]
    name = f"war_decompose_{tag_p}_vs_{tag_w}.md"
    (out / str(season) / name).write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args(
        "用排除法找被放掉的東西：把六季 WAR 拆成四塊，看模型（打擊成績單 + 球探評價）各解釋多少、剩下的是哪一塊。"
    )
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
