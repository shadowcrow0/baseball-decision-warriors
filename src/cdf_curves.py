"""連續量用 CDF 呈現：算出每條曲線的點，存成小 json（投影片與報告畫圖用；不畫圖、不取樣）。

    python src/cdf_curves.py data/ <season> outputs/ --promoted <tag> --war <tag>

曲線（每組一條階梯 CDF，x 從小到大，y = 累積比例）：
    debut_by_bat       幾年內登上大聯盟 × 打擊能力低中高（六年設限：曲線到 6 停）
    debut_by_avg       幾年內登上 × 小聯盟打擊率低中高，在「模型期望 WAR」相近的人裡比（能力相同、打擊率不同）
    debut_by_pos       幾年內登上 × 守位（捕手、游擊二壘、外野、一壘指定）
    debut_by_club      幾年內登上 × 母隊弱農場弱 vs 其他
    seasons_by_avg     登上後待了幾季 × 打擊率
    pa_by_avg          每季平均打席 × 打擊率
    war_by_pos         六季 WAR（登上者）× 守位
    debut_model_vs_obs 登上時間：模型預測的 CDF（log-normal 平均）vs 實際階梯，分打擊能力三組
輸出 outputs/<season>/cdf_curves.json。
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from misplaced_profile import build_candidates, add_position, add_club_strength
from war_decompose import six_season_components, WINDOW
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]


def ecdf_points(x: pd.Series, xmax: float = None, censored_at: float = None) -> list:
    """階梯 CDF 的點：[(x, F(x)), ...]。censored_at：超過的值視為「沒發生」，曲線在那裡停住，不假裝知道後面。"""
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return []
    if censored_at is not None:
        obs = np.sort(x[x < censored_at])
    else:
        obs = np.sort(x)
    pts = [(0.0, 0.0)] if obs.size and obs.min() > 0 else []
    for v in np.unique(obs):
        pts.append((float(v), float((obs <= v).sum() / n)))
    if censored_at is not None:
        pts.append((float(censored_at - 0.5), float(len(obs) / n)))  # 水平延到窗口末端
    return pts


def tercile(x: pd.Series, by: pd.Series) -> pd.Series:
    return x.groupby(by).transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["低", "中", "高"]))


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_position(m, data)
    m = add_club_strength(m, data)
    m, _ = build_candidates(m)
    eta_p = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
    sc = eta_p[["key_mlbam"] + [f"eta_{f}" for f in FACTORS]].merge(
        eta_w[["key_mlbam", "score_war", "score_preach", "score_debut_logtime"]], on="key_mlbam", how="inner"
    )
    df = m.merge(sc, on="key_mlbam", how="inner")
    df["exp_war"] = df["score_preach"] * df["score_war"]
    df = df[(df["PA"] >= 100) & df["war_observable"].astype(bool) & df["exp_war"].notna()].reset_index(drop=True)
    # 幾年後登上：登上者 = 首登年 − 本季（同季 = 0）；沒登上 = 設限
    T = (df["mlb_played_first"] - df["season"]).where(df["reached_fwd"].astype(bool), WINDOW + 1.0)  # 沒登上 = 超過窗口
    df["T"] = T
    df["bat"] = (df["eta_contact"] + df["eta_power"]) / 2
    df["bat_tier"] = tercile(df["bat"], df["level_main"])
    df["avg_tier"] = tercile(df["rate_AVG"], df["level_main"])
    df["exp_tier"] = tercile(df["exp_war"], df["level_main"])  # 模型期望相近的人裡比打擊率
    org = df["org"].fillna("").astype(str).str.split("/").str[0]
    farm = df.groupby(org)["exp_war"].mean()
    club = df.groupby(org)["club_war_t"].first()
    weak_both = (~org.map(farm > farm.median()).fillna(False)) & (~org.map(club > club.median()).fillna(False))
    df["club_group"] = np.where(weak_both, "母隊弱且農場弱", "其他")
    fg = pd.read_parquet(
        data / "fg_batting.parquet",
        columns=["key_mlbam", "Season", "PA", "G", "WAR", "Bats", "Batting", "wBsR", "Defense", "Replacement"],
    )
    comp = six_season_components(fg, season)
    df = df.merge(comp[["key_mlbam", "n_seasons", "pa_per_season"]], on="key_mlbam", how="left")
    reached = df[df["reached_fwd"].astype(bool)]

    curves = {}
    curves["debut_by_bat"] = {
        t: ecdf_points(df.loc[df["bat_tier"] == t, "T"], censored_at=WINDOW + 0.5) for t in ["低", "中", "高"]
    }
    mid = df[df["exp_tier"] == "中"]
    curves["debut_by_avg"] = {
        t: ecdf_points(mid.loc[mid["avg_tier"] == t, "T"], censored_at=WINDOW + 0.5) for t in ["低", "中", "高"]
    }
    curves["debut_by_pos"] = {
        p: ecdf_points(df.loc[df["pos_group"] == p, "T"], censored_at=WINDOW + 0.5)
        for p in ["捕手", "游擊/二壘", "外野", "一壘/指定"]
    }
    curves["debut_by_club"] = {
        g: ecdf_points(df.loc[df["club_group"] == g, "T"], censored_at=WINDOW + 0.5) for g in ["母隊弱且農場弱", "其他"]
    }
    curves["seasons_by_avg"] = {
        t: ecdf_points(reached.loc[reached["avg_tier"] == t, "n_seasons"]) for t in ["低", "中", "高"]
    }
    curves["pa_by_avg"] = {
        t: ecdf_points(reached.loc[reached["avg_tier"] == t, "pa_per_season"]) for t in ["低", "中", "高"]
    }
    curves["war_by_pos"] = {
        p: ecdf_points(reached.loc[reached["pos_group"] == p, "war_fwd"])
        for p in ["捕手", "游擊/二壘", "外野", "一壘/指定"]
    }
    # 模型 vs 實際：模型對每人給 log-normal(μ=score_debut_logtime, σ=sigma_t)；組的預測 CDF = 個人 CDF 的平均。σ_t 從 struct_M1.csv 讀
    st = pd.read_csv(out / str(season) / tag_w / "struct_M1.csv", index_col=0)
    sigma_t = float(st.loc["sigma_t", "mean"]) if "sigma_t" in st.index else None
    mv = {}
    for t in ["低", "中", "高"]:
        g = df[df["bat_tier"] == t]
        obs = ecdf_points(g["T"], censored_at=WINDOW + 0.5)
        pred = []
        if sigma_t is not None:
            for yr in range(0, WINDOW + 1):
                # 模型的 T 定義是 首登年 − 本季 + 1，所以 x 年內登上 = T ≤ x + 1
                pred.append((float(yr), float(norm.cdf((np.log(yr + 1) - g["score_debut_logtime"]) / sigma_t).mean())))
        mv[t] = {"observed": obs, "model": pred, "n": int(len(g))}
    curves["debut_model_vs_obs"] = mv
    counts = {
        k: {g: int(len(v)) for g, v in d.items()} if k != "debut_model_vs_obs" else {g: v["n"] for g, v in d.items()}
        for k, d in curves.items()
    }
    meta = {
        "season": season,
        "promoted": tag_p,
        "war": tag_w,
        "window": WINDOW,
        "sigma_t": sigma_t,
        "n_total": int(len(df)),
        "n_reached": int(len(reached)),
        "groups_n": {k: {g: int((df["bat_tier"] == g).sum()) for g in ["低", "中", "高"]} for k in ["bat_tier"]},
    }
    (out / str(season) / "cdf_curves.json").write_text(
        json.dumps({"meta": meta, "curves": curves}, ensure_ascii=False, indent=1)
    )
    for k, d in curves.items():
        if k == "debut_model_vs_obs":
            print(k, {g: (len(v["observed"]), len(v["model"])) for g, v in d.items()})
        else:
            print(k, {g: (len(v), round(v[-1][1], 2) if v else None) for g, v in d.items()})
    print(f"-> {out / str(season) / 'cdf_curves.json'}")


if __name__ == "__main__":
    args = read_args("連續量用 CDF 呈現：算出每條曲線的點，存成小 json（投影片與報告畫圖用；不畫圖、不取樣）。")
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
