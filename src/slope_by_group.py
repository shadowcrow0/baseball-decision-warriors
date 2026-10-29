"""強隊、弱隊看能力的方式一不一樣？把球員依母隊強弱、農場強弱分組，各組各自估一次係數，比斜率（不取樣）。

    python src/slope_by_group.py data/ <season> outputs/ --promoted <tag> --war <tag>

三個結果各一張表：
    升級        球團隔年升不升 ~ 四種能力 + 球探評價 + 非選秀 + 層級（邏輯迴歸，IRLS 自己寫，不靠套件）
    登上        六季內登上與否 ~ 同上（邏輯迴歸）
    WAR|登上    登上者的 asinh(六季 WAR) ~ 同上（最小平方）
每張表：全部、母隊強、母隊弱、農場強、農場弱 五欄的係數（括號是標準誤）。
係數在強弱之間差超過兩倍標準誤的，標 ◆。這是「模型對所有球團用同一組係數」這個假設的檢查。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from misplaced_profile import add_club_strength
from fit_year import add_extra

FACTORS = ["contact", "power", "speed", "discipline"]
X_NAMES = ["打得到球", "打得遠", "跑得快", "選得到球", "球探評價", "非選秀"]


def design(d: pd.DataFrame) -> tuple[np.ndarray, list]:
    """設計矩陣：常數、四種能力、球探評價（非選秀補 0）、非選秀、層級 dummies（基準 = 最多人的層級）。"""
    cols = [np.ones(len(d))] + [d[f"eta_{f}"].to_numpy(dtype=float) for f in FACTORS]
    cols.append(d["zeta_scout"].fillna(0.0).to_numpy(dtype=float))
    cols.append(d["undrafted"].to_numpy(dtype=float))
    names = ["常數"] + X_NAMES
    base = d["level_main"].mode().iloc[0]
    for lvl in sorted(d["level_main"].unique()):
        if lvl != base:
            cols.append((d["level_main"] == lvl).to_numpy(dtype=float))
            names.append(f"層級 {lvl}")
    return np.column_stack(cols), names


def logistic_irls(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """邏輯迴歸，IRLS。回傳 (係數, 標準誤)。ridge 很小，只是防止完全分離。"""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * b
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    w = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    s2 = resid @ resid / max(len(y) - X.shape[1], 1)
    cov = s2 * np.linalg.inv(X.T @ X + 1e-9 * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def fit_by_group(d: pd.DataFrame, y: np.ndarray, groups: dict, kind: str) -> pd.DataFrame:
    """groups：{欄名: 布林遮罩}。每組各估一次，只報四種能力與球探評價、非選秀。"""
    out = {}
    for gname, mask in groups.items():
        X, names = design(d[mask])
        yy = y[mask]
        if kind == "logit" and (yy.sum() < 10 or (1 - yy).sum() < 10):
            continue
        b, se = logistic_irls(X, yy) if kind == "logit" else ols(X, yy)
        out[gname] = {n: (b[i], se[i]) for i, n in enumerate(names) if n in X_NAMES}
        out[gname]["n"] = (int(mask.sum()), 0)
    return out


def table(fits: dict, pairs: list) -> pd.DataFrame:
    rows = []
    for name in X_NAMES + ["n"]:
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


def main(data_dir, season, out_dir, tag_p, tag_w):
    data = Path(data_dir)
    out = Path(out_dir)
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_club_strength(m, data)
    eta_p = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
    sc = eta_p[["key_mlbam"] + [f"eta_{f}" for f in FACTORS] + (["zeta_scout"] if "zeta_scout" in eta_p else [])].merge(
        eta_w[["key_mlbam", "score_war", "score_preach"]], on="key_mlbam", how="inner"
    )
    d = m.merge(sc, on="key_mlbam", how="inner")
    d["undrafted"] = d["pick_overall"].isna().astype(float)
    if "zeta_scout" not in d:
        d["zeta_scout"] = 0.0
    d = d[d["PA"] >= 100].reset_index(drop=True)
    d["exp_war"] = d["score_preach"] * d["score_war"]
    org = d["org"].fillna("").astype(str).str.split("/").str[0]
    farm = d.groupby(org)["exp_war"].mean()
    club = d.groupby(org)["club_war_t"].first()
    strong_club = org.map(club > club.median()).fillna(False).to_numpy(dtype=bool)
    strong_farm = org.map(farm > farm.median()).fillna(False).to_numpy(dtype=bool)
    groups = {
        "全部": np.ones(len(d), dtype=bool),
        "母隊強": strong_club,
        "母隊弱": ~strong_club,
        "農場強": strong_farm,
        "農場弱": ~strong_farm,
    }
    pairs = [("母隊強", "母隊弱"), ("農場強", "農場弱")]
    md = [
        f"# {season} 強隊弱隊的係數一不一樣 · {tag_p} / {tag_w}\n",
        "球員依母隊強弱（當季大聯盟團隊 WAR 中位數切）與農場強弱（該球團球員期望 WAR 平均中位數切）分組，各組各自估係數。括號 = 標準誤；差超過兩倍標準誤標 ◆。打席 ≥ 100。\n",
    ]
    y = d["promoted"].astype(float).to_numpy()
    md += [
        "## 升級：球團隔年升不升（邏輯迴歸）\n",
        table(fit_by_group(d, y, groups, "logit"), pairs).to_markdown(index=False),
        "",
    ]
    obs = d["war_observable"].astype(bool).to_numpy()
    dd = d[obs].reset_index(drop=True)
    g2 = {k: v[obs] for k, v in groups.items()}
    md += [
        "## 登上：六季內登上與否（邏輯迴歸）\n",
        table(fit_by_group(dd, dd["reached_fwd"].astype(float).to_numpy(), g2, "logit"), pairs).to_markdown(
            index=False
        ),
        "",
    ]
    r = obs & d["reached_fwd"].astype(bool).to_numpy()
    dr = d[r].reset_index(drop=True)
    g3 = {k: v[r] for k, v in groups.items()}
    md += [
        "## WAR|登上：登上者的 asinh(六季 WAR)（最小平方）\n",
        table(fit_by_group(dr, np.arcsinh(dr["war_fwd"].to_numpy(dtype=float)), g3, "ols"), pairs).to_markdown(
            index=False
        ),
        "",
    ]
    name = f"slope_by_group_{tag_p}_vs_{tag_w}.md"
    (out / str(season) / name).write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args(
        "強隊、弱隊看能力的方式一不一樣？把球員依母隊強弱、農場強弱分組，各組各自估一次係數，比斜率（不取樣）。"
    )
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war)
