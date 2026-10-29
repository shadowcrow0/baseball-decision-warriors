"""兩段式：先在大聯盟學「這些進階數字值多少分」，再把學到的權重套回 3A，看球隊升人有沒有照它。

    python src/learn_weights.py outputs/ data/ --promoted promoted_f4sed_linear_p2 [--level AAA]
    python src/learn_weights.py --selftest

A 段（大聯盟 2015 起，FanGraphs 季彙總；data/fg_batting.parquet 要有 Statcast 欄，share/ 那份只有 48 欄不夠）：
  樣本 = 打席 ≥ 200 的人季，而且隔季打席 ≥ 100。
  自變數 = 九個在 3A 逐球也算得出來、定義一樣的量（平均初速、強勁擊球率、barrel 率、平均仰角、追打率、區內接觸率、拉打率、滾地率、內野高飛 ÷ 進場球），季內標準化。
  因變數 = 隔季的打擊跑分 ÷ 打席 × 600（FanGraphs 的 Batting 欄；不用 WAR，WAR 有四成是上場機會、那是球隊的決定）。
  模型 = 季固定效應 + ridge 線性；留一季驗證（估其他季、測那一季），比三個模型：只用當季 wOBA、只用九個進階量、兩個都用。
  產出 = 一組權重 w（每個量高一個標準差，隔季每 600 打席多幾分）、逐季權重的符號穩不穩。
B 段（3A 2023 起，Savant 逐球；跟 hit_profile_statcast 同一套彙整與基底）：
  S = Σ w_j × z_j（z 是 3A 季內標準化的同一個量）= 「大聯盟說這個人的進階數字值多少分」，再季內標準化。
  問：球隊升不升給 S 幾分；S 對到目前為止登上、登上者 fWAR 幾分；S 放進去之後，每個原始量還剩多少作用（= 球隊看重但大聯盟的帳不給的部分）。
輸出 outputs/learn_weights_<promoted tag>.md、outputs/statcast_weights.csv。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dist_profile import pooled_coef
from bf import bf_str, log10_bf10
from hit_profile import BASE, coef_se
from hit_profile_statcast import LABELS as SC_LABELS
from hit_profile_statcast import load

# 3A 逐球變數名 → (FanGraphs 季彙總欄名, 中文)。PitchInfo 版（pfx 前綴）的好球帶比較接近 Statcast，有就優先用。
FEATURES = {
    "ev_mean": ("EV", "平均初速"),
    "hard_bip": ("HardHit%", "強勁擊球率"),
    "barrel_bip": ("Barrel%", "barrel 率"),
    "la_mean": ("LA", "平均仰角"),
    "chase": ("O-Swing%", "追打壞球率"),
    "zcon": ("Z-Contact%", "區內接觸率"),
    "pull_bip": ("Pull%", "拉打率"),
    "gb_bip": ("GB%", "滾地率"),
    "popup_bip": ("IFFB_BIP", "內野高飛 ÷ 進場球"),  # FanGraphs 的 IFFB% 分母是飛球，乘 FB% 才是 ÷ 進場球
}
MIN_PA, MIN_PA_NEXT, FIRST_SEASON = 200, 100, 2015
RIDGE = 1.0


def load_mlb(data: Path) -> pd.DataFrame:
    """FanGraphs 大聯盟人季 + 隔季打擊跑分；Statcast 欄不在就報錯告訴怎麼補。"""
    fg = pd.read_parquet(data / "fg_batting.parquet")
    for pfx in ("pfxO-Swing%", "pfxZ-Contact%"):
        if pfx in fg.columns:
            fg[pfx[3:]] = fg[pfx].where(fg[pfx].notna(), fg.get(pfx[3:]))
    if "IFFB%" in fg.columns and "FB%" in fg.columns:
        fg["IFFB_BIP"] = fg["IFFB%"] * fg["FB%"]
    need = [c for _, (c, _) in FEATURES.items()] + ["Batting", "wOBA", "PA", "Season", "key_mlbam"]
    missing = [c for c in need if c not in fg.columns]
    if missing:
        raise SystemExit(f"fg_batting.parquet 缺欄 {missing}：用 src/fetch_war.py 重抓（新 API 一次回全部欄位）")
    fg = fg.dropna(subset=["key_mlbam"]).astype({"key_mlbam": "int64", "Season": "int64"})
    fg = fg[fg["Season"] >= FIRST_SEASON]
    nxt = fg[fg["PA"] >= MIN_PA_NEXT][["key_mlbam", "Season", "Batting", "PA"]].copy()
    nxt["Season"] = nxt["Season"] - 1
    nxt["y_next"] = nxt["Batting"] / nxt["PA"] * 600
    d = fg[fg["PA"] >= MIN_PA].merge(nxt[["key_mlbam", "Season", "y_next"]], on=["key_mlbam", "Season"], how="inner")
    cols = [c for _, (c, _) in FEATURES.items()]
    d = d.dropna(subset=cols + ["y_next", "wOBA"]).copy()
    for c in cols + ["wOBA"]:
        d[c] = d[c].astype(float)
        d[c] = d.groupby("Season")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return d


def design(d: pd.DataFrame, feats: list, seasons: list) -> np.ndarray:
    """季虛擬變數（全部，沒有截距）+ 特徵。"""
    dummies = np.column_stack([(d["Season"] == s).to_numpy(dtype=float) for s in seasons])
    if not feats:
        return dummies
    return np.column_stack([dummies] + [d[f].to_numpy(dtype=float) for f in feats])


def fit(d: pd.DataFrame, feats: list, seasons: list) -> np.ndarray:
    X = design(d, feats, seasons)
    y = d["y_next"].to_numpy(dtype=float)
    pen = np.r_[np.zeros(len(seasons)), np.full(len(feats), RIDGE)]
    return np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)


def loo_r2(d: pd.DataFrame, feats: list) -> float:
    """留一季：估其他季、測那一季；測那季的季截距用訓練季截距的平均。回傳合併的樣本外 R²。"""
    seasons = sorted(d["Season"].unique())
    sse, sst = 0.0, 0.0
    for s in seasons:
        tr, te = d[d["Season"] != s], d[d["Season"] == s]
        tr_seasons = [t for t in seasons if t != s]
        b = fit(tr, feats, tr_seasons)
        pred = b[: len(tr_seasons)].mean() + (te[feats].to_numpy(dtype=float) @ b[len(tr_seasons):] if feats else 0.0)
        y = te["y_next"].to_numpy(dtype=float)
        sse += float(((y - pred) ** 2).sum())
        sst += float(((y - y.mean()) ** 2).sum())
    return 1 - sse / sst


def learn(d: pd.DataFrame) -> tuple:
    """A 段：權重表（合併、逐季符號）與三個模型的留一季 R²。"""
    cols = [c for _, (c, _) in FEATURES.items()]
    seasons = sorted(d["Season"].unique())
    b_all = fit(d, cols, seasons)[len(seasons):]
    signs = {c: [0, 0] for c in cols}
    for s in seasons:
        b = fit(d[d["Season"] == s], cols, [s])[1:]
        for c, v in zip(cols, b):
            signs[c][0 if v > 0 else 1] += 1
    rows = [
        {
            "進階量": FEATURES[k][1],
            "FanGraphs 欄": c,
            "權重（每 600 打席的跑分）": round(float(v), 2),
            "幾季正 / 幾季負": f"{signs[c][0]} / {signs[c][1]}",
            "_key": k,
        }
        for (k, (c, _)), v in zip(FEATURES.items(), b_all)
    ]
    weights = pd.DataFrame(rows)
    r2 = pd.DataFrame(
        [
            {"模型": "只有當季 wOBA", "留一季 R²": round(loo_r2(d, ["wOBA"]), 3)},
            {"模型": "只有九個進階量", "留一季 R²": round(loo_r2(d, cols), 3)},
            {"模型": "wOBA + 九個進階量", "留一季 R²": round(loo_r2(d, ["wOBA"] + cols), 3)},
        ]
    )
    return weights, r2


def apply_weights(d: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """B 段：S = Σ w_j z_j，用 3A 季內標準化後的同名變數；缺的量當 0（= 層級平均）；最後再季內標準化。"""
    s = pd.Series(0.0, index=d.index)
    for _, r in weights.iterrows():
        if r["_key"] in d.columns:
            s = s + r["權重（每 600 打席的跑分）"] * d[r["_key"]].fillna(0.0)
    return d.assign(_s=s).groupby("season")["_s"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))


def residual_table(d: pd.DataFrame, names: list, outcomes: list) -> pd.DataFrame:
    """S 放進基底之後，每個原始量還剩多少作用。"""
    rows = []
    base = BASE + [c for c in d.columns if c.startswith("lvl_")] + ["S"]
    for v in names:
        row = {"原始量": SC_LABELS.get(v, v)}
        for outcome, label in outcomes:
            bs, ses = [], []
            for s in sorted(d["season"].unique()):
                b, se = coef_se(d[d["season"] == s], outcome, base + [v])
                if np.isfinite(b) and np.isfinite(se) and se > 0:
                    bs.append(b)
                    ses.append(se)
            if bs:
                w = 1.0 / np.array(ses) ** 2
                pooled = float((np.array(bs) * w).sum() / w.sum())
                row[label] = f"{pooled:+.2f} [BF {bf_str(log10_bf10(pooled, float(np.sqrt(1 / w.sum()))))}]"
        rows.append(row)
    return pd.DataFrame(rows)


def main(out_dir: str, data_dir: str, tag_p: str, level: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    mlb = load_mlb(data)
    weights, r2 = learn(mlb)
    weights.drop(columns="_key").to_csv(out / "statcast_weights.csv", index=False)
    md = [
        f"# 先在大聯盟學進階數字值多少分，再看 3A 球隊有沒有照它升人 · {tag_p}\n",
        f"## A 段：大聯盟 {int(mlb['Season'].min())}–{int(mlb['Season'].max())}，{len(mlb):,} 人季（打席 ≥ {MIN_PA}、隔季 ≥ {MIN_PA_NEXT}）\n",
        "因變數 = 隔季打擊跑分 ÷ 打席 × 600。權重 = 這個量（季內標準化）高一個標準差，隔季每 600 打席多幾分；季固定效應；ridge。\n",
        weights.drop(columns="_key").to_markdown(index=False),
        "",
        "留一季樣本外（估其他季、測那一季）：\n",
        r2.to_markdown(index=False),
        "",
    ]
    d, names = load(out, data, tag_p, level)
    d["S"] = apply_weights(d, weights)
    outcomes = [
        ("y_prom", "隔年升上大聯盟"),
        ("y_keep", "被留下來"),
        ("y_reach", "到目前為止登上"),
        ("y_war", "登上者 fWAR"),
    ]
    outcomes = [(o, l) for o, l in outcomes if o in d.columns and d[o].notna().sum() >= 50 and d[o].nunique() >= 2]
    seasons = ", ".join(str(int(s)) for s in sorted(d["season"].unique()))
    md += [
        f"## B 段：3A {seasons}，{len(d):,} 人季（還沒登上的人，進場球 ≥ 50）\n",
        "S = 大聯盟學到的權重 × 這個人 3A 的同一組量（季內標準化），再標準化。係數 = 四把尺、球探、年齡、打席一樣，S 高一個標準差，log-odds 差多少（WAR 是 asinh 線性）；BF10 是 Savage–Dickey（先驗 N(0, 0.5²)）。\n",
    ]
    rows = []
    for outcome, label in outcomes:
        b, z, pos, neg = pooled_coef(d, outcome, "S")
        rows.append({"結果": label, "S 的係數": round(b, 3), "BF10": bf_str(log10_bf10(b, abs(b / z) if z else np.nan)), "幾季正 / 幾季負": f"{pos} / {neg}"})
    md += [pd.DataFrame(rows).to_markdown(index=False), ""]
    used = [k for k in FEATURES if k in names]
    md += ["### S 放進去之後，每個原始量還剩多少（= 球隊看重但大聯盟的帳不給、或帳給但球隊沒看）\n"]
    md += [residual_table(d, used, outcomes).to_markdown(index=False), ""]
    text = "\n".join(md)
    path = out / f"learn_weights_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成大聯盟：每人真實特徵固定、每季量到的帶噪音；跑分 = 2×barrel − 1×追打 + 0.5×初速 + 噪音。權重符號與大小要對；兩個都用的 R² 不能明顯輸只用 wOBA。"""
    rng = np.random.default_rng(5)
    root = Path(tempfile.mkdtemp())
    data = root / "data"
    data.mkdir()
    rows = []
    cols = [c for _, (c, _) in FEATURES.items() if c != "IFFB_BIP"]
    true = {i: {c: rng.normal() for c in cols} for i in range(400)}  # 每個人的真實特徵固定，每季量到的是真實 + 噪音
    for season in range(2015, 2021):
        for i in range(400):
            f = {c: true[i][c] + rng.normal(0, 0.3) for c in cols}
            f["IFFB%"], f["FB%"] = rng.uniform(0, 0.2), rng.uniform(0.2, 0.5)
            batting = 2 * true[i]["Barrel%"] - 1 * true[i]["O-Swing%"] + 0.5 * true[i]["EV"] + rng.normal(0, 1.5)
            rows.append({"key_mlbam": 1000 + i, "Season": season, "PA": 500, "wOBA": 0.32 + 0.01 * batting + rng.normal(0, 0.02), "Batting": batting * 500 / 600, **f})
    pd.DataFrame(rows).to_parquet(data / "fg_batting.parquet", index=False)
    d = load_mlb(data)
    weights, r2 = learn(d)
    w = weights.set_index("_key")["權重（每 600 打席的跑分）"]
    assert w["barrel_bip"] > 1.0 and w["chase"] < -0.4 and w["ev_mean"] > 0.1, f"權重不對：{w.to_dict()}"
    assert r2.iloc[2, 1] >= r2.iloc[0, 1] - 0.02, f"R² 不對：{r2}"
    fake = pd.DataFrame({"season": [2023] * 4 + [2024] * 4, "barrel_bip": [1, 0, -1, 0, 1, 0, -1, 0], "chase": [0, 1, 0, -1, 0, 1, 0, -1]})
    s = apply_weights(fake, weights)
    assert s.iloc[0] > s.iloc[2] and s.iloc[3] > s.iloc[1], "套權重的方向不對"
    print(weights.drop(columns="_key").to_string(index=False))
    print(r2.to_string(index=False))
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="大聯盟學進階數字的權重，套回 3A 看球隊升人")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("--promoted", default=None, help="升級模型的 tag，例如 promoted_f4sed_linear_p2")
    p.add_argument("--level", default="AAA", choices=["AAA", "A"])
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.promoted is None:
        p.error("--promoted 是必要的（或用 --selftest）")
    else:
        main(a.out_dir, a.data_dir, a.promoted, a.level)
