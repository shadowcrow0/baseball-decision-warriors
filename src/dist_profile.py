"""不把一個人壓成平均和標準差：用他每一筆資料形成的分布（經驗 CDF 的九個十分位）看球隊在看分布的哪一段。

    python src/dist_profile.py outputs/ data/ --promoted promoted_f4sed_linear_p2 [--level AAA]
    python src/dist_profile.py --selftest

資料：Savant 3A 逐球（跟 hit_profile_statcast 一樣，data/ 沒有就用 share/）。每人季三個分布，各記九個十分位（10%、20%、…、90%）：
  ev   每一顆進場球的初速（打多強的分布：下尾是弱擊球、上尾是最強那幾顆）
  xw   每一顆進場球的接觸後預期 wOBA（把初速與仰角合起來的「這球值多少」的分布）
  gm   每一場比賽的 wOBA（打席 ≥ 3 的場次；穩不穩、會不會冷熱交替的分布）
另外留平均與標準差當對照組。
問三件事（結果：隔年升不升、被留下來、到目前為止登上沒、登上者 fWAR；基底 = 四把尺 + 球探 + 年齡 + 打席）：
  1. 分布的哪一段有作用：每個十分位單獨放進基底，看係數——球隊看的是上尾、下尾還是中間。
  2. 整個分布比平均加標準差多知道多少：基底 + 平均 + 標準差 vs 基底 + 九個十分位，前一季估、後一季測的 log-loss。
  3. 像誰：每個人的 CDF 到「前一季被升的人」合併 CDF 的 Wasserstein 距離，減到「沒被升的人」的距離；距離差當一個變數。
      這是 decision_style 裡 exemplar 學習者的分布版：球隊是不是在拿整個形狀比對。
輸出 outputs/dist_profile_<promoted tag>.md。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decision_style import fit_logit, log_loss, logit_predict
from hit_profile import BASE, coef_se, design
from bf import bf_str, log10_bf10
from hit_profile_statcast import HIT_EVENTS, load

QS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
DISTS = {"ev": "進場球初速", "xw": "接觸後預期 wOBA", "gm": "每場 wOBA"}
MIN_GAMES = 20  # 每場 wOBA 的分布至少要這麼多場


def labels_all() -> dict:
    """變數名 → 中文：三個分布各九個十分位、平均、標準差、像誰。"""
    out = {}
    for k, name in DISTS.items():
        for q in QS:
            out[f"{k}_q{q}"] = f"{name} 第 {q} 百分位"
        out[f"{k}_mean"] = f"{name} 平均"
        out[f"{k}_sd"] = f"{name} 標準差"
        out[f"{k}_like"] = f"{name} 像被升的人（Wasserstein 距離差）"
    return out


LABELS = labels_all()


def quantile_block(values: pd.Series, keys: pd.DataFrame, prefix: str, min_n: int) -> pd.DataFrame:
    """一組「每筆一列」的值 → 每個 key 一列的九個十分位、平均、標準差；筆數不到 min_n 留空。"""
    df = keys.copy()
    df["v"] = values.to_numpy(dtype=float)
    df = df.dropna(subset=["v"])
    g = df.groupby(list(keys.columns))["v"]
    out = pd.DataFrame({f"{prefix}_q{q}": g.quantile(q / 100) for q in QS})
    out[f"{prefix}_mean"] = g.mean()
    out[f"{prefix}_sd"] = g.std()
    n = g.size()
    out.loc[n < min_n, :] = np.nan
    return out.reset_index()


def per_player_dist(pitches: pd.DataFrame) -> pd.DataFrame:
    """逐球 → 每人季每層級一列：三個分布的十分位。pa、h、bip 欄給 load() 的門檻用。"""
    keys = ["batter", "game_year", "level"]
    pa = pitches[pitches["events"].notna()].copy()
    pa["is_bip"] = pa["bb_type"].notna()
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS)
    g = pa.groupby(keys)
    base = pd.DataFrame({"pa": g.size(), "h": g["is_hit"].sum(), "bip": g["is_bip"].sum()}).reset_index()
    bip = pa[pa["is_bip"]]
    ev = quantile_block(pd.to_numeric(bip["launch_speed"], errors="coerce"), bip[keys], "ev", 30)
    xw = quantile_block(
        pd.to_numeric(bip["estimated_woba_using_speedangle"], errors="coerce"), bip[keys], "xw", 30
    )
    # 每場 wOBA：woba_value 加總 ÷ woba_denom 加總，打席 ≥ 3 的場次
    pa["wv"] = pd.to_numeric(pa["woba_value"], errors="coerce")
    pa["wd"] = pd.to_numeric(pa["woba_denom"], errors="coerce")
    game = pa.groupby(keys + ["game_pk"]).agg(wv=("wv", "sum"), wd=("wd", "sum"), n=("wv", "size")).reset_index()
    game = game[(game["n"] >= 3) & (game["wd"] > 0)]
    game["woba"] = game["wv"] / game["wd"]
    gm = quantile_block(game["woba"], game[keys], "gm", MIN_GAMES)
    out = base.merge(ev, on=keys, how="left").merge(xw, on=keys, how="left").merge(gm, on=keys, how="left")
    return out.rename(columns={"batter": "key_mlbam", "game_year": "season", "level": "level_main"})


def pooled_coef(d: pd.DataFrame, outcome: str, var: str) -> tuple:
    """逐季係數的反變異數加權合併：(合併係數, z, 幾季正, 幾季負)。"""
    feats = BASE + [c for c in d.columns if c.startswith("lvl_")] + [var]
    bs, ses = [], []
    for s in sorted(d["season"].unique()):
        b, se = coef_se(d[d["season"] == s], outcome, feats)
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            bs.append(b)
            ses.append(se)
    if not bs:
        return np.nan, np.nan, 0, 0
    w = 1.0 / np.array(ses) ** 2
    pooled = float((np.array(bs) * w).sum() / w.sum())
    return pooled, pooled / float(np.sqrt(1.0 / w.sum())), sum(b > 0 for b in bs), sum(b < 0 for b in bs)


def which_segment(d: pd.DataFrame, outcomes: list) -> pd.DataFrame:
    """表一：每個分布的每個十分位，單獨放進基底的係數（每個結果一欄「係數 (z)」）。"""
    rows = []
    for k, name in DISTS.items():
        for q in QS:
            row = {"分布": name, "十分位": q}
            for outcome, label in outcomes:
                if d[outcome].notna().sum() < 50:
                    continue
                b, z, _, _ = pooled_coef(d, outcome, f"{k}_q{q}")
                row[label] = f"{b:+.2f} [BF {bf_str(log10_bf10(b, abs(b / z) if z else np.nan))}]" if np.isfinite(b) else "—"
            rows.append(row)
    return pd.DataFrame(rows)


def oos(d: pd.DataFrame, outcome: str, train: int, test: int, feats: list, ridge: float) -> float:
    """前一季估、後一季測；0/1 結果 log-loss（ridge 邏輯迴歸），WAR 均方誤差（ridge 線性）。"""
    tr = d[(d["season"] == train) & d[outcome].notna()]
    te = d[(d["season"] == test) & d[outcome].notna()]
    if len(tr) < 60 or len(te) < 10:
        return np.nan
    ya, yb = tr[outcome].to_numpy(dtype=float), te[outcome].to_numpy(dtype=float)
    if outcome == "y_war":
        Xa, Xb = design(tr, feats), design(te, feats)
        coef = np.linalg.solve(Xa.T @ Xa + ridge * np.eye(Xa.shape[1]), Xa.T @ ya)
        return float(((yb - Xb @ coef) ** 2).mean())
    if ya.sum() < 10 or (1 - ya).sum() < 10:
        return np.nan
    coef = fit_logit(tr[feats].to_numpy(dtype=float), ya, ridge=ridge)
    return log_loss(yb, logit_predict(coef, te[feats].to_numpy(dtype=float)))


def whole_vs_summary(d: pd.DataFrame, outcomes: list) -> pd.DataFrame:
    """表二：基底、基底 + 平均 + 標準差、基底 + 九個十分位，兩個方向（前季估後季測）的樣本外損失。"""
    base = BASE + [c for c in d.columns if c.startswith("lvl_")]
    seasons = sorted(d["season"].unique())
    folds = [(a, b) for a, b in zip(seasons[:-1], seasons[1:])] + [(b, a) for a, b in zip(seasons[:-1], seasons[1:])]
    rows = []
    for k, name in DISTS.items():
        summ = base + [f"{k}_mean", f"{k}_sd"]
        full = base + [f"{k}_q{q}" for q in QS]
        for outcome, label in outcomes:
            if d[outcome].notna().sum() < 100:
                continue
            for train, test in folds:
                l0 = oos(d, outcome, train, test, base, 1e-2)
                l1 = oos(d, outcome, train, test, summ, 1e-2)
                l2 = oos(d, outcome, train, test, full, 1.0)  # 九個相關的十分位，ridge 放大
                if not np.isfinite(l0):
                    continue
                rows.append(
                    {
                        "分布": name,
                        "結果": label,
                        "估 → 測": f"{train} → {test}",
                        "基底": round(l0, 4),
                        "加平均+標準差": round(l1, 4),
                        "加九個十分位": round(l2, 4),
                        "分布比平均多賺": round(l1 - l2, 4),
                    }
                )
    return pd.DataFrame(rows)


def wasserstein_like(d: pd.DataFrame, k: str, outcome: str) -> pd.Series:
    """每個人季：他的 CDF（九個十分位）到「其他季結果 = 0 的人」合併十分位的 W1 距離，減到「結果 = 1 的人」的距離。
    正 = 形狀比較像被升／登上的人。十分位的差的絕對值平均，就是兩個離散 CDF 的 W1 距離近似。"""
    cols = [f"{k}_q{q}" for q in QS]
    out = pd.Series(np.nan, index=d.index)
    for s in d["season"].unique():
        others = d[(d["season"] != s) & d[outcome].notna()]
        if len(others) < 60:
            continue
        ref1 = others.loc[others[outcome] == 1, cols].median().to_numpy()
        ref0 = others.loc[others[outcome] == 0, cols].median().to_numpy()
        here = d.loc[d["season"] == s, cols].to_numpy(dtype=float)
        d1 = np.abs(here - ref1).mean(axis=1)
        d0 = np.abs(here - ref0).mean(axis=1)
        out.loc[d["season"] == s] = d0 - d1
    return out


def main(out_dir: str, data_dir: str, tag_p: str, level: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    d, names = load(out, data, tag_p, level, profile_fn=per_player_dist, labels=LABELS)
    outcomes = [
        ("y_prom", "隔年升上大聯盟"),
        ("y_keep", "被留下來"),
        ("y_reach", "到目前為止登上"),
        ("y_war", "登上者 fWAR"),
    ]
    outcomes = [(o, l) for o, l in outcomes if o in d.columns and d[o].notna().sum() >= 50 and d[o].nunique() >= 2]
    seasons = ", ".join(str(int(s)) for s in sorted(d["season"].unique()))
    md = [
        f"# 不用平均用分布：球隊看的是一個人資料分布的哪一段 · {level} · {tag_p}\n",
        f"樣本：{len(d):,} 人季（{seasons}），還沒登上大聯盟的人，進場球 ≥ 50。每個人季三個分布各記九個十分位（進場球初速、接觸後預期 wOBA、每場 wOBA），"
        "季內標準化。係數 = 四把尺、球探、年齡、打席一樣的兩個人，那個十分位差一個標準差，log-odds 差多少（WAR 是 asinh 線性係數）；方括號是 BF10（Savage–Dickey，先驗 N(0, 0.5²)；> 10 強、< 1/3 支持零）。"
        "初速第 90 百分位 = 他最強那一成球有多強；第 10 百分位 = 最弱那一成。每場 wOBA 的第 10 百分位 = 冷的時候多冷。\n",
        "## 一、分布的哪一段有作用\n",
        which_segment(d, outcomes).to_markdown(index=False),
        "",
        "## 二、整個分布比「平均 + 標準差」多知道多少（前一季估、後一季測；0/1 是 log-loss，WAR 是均方誤差，越小越好）\n",
    ]
    cmp = whole_vs_summary(d, outcomes)
    md += [cmp.to_markdown(index=False) if not cmp.empty else "（季數不夠，算不出樣本外）", ""]
    md += ["## 三、像誰：CDF 到「其他季被升／登上的人」的距離差，當一個變數放進基底\n"]
    rows = []
    for k, name in DISTS.items():
        for outcome, label in outcomes:
            if outcome == "y_war":
                continue
            d[f"{k}_like"] = wasserstein_like(d, k, outcome)
            if d[f"{k}_like"].notna().sum() < 60:
                continue
            d[f"{k}_like"] = d.groupby("season")[f"{k}_like"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            b, z, pos, neg = pooled_coef(d, outcome, f"{k}_like")
            rows.append({"分布": name, "結果": label, "係數": round(b, 3), "BF10": bf_str(log10_bf10(b, abs(b / z) if z else np.nan)), "幾季正 / 幾季負": f"{pos} / {neg}"})
    md += [pd.DataFrame(rows).to_markdown(index=False) if rows else "（季數不夠）", ""]
    text = "\n".join(md)
    path = out / f"dist_profile_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成：升不升只看初速的上尾（第 90 百分位），平均一樣。檢查 q90 的係數要明顯大於 q10 的。"""
    rng = np.random.default_rng(3)
    root = Path(tempfile.mkdtemp())
    data, out = root / "data", root / "outputs"
    data.mkdir()
    etas = ["eta_" + f for f in ("contact", "power", "speed", "discipline")]
    rows, table = [], []
    for season in (2023, 2024, 2025):
        for i in range(300):
            key, n = 700000 + i, int(rng.integers(250, 450))
            tail = rng.uniform(0, 1)  # 上尾有多重：平均固定 88，重尾的人最強的球更強、最弱的更弱一點
            ev = np.where(rng.random(n) < 0.15, 88 + 12 * tail + rng.normal(0, 2, n), 88 - 2 * tail + rng.normal(0, 5, n))
            bb = rng.choice(["ground_ball", "fly_ball", "line_drive", None], n, p=[0.3, 0.3, 0.15, 0.25])
            events = np.where(bb == None, "strikeout", rng.choice(["field_out", "single", "home_run"], n, p=[0.7, 0.25, 0.05]))  # noqa: E711
            rows.append(
                pd.DataFrame(
                    {
                        "batter": key,
                        "game_year": season,
                        "level": "AAA",
                        "game_pk": rng.integers(1, 120, n),
                        "events": events,
                        "bb_type": bb,
                        "launch_speed": np.where(bb == None, np.nan, ev),  # noqa: E711
                        "estimated_woba_using_speedangle": np.where(bb == None, np.nan, rng.uniform(0, 1, n)),  # noqa: E711
                        "woba_value": np.where(events == "strikeout", 0, np.where(events == "single", 0.9, np.where(events == "home_run", 2.0, 0))),
                        "woba_denom": 1,
                    }
                )
            )
            prom = rng.random() < 1 / (1 + np.exp(-(-1.5 + 3 * tail)))
            table.append(
                {
                    "key_mlbam": key,
                    "season": season,
                    "level_main": "AAA",
                    "org": "NYY",
                    "PA": n,
                    "promoted": prom,
                    "next_status": "mlb" if prom else "same",
                    "mlb_played_first": np.nan,
                    "structural": True,
                    "age_rel": rng.normal(),
                    **{e: rng.normal() for e in etas},
                }
            )
    pitches = pd.concat(rows, ignore_index=True)
    m = pd.DataFrame(table)
    for season in (2023, 2024, 2025):
        pitches[pitches["game_year"] == season].to_parquet(data / f"savant_minors_{season}.parquet", index=False)
        (out / str(season) / "promoted_test").mkdir(parents=True)
        m[m["season"] == season][["key_mlbam", "season"] + etas].assign(zeta_scout=np.nan).to_parquet(
            out / str(season) / "promoted_test" / "eta_M1.parquet", index=False
        )
    m.drop(columns=etas).to_parquet(data / "model_bat.parquet", index=False)
    prof = per_player_dist(pitches)
    assert (prof["ev_q90"] >= prof["ev_q10"]).all() and prof["gm_q50"].notna().any(), "十分位不對"
    main(str(out), str(data), "promoted_test", "AAA")
    d, _ = load(out, data, "promoted_test", "AAA", profile_fn=per_player_dist, labels=LABELS)
    b90, _, _, _ = pooled_coef(d, "y_prom", "ev_q90")
    b10, _, _, _ = pooled_coef(d, "y_prom", "ev_q10")
    assert b90 > 0.3 and b90 > b10 + 0.3, f"上尾沒抓到：q90 {b90:.2f}, q10 {b10:.2f}"
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="用每個人資料的分布（十分位）而不是平均看球隊在看哪一段")
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
