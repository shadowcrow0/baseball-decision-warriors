"""球隊買的是平均還是分布：母隊強弱 × 選手型態，誰升得快。

    python src/club_risk.py outputs/ data/ --promoted promoted_f4sed_linear_p2
    python src/club_risk.py --selftest

想法：勝率對得分是凸的。弱隊只在少數場次要贏，多一個高變異的打者（打得遠、三振多、有時爆發）比多一個穩定的打者划算；
強隊反過來，要的是不拖後腿的穩定輸出。這是「球隊的狀況決定它要哪一種分布」，跟守位缺口是不同的「需要」。
資料：建模表 + 升級模型的四把尺分數（跟 hit_profile 同一套基底、同一個季 × 層級內標準化）；母隊強度 = 母隊當季大聯盟團隊 WAR（club_features.add_club_strength），季內標準化。
選手型態 = 打得遠 − 打得到球（正 = 高變異型、負 = 穩定型）。
表一：母隊強度、母隊強度 × 型態、母隊強度 × 各把尺，對升級、登上、WAR 的合併係數（基底之外；逐季估、反變異數合併）。
表二：母隊強弱三等 × 型態三等的九格升級率（全部季合起來），直接看弱隊是不是偏升高變異型。
輸出 outputs/club_risk_<promoted tag>.md。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from club_features import add_club_strength
from hit_profile import BASE, coef_se, load
from bf import bf_str, log10_bf10

TYPE_LABEL = {"type_pc": "型態（打得遠 − 打得到球）", "eta_power": "打得遠", "eta_contact": "打得到球", "eta_discipline": "選得到球", "eta_speed": "跑得快"}


def add_strength(d: pd.DataFrame, data: Path) -> pd.DataFrame:
    """母隊當季團隊 WAR → 季內標準化的 club_c；沒資料補 0。再做型態與交互作用欄。"""
    d = add_club_strength(d, data)
    d["club_c"] = d.groupby("season")["club_war_t"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
    d["type_pc"] = d["eta_power"] - d["eta_contact"]
    d["type_pc"] = d.groupby(["season", "level_main"])["type_pc"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    for k in TYPE_LABEL:
        d[f"club_x_{k}"] = d["club_c"] * d[k]
    return d


def pooled(d: pd.DataFrame, outcome: str, feats: list) -> tuple:
    """逐季最後一個特徵的係數，反變異數合併：(係數, z, 幾季正, 幾季負)。"""
    bs, ses = [], []
    for s in sorted(d["season"].unique()):
        b, se = coef_se(d[d["season"] == s], outcome, feats)
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            bs.append(b)
            ses.append(se)
    if not bs:
        return np.nan, np.nan, 0, 0
    w = 1.0 / np.array(ses) ** 2
    pooled_b = float((np.array(bs) * w).sum() / w.sum())
    return pooled_b, pooled_b / float(np.sqrt(1 / w.sum())), sum(b > 0 for b in bs), sum(b < 0 for b in bs)


def coef_table(d: pd.DataFrame, outcomes: list) -> pd.DataFrame:
    base = BASE + [c for c in d.columns if c.startswith("lvl_")]
    rows = []
    specs = [("母隊強度", ["club_c"])] + [(f"母隊強度 × {TYPE_LABEL[k]}", ["club_c", f"club_x_{k}"]) for k in TYPE_LABEL]
    for name, extra in specs:
        row = {"變數": name}
        for outcome, label in outcomes:
            b, z, pos, neg = pooled(d, outcome, base + extra)
            row[label] = f"{b:+.3f} [BF {bf_str(log10_bf10(b, abs(b / z) if z else np.nan))}]  {pos}/{neg}" if np.isfinite(b) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def cell_table(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """母隊強弱三等（季內）× 型態三等（季 × 層級內）→ 結果平均與人數。"""
    ok = d[d[outcome].notna()].copy()
    ok["club_bin"] = ok.groupby("season")["club_c"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["弱隊", "中", "強隊"]))
    ok["type_bin"] = ok.groupby(["season", "level_main"])["type_pc"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["穩定型", "中", "高變異型"])
    )
    t = ok.pivot_table(index="club_bin", columns="type_bin", values=outcome, aggfunc=["mean", "size"], observed=True)
    out = t["mean"].round(3).astype(str) + " (n=" + t["size"].astype(int).astype(str) + ")"
    return out.reset_index().rename(columns={"club_bin": "母隊 \\ 選手"})


def main(out_dir: str, data_dir: str, tag_p: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    d, _ = load(out, data, tag_p)
    d = add_strength(d, data)
    outcomes = [("y_prom", "隔年升不升"), ("y_reach", "六季內登上"), ("y_war", "登上者六季 WAR")]
    outcomes = [(o, l) for o, l in outcomes if d[o].notna().sum() >= 100]
    md = [
        f"# 球隊買的是平均還是分布：母隊強弱 × 選手型態 · {tag_p}\n",
        f"樣本：{len(d):,} 人季，{int(d['season'].min())}–{int(d['season'].max())}，打席 ≥ 100。母隊強度 = 母隊當季大聯盟團隊 WAR（季內標準化）；"
        "型態 = 打得遠 − 打得到球（季 × 層級內標準化；正 = 高變異型）。係數在四把尺、球探、年齡、打席、層級之外；格式「係數 [BF10]  幾季正/幾季負」，BF10 是 Savage–Dickey（先驗 N(0, 0.5²)），> 10 強、< 1/3 支持零。"
        "交互作用為負 = 弱隊比強隊更偏好高變異型（勝率凸性的預測）。\n",
        "## 一、係數\n",
        coef_table(d, outcomes).to_markdown(index=False),
        "",
    ]
    for outcome, label in outcomes:
        md += [f"## 二、九格：{label}（平均，括號人數）\n", cell_table(d, outcome).to_markdown(index=False), ""]
    text = "\n".join(md)
    path = out / f"club_risk_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成：弱隊偏升高變異型、強隊偏升穩定型。交互作用要是負的、九格左上右下要高。"""
    rng = np.random.default_rng(7)
    root = Path(tempfile.mkdtemp())
    data, out = root / "data", root / "outputs"
    data.mkdir()
    teams = [f"T{i:02d}" for i in range(30)]
    strength = {t: rng.normal(0, 10) for t in teams}
    etas = ["eta_contact", "eta_power", "eta_speed", "eta_discipline"]
    rows, fgb, fgp = [], [], []
    for season in (2015, 2016, 2017, 2018):
        for t in teams:
            fgb.append({"Team": t, "Season": season, "WAR": 20 + strength[t] / 2})
            fgp.append({"Team": t, "Season": season, "WAR": 20 + strength[t] / 2})
        for i in range(600):
            t = teams[i % 30]
            e = {k: rng.normal() for k in etas}
            typ = e["eta_power"] - e["eta_contact"]
            logit = -1 + 0.4 * e["eta_contact"] + 0.3 * e["eta_power"] - 0.08 * strength[t] * typ
            rows.append(
                {
                    "key_mlbam": 10000 * (season - 2014) + i,
                    "season": season,
                    "level_main": "AA",
                    "org": t,
                    "PA": 400,
                    "promoted": rng.random() < 1 / (1 + np.exp(-logit)),
                    "next_status": "same",
                    "age_rel": rng.normal(),
                    "war_observable": True,
                    "reached_fwd": rng.random() < 0.3,
                    "war_fwd": rng.normal(1, 2),
                    **e,
                }
            )
    m = pd.DataFrame(rows)
    pd.DataFrame(fgb).to_parquet(data / "fg_batting.parquet", index=False)
    pd.DataFrame(fgp).to_parquet(data / "fg_pitching.parquet", index=False)
    for season in sorted(m["season"].unique()):
        (out / str(season) / "promoted_test").mkdir(parents=True)
        m[m["season"] == season][["key_mlbam", "season"] + etas].assign(zeta_scout=np.nan).to_parquet(
            out / str(season) / "promoted_test" / "eta_M1.parquet", index=False
        )
    m.drop(columns=etas).to_parquet(data / "model_bat.parquet", index=False)
    main(str(out), str(data), "promoted_test")
    d, _ = load(out, data, "promoted_test")
    d = add_strength(d, data)
    base = BASE + [c for c in d.columns if c.startswith("lvl_")]
    b, z, _, neg = pooled(d, "y_prom", base + ["club_c", "club_x_type_pc"])
    assert b < 0 and z < -3 and neg >= 3, f"交互作用沒抓到：{b:.3f} z {z:.1f}"
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="母隊強弱 × 選手型態：球隊買的是平均還是分布")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("--promoted", default=None, help="升級模型的 tag，例如 promoted_f4sed_linear_p2")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.promoted is None:
        p.error("--promoted 是必要的（或用 --selftest）")
    else:
        main(a.out_dir, a.data_dir, a.promoted)
