"""每個球團有沒有自己的尺：把 --org-slopes fit 的「該隊權重 − 全聯盟共識」跨季合併，看哪隊的偏離每季同一個方向。

    python src/org_rulers.py outputs/ --tag promoted_f4so_linear_p2
    python src/org_rulers.py --selftest

每季每隊每把尺：偏離 d（後驟平均）與 sd，來自 org_M1.csv 的「升級 β <尺> 偏離」欄；
舊檔沒有那兩欄時，偏離 = 該隊 β − struct_M1.csv 的共識 β，sd 用 summary.md 裡 beta_<尺>_org[i] 那排 sd 的中位數。
跨季：反變異數合併 → 合併偏離、se、幾季正／負；證據 Savage–Dickey BF10（先驗 N(0, 0.2²)：0.2 是「一隊的權重跟共識差 0.2」，相當於共識權重的三分之一到一半）。
另列每季的 σ_org（三十隊偏離的散布）跟共識 β，看「隊間差異 ÷ 共識」有多大。
輸出 outputs/org_rulers_<tag>.md、outputs/org_rulers_<tag>.csv。
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bf import bf_str, log10_bf10  # noqa: E402

RULERS = ["contact", "power", "speed", "discipline", "scout"]
PRIOR_SD = 0.2


def season_deviations(fit_dir: Path) -> pd.DataFrame:
    """一季：每隊每把尺的偏離與 sd（長表：org, ruler, dev, sd）。"""
    org = pd.read_csv(fit_dir / "org_M1.csv")
    struct = pd.read_csv(fit_dir / "struct_M1.csv", index_col=0)
    text = (fit_dir / "summary.md").read_text() if (fit_dir / "summary.md").exists() else ""
    rows = []
    for r in RULERS:
        col = f"升級 β {r}"
        if col not in org.columns:
            continue
        if f"{col} 偏離" in org.columns:
            dev = org[f"{col} 偏離"].to_numpy(dtype=float)
            sd = org[f"{col} 偏離 sd"].to_numpy(dtype=float)
        else:  # 舊檔：自己減共識，sd 用 summary 的中位數
            common_key = "beta_scout" if r == "scout" else f"beta_{r}[0]"
            dev = org[col].to_numpy(dtype=float) - float(struct.loc[common_key, "mean"])
            sds = [float(m.group(1)) for m in re.finditer(rf"\| beta_{r}_org\[\d+\]\s*\|\s*[-\d.]+\s*\|\s*([\d.]+)", text)]
            sd = np.full(len(org), float(np.median(sds)) if sds else np.nan)
        for o, d_, s_ in zip(org["org"], dev, sd):
            rows.append({"org": o, "ruler": r, "dev": d_, "sd": s_})
    return pd.DataFrame(rows)


def season_spread(fit_dir: Path) -> dict:
    """一季：共識 β 與 σ_org（隊間散布）。"""
    struct = pd.read_csv(fit_dir / "struct_M1.csv", index_col=0)
    out = {}
    for r in RULERS:
        ck = "beta_scout" if r == "scout" else f"beta_{r}[0]"
        sk = f"sigma_org_{r}"
        if ck in struct.index:
            out[f"共識 {r}"] = float(struct.loc[ck, "mean"])
        if sk in struct.index:
            out[f"σ_org {r}"] = float(struct.loc[sk, "mean"])
    return out


def pool_clubs(d: pd.DataFrame) -> pd.DataFrame:
    """跨季反變異數合併，每隊每把尺一列。"""
    rows = []
    for (o, r), g in d.groupby(["org", "ruler"]):
        g = g[np.isfinite(g["dev"]) & np.isfinite(g["sd"]) & (g["sd"] > 0)]
        if g.empty:
            continue
        w = 1.0 / g["sd"] ** 2
        b = float((g["dev"] * w).sum() / w.sum())
        se = float(np.sqrt(1.0 / w.sum()))
        rows.append(
            {
                "org": o,
                "ruler": r,
                "季數": len(g),
                "合併偏離": b,
                "se": se,
                "幾季正": int((g["dev"] > 0).sum()),
                "幾季負": int((g["dev"] < 0).sum()),
                "log10 BF10": log10_bf10(b, se, PRIOR_SD),
            }
        )
    return pd.DataFrame(rows)


def wide_table(pooled: pd.DataFrame) -> pd.DataFrame:
    """三十隊 × 五把尺：「偏離 (se) · 正/負季數 · BF」。"""
    cells = {}
    for r in pooled.itertuples():
        cells.setdefault(r.org, {})[r.ruler] = f"{r.合併偏離:+.2f} ({r.se:.2f}) {r.幾季正}/{r.幾季負} · {bf_str(r._8)}"
    t = pd.DataFrame.from_dict(cells, orient="index").reindex(columns=[r for r in RULERS if r in pooled["ruler"].unique()])
    t.index.name = "org"
    strongest = pooled.groupby("org")["log10 BF10"].max()
    return t.loc[strongest.sort_values(ascending=False).index]


def main(out_dir: str, tag: str) -> str:
    out = Path(out_dir)
    frames, spreads = [], []
    for p in sorted(out.glob(f"20*/{tag}/org_M1.csv")):
        season = int(p.parent.parent.name)
        f = season_deviations(p.parent)
        f["season"] = season
        frames.append(f)
        sp = season_spread(p.parent)
        sp["季"] = season
        spreads.append(sp)
        print(f"  {season}: {f['org'].nunique()} 隊", flush=True)
    if not frames:
        raise SystemExit(f"找不到 {tag} 的 org_M1.csv")
    d = pd.concat(frames, ignore_index=True)
    pooled = pool_clubs(d)
    wide = wide_table(pooled)
    spread = pd.DataFrame(spreads).set_index("季").round(2)
    n_strong = int((pooled["log10 BF10"] >= 1).sum())
    n_null = int((pooled["log10 BF10"] <= -0.5).sum())
    md = [
        f"# 每個球團有沒有自己的尺 · {tag} · {d['season'].nunique()} 季\n",
        f"每格：合併偏離（該隊權重 − 全聯盟共識，反變異數合併）(se)、幾季正/幾季負、BF10（偏離 ≠ 0 vs = 0，先驗 N(0, {PRIOR_SD}²)）。"
        f"{len(pooled)} 個隊 × 尺的格子裡，BF10 ≥ 10 的有 {n_strong} 個，支持零（BF10 ≤ 1/3）的有 {n_null} 個。\n",
        "## 三十隊 × 五把尺（照最強證據排序）\n",
        wide.to_markdown(),
        "",
        "## 每季：共識權重與隊間散布 σ_org\n",
        spread.to_markdown(),
        "",
        "讀法：σ_org ÷ 共識 = 隊間差異相對共識有多大；某隊某尺幾季同號且 BF ≥ 10 = 這隊真的有自己的偏好；"
        "多數格子支持零 = 三十隊用的是同一把尺，sim_policy 看到的「球隊的尺 vs 真值尺」差是整個行業的。\n",
    ]
    text = "\n".join(md)
    (out / f"org_rulers_{tag}.md").write_text(text)
    pooled.to_csv(out / f"org_rulers_{tag}.csv", index=False)
    print(text, flush=True)
    return text


def selftest() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    orgs = [f"T{i:02d}" for i in range(12)]
    true_dev = {o: {r: 0.0 for r in RULERS} for o in orgs}
    true_dev["T00"]["speed"] = 0.25  # 只有一隊真的多看速度
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for s in range(2010, 2018):
            fd = out / str(s) / "X"
            fd.mkdir(parents=True)
            rows = {"org": orgs, "人季": 120}
            for r in RULERS:
                dev = np.array([true_dev[o][r] for o in orgs]) + rng.normal(0, 0.08, len(orgs))
                rows[f"升級 β {r}"] = 0.5 + dev
                rows[f"升級 β {r} 偏離"] = dev
                rows[f"升級 β {r} 偏離 sd"] = 0.08
            pd.DataFrame(rows).to_csv(fd / "org_M1.csv", index=False)
            struct = pd.DataFrame({"mean": [0.5] * 5 + [0.1] * 5}, index=[f"beta_{r}[0]" if r != "scout" else "beta_scout" for r in RULERS] + [f"sigma_org_{r}" for r in RULERS])
            struct.index.name = ""
            struct.to_csv(fd / "struct_M1.csv")
        main(tmp, "X")
        pooled = pd.read_csv(out / "org_rulers_X.csv")
    hit = pooled[(pooled["org"] == "T00") & (pooled["ruler"] == "speed")].iloc[0]
    assert hit["log10 BF10"] >= 2, hit
    others = pooled[~((pooled["org"] == "T00") & (pooled["ruler"] == "speed"))]
    assert (others["log10 BF10"] >= 1).mean() < 0.1, others.sort_values("log10 BF10").tail()
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="球團各自的尺：--org-slopes fit 的偏離跨季合併，BF")
    ap.add_argument("out_dir", nargs="?", default="outputs")
    ap.add_argument("--tag", default="promoted_f4so_linear_p2")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        main(a.out_dir, a.tag)
