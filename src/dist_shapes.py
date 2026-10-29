"""分布的形狀：每個人每一顆球的初速（還有接觸後 xwOBA、每場 wOBA）形成什麼樣的分布；升上去的人和沒升的人、WAR 高的人和低的人，形狀一樣嗎。

    python src/dist_shapes.py outputs/ data/ --promoted promoted_f4sed_linear_p2 [--level AAA]

描述一個分布的數學語言，這支都算一遍（每人季一組）：
  非參數：經驗 CDF F(x)、分位數函數 Q(p)（九個十分位）——不假設形狀，直接記。
  動差：平均 μ、標準差 σ、偏度 g1 = E[(x−μ)³]/σ³（尾巴往哪邊）、超額峰度 g2 = E[(x−μ)⁴]/σ⁴ − 3（尾巴多厚）。
  雙峰係數 BC = (g1² + 1) / (g2 + 3(n−1)²/((n−2)(n−3)))，> 0.555 表示像兩座山（初速常見：沒打好的一團 70 幾、打好的一團 95 以上）。
  參數族：兩個常態的混合 f(x) = w·N(μ_hi, σ_hi) + (1−w)·N(μ_lo, σ_lo)，用 EM 估，w = 打好的那一團佔幾成、μ_hi 那一團多強。
  兩個分布的距離：KS 統計量 sup|F−G|（最大落差）、Wasserstein W1 = ∫|F−G|dx（把一座山搬成另一座要搬多遠）。
比什麼：三個結果（隔年升不升、到目前為止登上、登上者 WAR 高三分之一 vs 低三分之一）各分兩群，
  (1) 畫圖：每群合併的密度、每個人的分位數函數（細線）與群平均（粗線）→ outputs/figures/。
  (2) 形狀參數表：兩群各自的平均、差 ÷ 標準差、平均差的 BF10（Savage–Dickey，先驗是 0.5 個標準差的效果）、合併分布的 KS 與 W1。
  (3) 形狀在平均與標準差之外有沒有用：基底（四把尺、球探、年齡、打席）+ 平均 + 標準差，再加一個形狀量，逐季係數合併。
輸出 outputs/dist_shapes_<promoted tag>.md 與 outputs/figures/dist_shapes_<tag>_*.png。
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]  # 中文標籤；沒有就退回英文字型
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from hit_profile import BASE, coef_se  # noqa: E402
from bf import bf_str, log10_bf10  # noqa: E402
from hit_profile_statcast import load, load_pitches  # noqa: E402

QS = np.arange(0.1, 0.91, 0.1)
SERIES = {"ev": "進場球初速 (mph)", "xw": "接觸後預期 wOBA", "gm": "每場 wOBA"}
SHAPES = {
    "mean": "平均",
    "sd": "標準差",
    "skew": "偏度 g1",
    "kurt": "超額峰度 g2",
    "bc": "雙峰係數 BC",
    "mix_w": "混合：打好那團佔幾成 w",
    "mix_mu_hi": "混合：打好那團的平均 μ_hi",
    "mix_mu_lo": "混合：沒打好那團的平均 μ_lo",
}


def raw_series(pitches: pd.DataFrame) -> dict:
    """每人季三個原始序列：每顆進場球初速、每顆進場球 xwOBA、每場 wOBA（打席 ≥ 3）。回傳 {(key, season): {"ev": array, ...}}。"""
    pa = pitches[pitches["events"].notna()].copy()
    pa["ev"] = pd.to_numeric(pa["launch_speed"], errors="coerce").where(pa["bb_type"].notna())
    pa["xw"] = pd.to_numeric(pa["estimated_woba_using_speedangle"], errors="coerce").where(pa["bb_type"].notna())
    pa["wv"] = pd.to_numeric(pa["woba_value"], errors="coerce")
    pa["wd"] = pd.to_numeric(pa["woba_denom"], errors="coerce")
    out = {}
    for (key, season), g in pa.groupby(["batter", "game_year"]):
        game = g.groupby("game_pk").agg(wv=("wv", "sum"), wd=("wd", "sum"), n=("wv", "size"))
        game = game[(game["n"] >= 3) & (game["wd"] > 0)]
        out[(int(key), int(season))] = {
            "ev": g["ev"].dropna().to_numpy(),
            "xw": g["xw"].dropna().to_numpy(),
            "gm": (game["wv"] / game["wd"]).to_numpy(),
        }
    return out


def em_two_normals(x: np.ndarray, iters: int = 60) -> tuple:
    """兩個常態的混合，EM 估：回傳 (w_hi, mu_lo, mu_hi)。起始用第 30 與第 80 百分位。"""
    mu = np.array([np.quantile(x, 0.3), np.quantile(x, 0.8)])
    sd = np.array([x.std(), x.std()]) / 2 + 1e-6
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        dens = np.column_stack([w[k] * stats.norm.pdf(x, mu[k], sd[k]) for k in range(2)]) + 1e-300
        r = dens / dens.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-9
        w = nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-6
    hi = int(np.argmax(mu))
    return float(w[hi]), float(mu[1 - hi]), float(mu[hi])


def shape_params(x: np.ndarray) -> dict:
    """一個序列的形狀參數；少於 30 筆回傳空。"""
    n = len(x)
    if n < 30:
        return {}
    g1, g2 = float(stats.skew(x)), float(stats.kurtosis(x))
    bc = (g1**2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    w, mu_lo, mu_hi = em_two_normals(x)
    return {
        "mean": float(x.mean()),
        "sd": float(x.std()),
        "skew": g1,
        "kurt": g2,
        "bc": float(bc),
        "mix_w": w,
        "mix_mu_hi": mu_hi,
        "mix_mu_lo": mu_lo,
    }


def groups_of(d: pd.DataFrame) -> dict:
    """三種分法：各回傳 (布林 Series 群 1, 布林 Series 群 0, 標籤 1, 標籤 0)。"""
    out = {"升級": (d["y_prom"] == 1, d["y_prom"] == 0, "隔年升上大聯盟", "沒升")}
    if "y_reach" in d.columns and d["y_reach"].notna().sum() > 50:
        out["登上"] = (d["y_reach"] == 1, d["y_reach"] == 0, "到目前為止登上", "沒登上")
    if "y_war" in d.columns and d["y_war"].notna().sum() > 60:
        w = d["y_war"]
        hi, lo = w.quantile(2 / 3), w.quantile(1 / 3)
        out["WAR"] = (w >= hi, w <= lo, "登上者 WAR 高三分之一", "登上者 WAR 低三分之一")
    return out


def pooled(arrs: list) -> np.ndarray:
    return np.concatenate([a for a in arrs if len(a)]) if arrs else np.array([])


def plot_all(d: pd.DataFrame, raw: dict, groups: dict, path_density: Path, path_quantile: Path, title: str) -> None:
    """圖一：每群合併密度；圖二：每個人的分位數函數（細線）與群平均（粗線）。列 = 三個序列，欄 = 三種分法。"""
    ncol = len(groups)
    fig1, ax1 = plt.subplots(3, ncol, figsize=(5 * ncol, 11), squeeze=False)
    fig2, ax2 = plt.subplots(3, ncol, figsize=(5 * ncol, 11), squeeze=False)
    colors = {1: "#c0392b", 0: "#2c3e50"}
    for j, (gname, (m1, m0, l1, l0)) in enumerate(groups.items()):
        for i, (s, sname) in enumerate(SERIES.items()):
            for flag, mask, label in ((1, m1, l1), (0, m0, l0)):
                keys = [(int(k), int(y)) for k, y in zip(d.loc[mask, "key_mlbam"], d.loc[mask, "season"])]
                arrs = [raw[k][s] for k in keys if k in raw and len(raw[k][s]) >= 30]
                pool = pooled(arrs)
                if len(pool) == 0:
                    continue
                lo, hi = np.quantile(pool, [0.005, 0.995])
                ax1[i, j].hist(pool, bins=60, range=(lo, hi), density=True, histtype="step", color=colors[flag], lw=1.8, label=f"{label}（{len(arrs)} 人）")
                qs = np.array([np.quantile(a, QS) for a in arrs])
                for q in qs[: min(len(qs), 150)]:
                    ax2[i, j].plot(QS * 100, q, color=colors[flag], alpha=0.08, lw=0.7)
                ax2[i, j].plot(QS * 100, qs.mean(axis=0), color=colors[flag], lw=2.6, label=f"{label}（{len(arrs)} 人）")
            ax1[i, j].set_title(f"{gname}：{sname} 的合併密度", fontsize=10)
            ax2[i, j].set_title(f"{gname}：每個人的分位數函數 Q(p)", fontsize=10)
            ax2[i, j].set_xlabel("百分位 p")
            ax2[i, j].set_ylabel(sname)
            ax1[i, j].set_xlabel(sname)
            ax1[i, j].legend(fontsize=8)
            ax2[i, j].legend(fontsize=8, loc="upper left")
    for fig, path in ((fig1, path_density), (fig2, path_quantile)):
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def shape_table(sh: pd.DataFrame, d: pd.DataFrame, raw: dict, groups: dict, s: str) -> pd.DataFrame:
    """一個序列：形狀參數在兩群的平均與 Mann–Whitney p；最後兩列是合併分布的 KS 與 W1。"""
    rows = []
    for gname, (m1, m0, l1, l0) in groups.items():
        for p, label in SHAPES.items():
            col = f"{s}_{p}"
            a, b = sh.loc[m1, col].dropna(), sh.loc[m0, col].dropna()
            if len(a) < 10 or len(b) < 10:
                continue
            diff = float(a.mean() - b.mean())
            se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)) + 1e-12
            l10 = log10_bf10(diff, se, tau=0.5 * pooled_sd)  # 先驗：差是 0.5 個標準差那種「中等」效果
            rows.append({"分法": f"{gname}：{l1} vs {l0}", "形狀參數": label, "群 1 平均": round(a.mean(), 3), "群 0 平均": round(b.mean(), 3), "差": round(diff, 3), "差 ÷ 標準差": round(diff / pooled_sd, 2), "BF10（有差）": bf_str(l10)})
        k1 = [(int(k), int(y)) for k, y in zip(d.loc[m1, "key_mlbam"], d.loc[m1, "season"])]
        k0 = [(int(k), int(y)) for k, y in zip(d.loc[m0, "key_mlbam"], d.loc[m0, "season"])]
        p1 = pooled([raw[k][s] for k in k1 if k in raw])
        p0 = pooled([raw[k][s] for k in k0 if k in raw])
        if len(p1) > 100 and len(p0) > 100:
            ks = stats.ks_2samp(p1, p0)
            rows.append({"分法": f"{gname}：{l1} vs {l0}", "形狀參數": "合併分布：KS 最大落差（群 1 欄）/ W1 距離（群 0 欄）", "群 1 平均": round(float(ks.statistic), 3), "群 0 平均": round(float(stats.wasserstein_distance(p1, p0)), 3), "差": np.nan, "差 ÷ 標準差": np.nan, "BF10（有差）": ""})
    return pd.DataFrame(rows)


def beyond_mean_sd(sh: pd.DataFrame, d: pd.DataFrame, s: str, outcomes: list) -> pd.DataFrame:
    """基底 + 平均 + 標準差之外，每個形狀量的逐季合併係數。"""
    dd = d.copy()
    cols = [f"{s}_{p}" for p in SHAPES]
    for c in cols:
        dd[c] = sh[c].astype(float)
        dd[c] = dd.groupby("season")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
    base = BASE + [c for c in dd.columns if c.startswith("lvl_")] + [f"{s}_mean", f"{s}_sd"]
    rows = []
    for p, label in SHAPES.items():
        if p in ("mean", "sd"):
            continue
        row = {"形狀量": label}
        for outcome, olabel in outcomes:
            bs, ses = [], []
            for season in sorted(dd["season"].unique()):
                b, se = coef_se(dd[dd["season"] == season], outcome, base + [f"{s}_{p}"])
                if np.isfinite(b) and np.isfinite(se) and se > 0:
                    bs.append(b)
                    ses.append(se)
            if bs:
                w = 1 / np.array(ses) ** 2
                pb = float((np.array(bs) * w).sum() / w.sum())
                row[olabel] = f"{pb:+.2f} [BF {bf_str(log10_bf10(pb, float(np.sqrt(1 / w.sum()))))}]"
        rows.append(row)
    return pd.DataFrame(rows)


def main(out_dir: str, data_dir: str, tag_p: str, level: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    d, _ = load(out, data, tag_p, level)
    pitches = pd.concat([load_pitches(data, int(s)) for s in sorted(d["season"].unique())], ignore_index=True)
    pitches = pitches[pitches["level"] == level]
    raw = raw_series(pitches)
    rows = []
    for k, y in zip(d["key_mlbam"], d["season"]):
        r = raw.get((int(k), int(y)), {})
        row = {}
        for s in SERIES:
            for p, v in shape_params(r.get(s, np.array([]))).items():
                row[f"{s}_{p}"] = v
        rows.append(row)
    sh = pd.DataFrame(rows, index=d.index)
    groups = groups_of(d)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    f_dens, f_quant = figdir / f"dist_shapes_{tag_p}_density.png", figdir / f"dist_shapes_{tag_p}_quantile.png"
    plot_all(d, raw, groups, f_dens, f_quant, f"{level} {', '.join(str(int(s)) for s in sorted(d['season'].unique()))}，{len(d)} 人季")
    outcomes = [(o, l) for o, l in (("y_prom", "升級"), ("y_reach", "登上"), ("y_war", "WAR")) if o in d.columns and d[o].notna().sum() >= 50]
    md = [
        f"# 分布的形狀：升上去的人、WAR 高的人，每顆球的分布長得一樣嗎 · {level} · {tag_p}\n",
        f"樣本 {len(d):,} 人季；每人季每顆進場球的初速、接觸後 xwOBA、每場 wOBA 三個序列（≥ 30 筆才算形狀）。圖：`{f_dens.relative_to(out)}`（合併密度）、`{f_quant.relative_to(out)}`（每個人的分位數函數）。\n",
        "形狀參數：偏度 g1（尾巴往哪邊）、超額峰度 g2（尾巴多厚）、雙峰係數 BC（> 0.555 像兩座山）、兩常態混合的 w（打好那團佔幾成）與 μ_hi、μ_lo（偏態常態 α 的最大概似在這種資料上發散，不列；它跟 g1 說同一件事）。"
        "KS = 兩群合併 CDF 的最大落差；W1 = Wasserstein 距離（把一座山搬成另一座的平均搬運距離，單位同變數）。\n",
    ]
    for s, sname in SERIES.items():
        md += [f"## {sname}\n", "### 兩群的形狀參數\n", shape_table(sh, d, raw, groups, s).to_markdown(index=False), "", "### 形狀在「基底 + 平均 + 標準差」之外的係數（方括號 BF10）\n", beyond_mean_sd(sh, d, s, outcomes).to_markdown(index=False), ""]
    text = "\n".join(md)
    path = out / f"dist_shapes_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="每個人每顆球的分布形狀：畫圖、形狀參數、兩群比較")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("--promoted", required=True, help="升級模型的 tag，例如 promoted_f4sed_linear_p2")
    p.add_argument("--level", default="AAA", choices=["AAA", "A"])
    a = p.parse_args()
    main(a.out_dir, a.data_dir, a.promoted, a.level)
