"""進階數據 vs 現有的尺，哪個比較會預測：同一批 3A 人季、同一個結果、同一套樣本外評估，直接對決。

    python src/statcast_vs_rulers.py outputs/ data/ --promoted promoted_f4sed_linear_p2 [--promoted5 promoted_f5sed_linear_p2]

比五組輸入（都加共變數：球探評價、年齡、打席）：
  共變數        只有球探評價、年齡、打席（基準）
  四把尺        成績單估出來的打得到球、打得遠、跑得快、選得到球
  五把尺        加擊球品質（LD%、IFFB%、BABIP 那把，--promoted5 給的 tag）
  Statcast      3A 逐球算的過程量：EV90、強勁擊球率、barrel 率、xwOBAcon、平均仰角、滾地率、拉打飛球率、區內接觸率、追打率、整體揮空率
  尺 + Statcast 兩個都放
結果：隔年升不升、到目前為止登上、登上者 fWAR（asinh）。
評估一：前一季估、後一季測（兩個方向），log-loss、AUC；WAR 用均方誤差與相關。
評估二：兩季合起來 5 折交叉驗證重複 20 次（430 人太少，單一切法會晃），報平均與重複間標準差。
評估三：單一變數對決——每把尺 vs 它對應的 Statcast 量，單變數 AUC（升級、登上）與相關（WAR）：
  打得到球 vs 區內接觸率；打得遠 vs EV90、barrel 率；選得到球 vs 追打率；擊球品質 vs xwOBAcon。
輸出 outputs/statcast_vs_rulers_<tag>.md 與 outputs/figures/statcast_vs_rulers_<tag>.png。
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from decision_style import fit_logit, log_loss, logit_predict  # noqa: E402
from hit_profile_statcast import load  # noqa: E402

COV = ["zeta_scout", "age_rel", "log_pa"]
RULERS4 = ["eta_contact", "eta_power", "eta_speed", "eta_discipline"]
STATCAST = ["ev90", "hard_bip", "barrel_bip", "xwoba_con", "la_mean", "gb_bip", "pullair_bip", "zcon", "chase", "whiff_sw"]
OUTCOMES = [("y_prom", "隔年升不升"), ("y_reach", "到目前為止登上"), ("y_war", "登上者 fWAR")]
DUELS = [
    ("打得到球", "eta_contact", "區內接觸率", "zcon"),
    ("打得遠", "eta_power", "EV90", "ev90"),
    ("打得遠", "eta_power", "barrel 率", "barrel_bip"),
    ("選得到球", "eta_discipline", "追打率（負）", "chase"),
    ("擊球品質", "eta_quality", "接觸後 xwOBA", "xwoba_con"),
]
RIDGE = 0.5


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """排序 AUC（Mann–Whitney）。"""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list, outcome: str) -> np.ndarray:
    Xa, Xb = tr[feats].to_numpy(dtype=float), te[feats].to_numpy(dtype=float)
    y = tr[outcome].to_numpy(dtype=float)
    if outcome == "y_war":
        A = np.column_stack([np.ones(len(Xa)), Xa])
        b = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ y)
        return np.column_stack([np.ones(len(Xb)), Xb]) @ b
    return logit_predict(fit_logit(Xa, y, ridge=RIDGE), Xb)


def score(y: np.ndarray, p: np.ndarray, outcome: str) -> dict:
    if outcome == "y_war":
        return {"主": float(((y - p) ** 2).mean()), "副": float(np.corrcoef(y, p)[0, 1]) if p.std() > 0 else np.nan}
    return {"主": log_loss(y, p), "副": auc(y, p)}


def season_to_season(d: pd.DataFrame, sets: dict, outcome: str) -> pd.DataFrame:
    """前一季估後一季測，兩個方向。"""
    seasons = sorted(d["season"].unique())
    rows = []
    for name, feats in sets.items():
        row = {"輸入": name}
        for a, b in [(seasons[0], seasons[1]), (seasons[1], seasons[0])]:
            tr = d[(d["season"] == a) & d[outcome].notna()]
            te = d[(d["season"] == b) & d[outcome].notna()]
            if len(tr) < 60 or len(te) < 30:
                continue
            s = score(te[outcome].to_numpy(dtype=float), fit_predict(tr, te, feats, outcome), outcome)
            row[f"{a}→{b} 主"] = round(s["主"], 4)
            row[f"{a}→{b} 副"] = round(s["副"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def cross_validated(d: pd.DataFrame, sets: dict, outcome: str, reps: int = 20, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """兩季合起來 k 折，重複 reps 次；報平均與重複間標準差。"""
    rng = np.random.default_rng(seed)
    dd = d[d[outcome].notna()].reset_index(drop=True)
    y_all = dd[outcome].to_numpy(dtype=float)
    out = {name: {"主": [], "副": []} for name in sets}
    for _ in range(reps):
        fold = rng.permutation(len(dd)) % folds
        for name, feats in sets.items():
            pred = np.zeros(len(dd))
            for f in range(folds):
                tr, te = dd[fold != f], dd[fold == f]
                pred[fold == f] = fit_predict(tr, te, feats, outcome)
            s = score(y_all, pred, outcome)
            out[name]["主"].append(s["主"])
            out[name]["副"].append(s["副"])
    main_label = "均方誤差" if outcome == "y_war" else "log-loss"
    sub_label = "相關" if outcome == "y_war" else "AUC"
    rows = []
    for name in sets:
        rows.append(
            {
                "輸入": name,
                main_label: f"{np.mean(out[name]['主']):.4f} ± {np.std(out[name]['主']):.4f}",
                sub_label: f"{np.nanmean(out[name]['副']):.3f} ± {np.nanstd(out[name]['副']):.3f}",
                "_main": float(np.mean(out[name]["主"])),
                "_sub": float(np.nanmean(out[name]["副"])),
            }
        )
    return pd.DataFrame(rows)


def duels(d: pd.DataFrame) -> pd.DataFrame:
    """單一變數對決：各自單獨放（加共變數）的 5 折 AUC／相關。"""
    rng = np.random.default_rng(1)
    rows = []
    for ruler_label, ruler, sc_label, sc in DUELS:
        if ruler not in d.columns or sc not in d.columns:
            continue
        row = {"尺": ruler_label, "Statcast 量": sc_label}
        for outcome, olabel in OUTCOMES:
            dd = d[d[outcome].notna()].reset_index(drop=True)
            y = dd[outcome].to_numpy(dtype=float)
            res = {}
            for tag, feats in (("尺", COV + [ruler]), ("sc", COV + [sc])):
                vals = []
                for _ in range(10):
                    fold = rng.permutation(len(dd)) % 5
                    pred = np.zeros(len(dd))
                    for f in range(5):
                        pred[fold == f] = fit_predict(dd[fold != f], dd[fold == f], feats, outcome)
                    vals.append(score(y, pred, outcome)["副"])
                res[tag] = float(np.nanmean(vals))
            row[f"{olabel}：尺"] = round(res["尺"], 3)
            row[f"{olabel}：Statcast"] = round(res["sc"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def plot(cv_tables: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(cv_tables), figsize=(5.2 * len(cv_tables), 4.2), squeeze=False)
    for ax, (olabel, tab) in zip(axes[0], cv_tables.items()):
        base = tab["_main"].iloc[0]
        gain = base - tab["_main"]
        colors = ["#7f8c8d" if i == 0 else "#2c3e50" if "Statcast" not in n else "#c0392b" for i, n in enumerate(tab["輸入"])]
        ax.barh(tab["輸入"], gain, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{olabel}：比「只有共變數」少掉多少{'均方誤差' if 'WAR' in olabel else ' log-loss'}", fontsize=10)
        ax.invert_yaxis()
        for i, (g, s) in enumerate(zip(gain, tab["_sub"])):
            ax.text(max(g, 0) + gain.max() * 0.02, i, f"{'相關' if 'WAR' in olabel else 'AUC'} {s:.3f}", va="center", fontsize=8)
        ax.set_xlim(min(0, gain.min() * 1.1), gain.max() * 1.45)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out_dir: str, data_dir: str, tag_p: str, tag5: str | None, level: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    d, names = load(out, data, tag_p, level)
    sets = {"共變數": COV, "四把尺": COV + RULERS4}
    if tag5:
        q = []
        for season in sorted(d["season"].unique()):
            p = out / str(int(season)) / tag5 / "eta_M1.parquet"
            if p.exists():
                q.append(pd.read_parquet(p)[["key_mlbam", "season", "eta_quality"]])
        if q:
            d = d.merge(pd.concat(q), on=["key_mlbam", "season"], how="left")
            d["eta_quality"] = d.groupby("season")["eta_quality"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
            sets["五把尺"] = COV + RULERS4 + ["eta_quality"]
    sc = [v for v in STATCAST if v in d.columns and d[v].abs().sum() > 0]
    sets["Statcast"] = COV + sc
    sets["尺 + Statcast"] = sets.get("五把尺", sets["四把尺"]) + sc
    outcomes = [(o, l) for o, l in OUTCOMES if o in d.columns and d[o].notna().sum() >= 60 and d[o].nunique() >= 2]
    seasons = ", ".join(str(int(s)) for s in sorted(d["season"].unique()))
    md = [
        f"# 進階數據 vs 現有的尺：同一批人、同一個結果，哪個比較會預測 · {level} · {tag_p}\n",
        f"樣本 {len(d):,} 人季（{seasons}），還沒登上大聯盟、進場球 ≥ 50。五組輸入都含共變數（球探評價、年齡、打席）。Statcast 這組：{'、'.join(sc)}。ridge {RIDGE}。\n",
    ]
    cv_tables = {}
    for outcome, olabel in outcomes:
        md += [f"## {olabel}\n", "### 前一季估、後一季測\n", season_to_season(d, sets, outcome).to_markdown(index=False), ""]
        cv = cross_validated(d, sets, outcome)
        cv_tables[olabel] = cv
        md += ["### 兩季合起來 5 折 × 20 次\n", cv.drop(columns=["_main", "_sub"]).to_markdown(index=False), ""]
    md += ["## 單一變數對決：每把尺 vs 對應的 Statcast 量（各自單獨加共變數，5 折 × 10 次；升級、登上是 AUC，WAR 是相關）\n", duels(d).to_markdown(index=False), ""]
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig = figdir / f"statcast_vs_rulers_{tag_p}.png"
    plot(cv_tables, fig, f"{level} {seasons}，{len(d)} 人季；5 折 × 20 次")
    md += [f"圖：`{fig.relative_to(out)}`\n"]
    text = "\n".join(md)
    path = out / f"statcast_vs_rulers_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="進階數據 vs 現有的尺：樣本外預測對決")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("--promoted", required=True, help="四把尺模型的 tag")
    p.add_argument("--promoted5", default=None, help="五把尺模型的 tag（有就多一組「五把尺」）")
    p.add_argument("--level", default="AAA", choices=["AAA", "A"])
    a = p.parse_args()
    main(a.out_dir, a.data_dir, a.promoted, a.promoted5, a.level)
