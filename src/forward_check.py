"""往前驗證：用幾個參考季估出來的「球隊怎麼加權四把尺」，拿去預測其他每一季的升級決定，看參數設定在接下來的資料上成不成立。

    python src/forward_check.py outputs/ data/ --promoted promoted_f4sed_linear_p2 --ref 2010 2015 2019 2023
    python src/forward_check.py --selftest

每一季 t 有自己的 fit（η、ζ 的後驗平均，struct_M1.csv 的係數）。這支不重配能力，只問結構段的權重能不能搬：
  固定的部分（offset）= 參考季平均的 β_contact·η + β_power·η + β_speed·η + β_discipline·η + β_scout·ζ + β_undrafted·非選秀 + β_multi·跨層級
  每季自由重估的部分 = 層級截距、年齡、打席（基準率逐季不同，這三個本來就不是「球隊看什麼」的主張）
三個預測器比 log-loss 與 AUC：
  基準      只有層級截距、年齡、打席
  參考季權重 基準 + 上面那個 offset（係數固定為 1）
  當季自己  基準 + 當季自己的 β（上界；跟參考季一樣只是重估）
再估「折扣係數」：把 offset 當一個變數放進去自由估斜率，斜率 ≈ 1 表示參考季的權重原樣可用，< 1 表示要打折，> 1 表示低估。
每個參考季單獨當 offset 也各跑一次，看四個參考季之間的分歧。輸出 outputs/forward_check_<tag>.md。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decision_style import auc, log_loss  # noqa: E402

RULERS = ["contact", "power", "speed", "discipline"]
FIXED = ["beta_contact[0]", "beta_power[0]", "beta_speed[0]", "beta_discipline[0]", "beta_scout", "undrafted", "multi_level"]
LEVELS = ["DSL", "R/CPX", "A-", "A", "A+", "AA", "AAA"]


def read_beta(out: Path, season: int, tag: str) -> pd.Series:
    st = pd.read_csv(out / str(season) / tag / "struct_M1.csv", index_col=0)["mean"]
    return st.reindex(FIXED).fillna(0.0)


def season_frame(out: Path, data: Path, season: int, tag: str) -> pd.DataFrame:
    """一季：η、ζ、升級、層級，加共變數（年齡、打席、非選秀、跨層級）。只留結構遮罩內有分數的人。"""
    import fit_year as fy

    e = pd.read_parquet(out / str(season) / tag / "eta_M1.parquet")
    m = fy.load(data, season)[["key_mlbam", "age_rel", "log_pa", "undrafted", "multi_level", "structural", "lvl"]]
    d = e.merge(m, on="key_mlbam", how="inner")
    d = d[d["structural"].astype(bool) & d["promoted"].notna() & d["score_promoted"].notna()].reset_index(drop=True)
    d["zeta_scout"] = d["zeta_scout"].fillna(0.0)
    d["y"] = d["promoted"].astype(float)
    return d


def offset_from(d: pd.DataFrame, beta: pd.Series) -> np.ndarray:
    o = np.zeros(len(d))
    for r in RULERS:
        o += beta[f"beta_{r}[0]"] * d[f"eta_{r}"].to_numpy(dtype=float)
    o += beta["beta_scout"] * d["zeta_scout"].to_numpy(dtype=float)
    o += beta["undrafted"] * d["undrafted"].to_numpy(dtype=float)
    o += beta["multi_level"] * d["multi_level"].to_numpy(dtype=float)
    return o


def base_design(d: pd.DataFrame) -> np.ndarray:
    """層級 0/1（沒有總截距）+ 年齡 + 打席（都標準化）。"""
    cols = [(d["lvl"].to_numpy() == l).astype(float) for l in np.unique(d["lvl"].to_numpy())]
    for c in ("age_rel", "log_pa"):
        x = d[c].to_numpy(dtype=float)
        cols.append((x - x.mean()) / (x.std() + 1e-9))
    return np.column_stack(cols)


def logit_offset(X: np.ndarray, y: np.ndarray, offset: np.ndarray, ridge: float = 1e-2, iters: int = 60) -> np.ndarray:
    """有 offset 的邏輯迴歸（IRLS）：logit p = X b + offset。回傳 b。"""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b + offset)))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def predict(X: np.ndarray, b: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ b + offset)))


def evaluate(d: pd.DataFrame, offset: np.ndarray) -> dict:
    """在一季內：基準、offset 固定為 1、offset 斜率自由估（折扣係數，含 se）。"""
    X, y = base_design(d), d["y"].to_numpy()
    b0 = logit_offset(X, y, np.zeros(len(d)))
    p0 = predict(X, b0, np.zeros(len(d)))
    b1 = logit_offset(X, y, offset)
    p1 = predict(X, b1, offset)
    Xs = np.column_stack([X, offset])
    bs = logit_offset(Xs, y, np.zeros(len(d)))
    ps = predict(Xs, bs, np.zeros(len(d)))
    w = ps * (1 - ps)
    cov = np.linalg.inv(Xs.T @ (Xs * w[:, None]) + 1e-2 * np.eye(Xs.shape[1]))
    return {
        "基準 log-loss": log_loss(y, p0),
        "基準 AUC": auc(y, p0),
        "固定權重 log-loss": log_loss(y, p1),
        "固定權重 AUC": auc(y, p1),
        "折扣係數": float(bs[-1]),
        "折扣係數 se": float(np.sqrt(cov[-1, -1])),
        "自由斜率 log-loss": log_loss(y, ps),
    }


def main(out_dir: str, data_dir: str, tag: str, refs: list) -> str:
    out, data = Path(out_dir), Path(data_dir)
    seasons = sorted(int(p.parent.parent.name) for p in out.glob(f"20*/{tag}/struct_M1.csv"))
    betas = {s: read_beta(out, s, tag) for s in seasons}
    ref_beta = pd.concat([betas[s] for s in refs if s in betas], axis=1).mean(axis=1)
    rows, by_ref = [], []
    for t in seasons:
        d = season_frame(out, data, t, tag)
        if len(d) < 200:
            continue
        own = evaluate(d, offset_from(d, betas[t]))
        ref = evaluate(d, offset_from(d, ref_beta))
        rows.append(
            {
                "季": t,
                "在參考季裡": "是" if t in refs else "",
                "n": len(d),
                "基準 log-loss": round(ref["基準 log-loss"], 4),
                "參考季權重 log-loss": round(ref["固定權重 log-loss"], 4),
                "當季自己 log-loss": round(own["固定權重 log-loss"], 4),
                "參考季權重 AUC": round(ref["固定權重 AUC"], 3),
                "當季自己 AUC": round(own["固定權重 AUC"], 3),
                "折扣係數（參考季權重）": f"{ref['折扣係數']:.2f} ± {ref['折扣係數 se']:.2f}",
                "權重搬過來損失的 log-loss": round(ref["固定權重 log-loss"] - own["固定權重 log-loss"], 4),
            }
        )
        for r in refs:
            if r in betas and r != t:
                e = evaluate(d, offset_from(d, betas[r]))
                by_ref.append({"目標季": t, "參考季": r, "log-loss": round(e["固定權重 log-loss"], 4), "AUC": round(e["固定權重 AUC"], 3), "折扣係數": round(e["折扣係數"], 2)})
    tab = pd.DataFrame(rows)
    br = pd.DataFrame(by_ref)
    piv = br.pivot(index="目標季", columns="參考季", values="折扣係數") if len(br) else pd.DataFrame()
    gain_base = (tab["基準 log-loss"] - tab["參考季權重 log-loss"]).mean()
    gain_own = (tab["基準 log-loss"] - tab["當季自己 log-loss"]).mean()
    md = [
        f"# 往前驗證：參考季 {refs} 的權重搬到其他季 · {tag}\n",
        "問題：四把尺的權重（加球探評價、非選秀、跨層級）在參考季估出來之後，原樣搬到別的季還能預測嗎？"
        "每季只重估層級截距、年齡、打席。折扣係數 ≈ 1 = 原樣可用。\n",
        "## 每一季\n",
        tab.to_markdown(index=False),
        "",
        f"平均：參考季權重比基準多賺 {gain_base:.4f} log-loss，當季自己多賺 {gain_own:.4f}；搬過來保留了 {gain_base / gain_own:.0%} 的預測力。\n",
        "## 單一參考季的折扣係數（列 = 目標季，欄 = 參考季；1 = 原樣可用）\n",
        piv.round(2).to_markdown() if len(piv) else "（無）",
        "",
    ]
    text = "\n".join(md)
    path = out / f"forward_check_{tag}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成：同一組權重生兩季 → 折扣係數 ≈ 1、搬過來幾乎不損失；權重減半生第三季 → 折扣係數 ≈ 0.5。"""
    rng = np.random.default_rng(0)
    n = 3000

    def make(scale):
        d = pd.DataFrame({f"eta_{r}": rng.normal(size=n) for r in RULERS})
        d["zeta_scout"] = rng.normal(size=n)
        d["undrafted"] = (rng.random(n) < 0.3).astype(float)
        d["multi_level"] = (rng.random(n) < 0.2).astype(float)
        d["age_rel"] = rng.normal(size=n)
        d["log_pa"] = rng.normal(size=n)
        d["lvl"] = rng.integers(3, 7, n)
        beta = pd.Series({"beta_contact[0]": 0.7, "beta_power[0]": 0.4, "beta_speed[0]": 0.1, "beta_discipline[0]": -0.3, "beta_scout": 0.4, "undrafted": -0.8, "multi_level": -0.5})
        lin = -0.5 + scale * offset_from(d, beta) + 0.3 * d["log_pa"]
        d["y"] = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
        return d, beta

    d1, beta = make(1.0)
    d2, _ = make(1.0)
    d3, _ = make(0.5)
    e_same = evaluate(d2, offset_from(d2, beta))
    e_half = evaluate(d3, offset_from(d3, beta))
    assert abs(e_same["折扣係數"] - 1) < 0.1, e_same
    assert abs(e_half["折扣係數"] - 0.5) < 0.1, e_half
    assert e_same["固定權重 log-loss"] < e_same["基準 log-loss"] - 0.05
    print(f"同權重折扣 {e_same['折扣係數']:.2f}，半權重折扣 {e_half['折扣係數']:.2f}")
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="參考季的權重搬到其他季還能預測嗎")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("--promoted", default="promoted_f4sed_linear_p2")
    p.add_argument("--ref", type=int, nargs="*", default=[2010, 2015, 2019, 2023])
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        main(a.out_dir, a.data_dir, a.promoted, a.ref)
