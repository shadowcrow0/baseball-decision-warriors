"""用後驗比較兩個理論：同一季、兩個 tag 的 M1，誰的構念量得準、球隊的決定跟著誰。

    python src/compare_fits.py outputs/ 2024 promoted_f5sed_linear_p2 promoted_f5sedS_linear_p2
    python src/compare_fits.py --selftest

A = 第一個 tag（例如成績單指標），B = 第二個 tag（例如加了 Statcast 指標）。比四件事：
  1. 負荷量：兩邊每個指標的後驗平均與 89% 區間並排；B 多出來的指標載到哪把尺、多大。
  2. 結構係數：每把尺對升級（或登上、WAR）的 β，A 與 B 的後驗並排，加 B − A 的後驗（兩個獨立後驗的差，抽樣兩兩相減）與 P(B > A)。
     這就是「用貝氏統計去比理論」：不是看點估計差多少，是看差的整個分布有沒有離開 0。
  3. 每個球員：同一個人在 A 與 B 的 η 後驗平均的相關；移動最多的人（誰在成績單上被看高或看低）。
  4. 球團：若兩邊都有 u_org（球團截距）或 beta_type_org（--club-type 的每隊型態斜率），並排每隊的後驗。
  5. 樣本一樣時（add 模式）：兩邊升級段的 LOO。
需要兩邊的 idata_M1.nc（Arc 上才有 netCDF）；eta_M1.parquet、lambda_M1.csv 兩邊都要有。輸出 outputs/<season>/compare_<A>_vs_<B>.md。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fit_year import FACTOR_NAMES
from bf import bf_str, log10_bf10_draws

PREFIX_LABEL = {"": "升級", "t_": "幾年登上（負 = 快）", "w_": "WAR"}


def loadings_side_by_side(dir_a: Path, dir_b: Path) -> pd.DataFrame:
    """lambda_M1.csv 兩邊並排。"""
    a = pd.read_csv(dir_a / "lambda_M1.csv", index_col=0)
    b = pd.read_csv(dir_b / "lambda_M1.csv", index_col=0)
    fmt = lambda t: t.apply(lambda r: f"{r['mean']:+.2f} [{r['eti89_lb']:+.2f}, {r['eti89_ub']:+.2f}]", axis=1)  # noqa: E731
    out = pd.DataFrame({"尺": b["factor"].combine_first(a["factor"]), "A": fmt(a), "B": fmt(b)})
    out.loc[out["A"].isna(), "A"] = "（A 沒有）"
    out.loc[out["B"].isna(), "B"] = "（B 沒有）"
    return out.sort_values("尺").reset_index().rename(columns={"index": "指標"})


def beta_difference(post_a, post_b, factors: int) -> pd.DataFrame:
    """每把尺、每段的 β：A、B 的後驗，B − A 的後驗（兩邊的抽樣攤平後配對相減），P(B > A)。只拿一次項。"""
    rows = []
    for prefix, label in PREFIX_LABEL.items():
        for k in range(factors):
            v = f"{prefix}beta_{FACTOR_NAMES[k]}"
            if v not in post_a.data_vars or v not in post_b.data_vars:
                continue
            a = post_a[v].values[..., 0].ravel()
            b = post_b[v].values[..., 0].ravel()
            n = min(len(a), len(b))
            d = b[:n] - a[:n]
            rows.append(
                {
                    "段": label,
                    "尺": FACTOR_NAMES[k],
                    "A": f"{a.mean():+.2f} [{np.quantile(a, 0.055):+.2f}, {np.quantile(a, 0.945):+.2f}]",
                    "B": f"{b.mean():+.2f} [{np.quantile(b, 0.055):+.2f}, {np.quantile(b, 0.945):+.2f}]",
                    "B − A": f"{d.mean():+.2f} [{np.quantile(d, 0.055):+.2f}, {np.quantile(d, 0.945):+.2f}]",
                    "P(B > A)": round(float((d > 0).mean()), 2),
                    "BF10（差 ≠ 0）": bf_str(log10_bf10_draws(d, np.sqrt(2.0))),  # 兩個 β 的先驗各 N(0,1)，差的先驗 N(0, 2)
                }
            )
    return pd.DataFrame(rows)


def player_shift(dir_a: Path, dir_b: Path, factors: int, top: int = 10) -> tuple:
    """同一個人兩邊 η 的相關；移動最多的人。"""
    a = pd.read_parquet(dir_a / "eta_M1.parquet")
    b = pd.read_parquet(dir_b / "eta_M1.parquet")
    cols = [f"eta_{FACTOR_NAMES[k]}" for k in range(factors) if f"eta_{FACTOR_NAMES[k]}" in a.columns and f"eta_{FACTOR_NAMES[k]}" in b.columns]
    d = a[["key_mlbam", "season", "level_main"] + cols].merge(b[["key_mlbam"] + cols], on="key_mlbam", suffixes=("_A", "_B"))
    corr = pd.DataFrame(
        [{"尺": c.replace("eta_", ""), "A 與 B 的相關": round(float(np.corrcoef(d[c + "_A"], d[c + "_B"])[0, 1]), 3), "人數": len(d)} for c in cols]
    )
    d["移動"] = np.sqrt(sum((d[c + "_B"] - d[c + "_A"]) ** 2 for c in cols))
    movers = d.sort_values("移動", ascending=False).head(top)
    show = movers[["key_mlbam", "level_main", "移動"] + [c + s for c in cols for s in ("_A", "_B")]].round(2)
    return corr, show


def org_side_by_side(post_a, post_b, var: str) -> pd.DataFrame | None:
    """球團層變數（u_org、beta_type_org）兩邊並排；只要其中一邊有就印那一邊。"""
    if var not in post_a.data_vars and var not in post_b.data_vars:
        return None
    rows = []
    n = max(post_a[var].shape[-1] if var in post_a.data_vars else 0, post_b[var].shape[-1] if var in post_b.data_vars else 0)
    for o in range(n):
        row = {"球團索引": o}
        for name, post in (("A", post_a), ("B", post_b)):
            if var in post.data_vars and o < post[var].shape[-1]:
                x = post[var].values[..., o].ravel()
                row[name] = f"{x.mean():+.2f} [{np.quantile(x, 0.055):+.2f}, {np.quantile(x, 0.945):+.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def loo_both(idata_a, idata_b, var: str) -> pd.DataFrame | None:
    import arviz as az

    try:
        la, lb = az.loo(idata_a, var_name=var), az.loo(idata_b, var_name=var)
    except Exception as e:  # noqa: BLE001
        print(f"  LOO 算不出來（{type(e).__name__}），樣本可能不同", flush=True)
        return None
    def elpd(x):  # arviz 1.x 叫 elpd，0.x 叫 elpd_loo
        return float(getattr(x, "elpd", getattr(x, "elpd_loo", np.nan)))

    rows = [{"模型": "A", "elpd_loo": round(elpd(la), 1), "se": round(float(la.se), 1)}, {"模型": "B", "elpd_loo": round(elpd(lb), 1), "se": round(float(lb.se), 1)}]
    rows.append({"模型": "B − A", "elpd_loo": round(elpd(lb) - elpd(la), 1), "se": f"log10 pseudo-BF {(elpd(lb) - elpd(la)) / np.log(10):+.1f}"})
    return pd.DataFrame(rows)


def main(out_dir: str, season: int, tag_a: str, tag_b: str) -> str:
    import arviz as az

    out = Path(out_dir) / str(season)
    dir_a, dir_b = out / tag_a, out / tag_b
    idata_a, idata_b = az.from_netcdf(dir_a / "idata_M1.nc"), az.from_netcdf(dir_b / "idata_M1.nc")
    post_a, post_b = idata_a.posterior, idata_b.posterior
    factors = post_b["eta"].shape[-1]
    md = [
        f"# 用後驗比兩個理論 · {season} · A = {tag_a} · B = {tag_b}\n",
        "## 一、負荷量（後驗平均 [89% 區間]）\n",
        loadings_side_by_side(dir_a, dir_b).to_markdown(index=False),
        "",
        "## 二、結構係數：每把尺的 β，與 B − A 的後驗\n",
        "B − A 的區間不含 0，才算兩個理論對「球隊看什麼」有不同的說法；P(B > A) 接近 0 或 1 同義。\n",
        beta_difference(post_a, post_b, factors).to_markdown(index=False),
        "",
    ]
    corr, movers = player_shift(dir_a, dir_b, factors)
    md += ["## 三、每個球員：同一個人兩邊的 η\n", corr.to_markdown(index=False), "", "移動最多的十個人（成績單與進階量看法最不同的人）：\n", movers.to_markdown(index=False), ""]
    for var, label in (("u_org", "球團截距 u_org（升級段）"), ("beta_type_org", "每隊對型態的斜率 beta_type_org（--club-type）")):
        t = org_side_by_side(post_a, post_b, var)
        if t is not None:
            md += [f"## 四、球團：{label}\n", t.to_markdown(index=False), ""]
    ll_var = "promoted" if "promoted" in getattr(idata_a, "log_likelihood", {}) else None
    if ll_var and post_a["eta"].shape[-2] == post_b["eta"].shape[-2]:
        t = loo_both(idata_a, idata_b, ll_var)
        if t is not None:
            md += ["## 五、LOO（同一批人）\n", t.to_markdown(index=False), ""]
    text = "\n".join(md)
    path = out / f"compare_{tag_a}_vs_{tag_b}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """不需要 netCDF 的部分：負荷量並排、球員移動、β 差的後驗（用 xarray 假後驗）。"""
    import xarray as xr

    root = Path(tempfile.mkdtemp())
    a, b = root / "A", root / "B"
    a.mkdir(), b.mkdir()
    lam = pd.DataFrame({"factor": ["contact", "power"], "mean": [-0.5, 0.4], "eti89_lb": [-0.6, 0.3], "eti89_ub": [-0.4, 0.5]}, index=["Kpct", "HR_AB"])
    lam.to_csv(a / "lambda_M1.csv")
    lam2 = pd.concat([lam, pd.DataFrame({"factor": ["power"], "mean": [0.8], "eti89_lb": [0.7], "eti89_ub": [0.9]}, index=["EV90"])])
    lam2.to_csv(b / "lambda_M1.csv")
    rng = np.random.default_rng(0)
    e = pd.DataFrame({"key_mlbam": range(50), "season": 2024, "level_main": "AAA", "eta_contact": rng.normal(size=50), "eta_power": rng.normal(size=50)})
    e.to_parquet(a / "eta_M1.parquet", index=False)
    e2 = e.copy()
    e2["eta_power"] = 0.8 * e2["eta_power"] + 0.6 * rng.normal(size=50)
    e2.to_parquet(b / "eta_M1.parquet", index=False)
    tab = loadings_side_by_side(a, b)
    assert (tab.loc[tab["指標"] == "EV90", "A"] == "（A 沒有）").all()
    corr, movers = player_shift(a, b, 2)
    assert corr.loc[corr["尺"] == "contact", "A 與 B 的相關"].iloc[0] > 0.99 and len(movers) == 10
    pa = xr.Dataset({"beta_contact": (("chain", "draw", "b"), rng.normal(0.5, 0.1, (2, 100, 1))), "beta_power": (("chain", "draw", "b"), rng.normal(0.3, 0.1, (2, 100, 1)))})
    pb = xr.Dataset({"beta_contact": (("chain", "draw", "b"), rng.normal(0.5, 0.1, (2, 100, 1))), "beta_power": (("chain", "draw", "b"), rng.normal(0.9, 0.1, (2, 100, 1)))})
    bd = beta_difference(pa, pb, 2)
    assert bd.loc[bd["尺"] == "power", "P(B > A)"].iloc[0] > 0.95 and 0.2 < bd.loc[bd["尺"] == "contact", "P(B > A)"].iloc[0] < 0.8
    print(tab.to_string(index=False))
    print(bd.to_string(index=False))
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="同一季兩個 tag 的 M1 後驗並排：負荷量、β 差的後驗、球員移動、球團")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("season", nargs="?", type=int)
    p.add_argument("tag_a", nargs="?")
    p.add_argument("tag_b", nargs="?")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif not (a.season and a.tag_a and a.tag_b):
        p.error("要 season、tag_a、tag_b（或 --selftest）")
    else:
        main(a.out_dir, a.season, a.tag_a, a.tag_b)
