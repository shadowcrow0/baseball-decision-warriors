"""主模型結構係數的 Bayes factor：每季每把尺「β = 0」vs「β ≠ 0」，Savage–Dickey 從 idata 的後驗抽樣算，再跨季合併。

    python src/bayes_factors.py outputs/ --promoted promoted_f4sed_linear_p2 [--war war_f4sed_linear_p2]
    python src/bayes_factors.py --selftest

每季：β 的先驗是 N(0, 1)（fit_year 的 feta linear），後驗在 0 的密度用核密度估，BF10 = 先驗在 0 ÷ 後驗在 0。
跨季：各季獨立，log10 BF 相加 = 「十七季全部為零」vs「十七季全部不為零」的 BF；另給中位數（典型的一季）。
要 idata_M1.nc（Arc）。輸出 outputs/bayes_factors_<tag>.md。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bf import bf_str, label, log10_bf10_draws  # noqa: E402
from fit_year import FACTOR_NAMES  # noqa: E402

PRIOR_SD = {"beta": 1.0, "beta_scout": 1.0, "a_club_war": 1.0, "beta_type_x_club": 1.0, "b_cov": 1.0}
PREFIX_LABEL = {"": "升級", "t_": "幾年登上", "w_": "WAR|登上"}


def season_table(post, prefixes: tuple) -> dict:
    """一季：{變數標籤: (後驗平均, log10 BF10)}。"""
    out = {}
    for pre in prefixes:
        if f"{pre}lin" not in post.data_vars and f"{pre}beta_scout" not in post.data_vars:
            continue
        for k, name in FACTOR_NAMES.items():
            v = f"{pre}beta_{name}"
            if v in post.data_vars:
                x = post[v].values[..., 0].ravel()
                out[f"{PREFIX_LABEL[pre]} · {name}"] = (float(x.mean()), log10_bf10_draws(x, PRIOR_SD["beta"]))
        for v, lab in ((f"{pre}beta_scout", "球探評價"), (f"{pre}a_club_war", "母隊 WAR（第二層）"), (f"{pre}beta_type_x_club", "母隊強弱 × 型態")):
            if v in post.data_vars:
                x = post[v].values.ravel()
                out[f"{PREFIX_LABEL[pre]} · {lab}"] = (float(x.mean()), log10_bf10_draws(x, 1.0))
        if f"{pre}b_cov" in post.data_vars:  # 共變數：非選秀、年資等（名字在 struct_M1.csv，這裡只標索引）
            pass
    return out


def main(out_dir: str, tags: list) -> str:
    import arviz as az

    out = Path(out_dir)
    rows = []
    for tag in tags:
        prefixes = ("",) if tag.startswith("promoted") else ("t_", "w_")
        for p in sorted(out.glob(f"20*/{tag}/idata_M1.nc")):
            season = int(p.parent.parent.name)
            post = az.from_netcdf(p).posterior
            for var, (mean, l10) in season_table(post, prefixes).items():
                rows.append({"tag": tag, "季": season, "變數": var, "後驗平均": mean, "log10 BF10": l10})
            print(f"  {tag} {season}", flush=True)
    if not rows:
        raise SystemExit("找不到 idata_M1.nc")
    d = pd.DataFrame(rows)
    wide = d.pivot_table(index="變數", columns="季", values="log10 BF10").round(1)
    summ = d.groupby("變數").agg(季數=("季", "count"), 平均β=("後驗平均", "mean"), log10BF_總和=("log10 BF10", "sum"), log10BF_中位=("log10 BF10", "median"), 幾季BF大於10=("log10 BF10", lambda x: int((x >= 1).sum())), 幾季支持零=("log10 BF10", lambda x: int((x <= -0.5).sum()))).round(2)
    summ["總和的 BF"] = summ["log10BF_總和"].map(bf_str)
    summ["典型一季"] = summ["log10BF_中位"].map(label)
    md = [
        f"# 結構係數的 Bayes factor · {', '.join(tags)}\n",
        "每季 Savage–Dickey：先驗 N(0, 1)，後驗在 0 的密度用核密度估；log10 BF10 > 1 = 「有效果」的證據是「沒效果」的十倍以上，< −0.5 = 支持零。"
        "跨季總和 = 所有季一起的 BF（各季獨立）；上限每季記 6。\n",
        "## 跨季合併\n",
        summ.to_markdown(),
        "",
        "## 每季的 log10 BF10\n",
        wide.to_markdown(),
        "",
    ]
    text = "\n".join(md)
    path = out / f"bayes_factors_{'_'.join(t.split('_')[0] for t in tags)}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    import xarray as xr

    rng = np.random.default_rng(0)
    post = xr.Dataset(
        {
            "lin": (("chain", "draw", "n"), rng.normal(size=(2, 500, 10))),
            "beta_contact": (("chain", "draw", "b"), rng.normal(0.7, 0.07, (2, 500, 1))),
            "beta_discipline": (("chain", "draw", "b"), rng.normal(0.0, 0.05, (2, 500, 1))),
            "beta_power": (("chain", "draw", "b"), rng.normal(0.1, 0.3, (2, 500, 1))),
            "beta_scout": (("chain", "draw"), rng.normal(0.5, 0.1, (2, 500))),
        }
    )
    t = season_table(post, ("",))
    assert t["升級 · contact"][1] >= 2, t
    assert t["升級 · discipline"][1] < -0.5, t
    assert -1 < t["升級 · power"][1] < 1, t  # 0.1 ± 0.3：分不出來，兩邊都不該強
    print({k: (round(v[0], 2), round(v[1], 1)) for k, v in t.items()})
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="結構係數的 Savage–Dickey Bayes factor，逐季與跨季")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("--promoted", default="promoted_f4sed_linear_p2")
    p.add_argument("--war", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        main(a.out_dir, [t for t in (a.promoted, a.war) if t])
