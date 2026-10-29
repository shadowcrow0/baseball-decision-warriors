"""後驗預測檢查（posterior predictive check）：模型假設的分布跟實際資料合不合。

    python src/ppc.py outputs/ data/ 2015 promoted_f4sed_linear_p2 [--statcast add] [--draws 200]
    python src/ppc.py --selftest

做法：從 M1 的後驗抽 200 組參數，每組照模型把每個人的指標重新「生」一次（二項：三振 ~ Binomial(打席, p)；
負二項：全壘打 ~ NegBin(打數 × 率, φ)；常態：EV90 ~ Normal(線性預測, σ/√(進場球/100))），得到 200 份「模型以為的資料」，
跟真實資料比四件事（每個指標一列）：
  1. 跨人的比率分布：真實的 1/5/25/50/75/95/99 百分位 vs 模型生的（看位置、尾巴對不對）。
  2. 過度離散：Pearson 卡方 T = Σ (y − 期望)² / 模型變異數，真實 vs 模型生的；貝氏 p = P(T_rep ≥ T_obs)。
     p 接近 0 = 真實比模型假設的更散（二項太窄，該用 beta-binomial）；接近 1 = 真實比模型更集中。
  3. 零的比例：真實有幾成的人這個計數是 0 vs 模型生的（零膨脹）。
  4. 極端值：真實的最大比率 vs 模型生的最大值的分布（P(max_rep ≥ max_obs)）。
結構段：升級 = 校準（預測十分位的真實升級率 vs 模型生的）；WAR tag = 登上後 asinh(WAR) 的分位數、≤ 0 的比例，
與幾年登上的分布（1–6 年與設限）真實 vs 模型生的。
圖：每個指標一格，黑線 = 真實跨人分布，紅線 = 20 份模型生的分布。
輸出 outputs/<season>/<tag>/ppc.md、outputs/figures/ppc_<season>_<tag>.png。需要 idata_M1.nc（Arc）。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.special import expit  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
import fit_year as fy  # noqa: E402

QS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def flat(post: xr.Dataset, var: str, idx: np.ndarray) -> np.ndarray:
    """(chain, draw, ...) → 攤平後取 idx 那些 draw → (S, ...)。"""
    a = post[var].values
    a = a.reshape((-1,) + a.shape[2:])
    return a[idx]


def replicate_indicator(post: xr.Dataset, m: pd.DataFrame, name: str, num: str, den: str, kind: str, k: int, idx: np.ndarray, rng: np.random.Generator) -> dict | None:
    """一個指標：真實 y、分母 d、模型生的 y_rep (S, n_ok)、期望與變異數（算卡方用）。"""
    tag = fy.indicator_tag(name)
    if f"lambda_{tag}" not in post.data_vars or num not in m.columns:
        return None
    y = m[num].to_numpy(dtype=float)
    d = m[den].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(d) & (d > 0)
    if kind == "binom":
        ok &= y <= d
    rows = np.where(ok)[0]
    if len(rows) == 0:
        return None
    lvl = m["lvl"].to_numpy()[rows]
    gamma = flat(post, f"gamma_{tag}", idx)  # (S, N_LEVELS)
    lam = flat(post, f"lambda_{tag}", idx)  # (S,)
    eta = flat(post, "eta", idx)  # (S, n, K)
    lin = gamma[:, lvl] + lam[:, None] * eta[:, rows, k]
    for var in post.data_vars:  # 交叉負荷 lambda_<tag>_x<factor>
        if var.startswith(f"lambda_{tag}_x"):
            k2 = {v: kk for kk, v in fy.FACTOR_NAMES.items()}[var.split("_x")[1]]
            lin = lin + flat(post, var, idx)[:, None] * eta[:, rows, k2]
    y, d = y[rows], d[rows]
    if kind == "binom":
        p = expit(lin)
        mean, var = d * p, d * p * (1 - p)
        rep = rng.binomial(d.astype(int), p)
    elif kind == "negbin":
        phi = flat(post, f"phi_{tag}", idx)[:, None]
        mu = np.exp(lin) * d
        mean, var = mu, mu + mu**2 / phi
        rep = rng.poisson(rng.gamma(phi * np.ones_like(mu), mu / phi))
    else:  # normal（Statcast 連續量，已標準化）
        sigma = flat(post, f"sigma_{tag}", idx)[:, None] * np.sqrt(100.0 / d)
        mean, var = lin, sigma**2
        rep = rng.normal(lin, sigma)
    return {"name": name, "kind": kind, "y": y, "d": d, "rep": rep, "mean": mean, "var": var}


def summarize(r: dict) -> dict:
    """一個指標的四個檢查。比率 = y / d（常態指標直接用值）。"""
    y, d, rep, mean, var = r["y"], r["d"], r["rep"], r["mean"], r["var"]
    obs_rate = y / d if r["kind"] != "normal" else y
    rep_rate = rep / d if r["kind"] != "normal" else rep
    q_obs = np.quantile(obs_rate, QS)
    q_rep = np.quantile(rep_rate, QS, axis=1).mean(axis=1)
    t_obs = (((y - mean) ** 2) / np.maximum(var, 1e-9)).sum(axis=1)  # (S,)：真實 y 對每組參數的卡方
    t_rep = (((rep - mean) ** 2) / np.maximum(var, 1e-9)).sum(axis=1)
    p_disp = float((t_rep >= t_obs).mean())
    zeros_obs, zeros_rep = float((y == 0).mean()), float((rep == 0).mean())
    p_max = float((rep_rate.max(axis=1) >= obs_rate.max()).mean())
    ks = float(np.mean([np.abs(np.searchsorted(np.sort(rr), np.sort(obs_rate), side="right") / len(rr) - (np.arange(1, len(obs_rate) + 1) / len(obs_rate))).max() for rr in rep_rate[:20]]))
    return {
        "指標": r["name"],
        "似然": {"binom": "二項", "negbin": "負二項", "normal": "常態"}[r["kind"]],
        "人季": len(y),
        "真實分位數 1/5/25/50/75/95/99": " ".join(f"{v:.3f}" for v in q_obs),
        "模型生的分位數": " ".join(f"{v:.3f}" for v in q_rep),
        "KS": round(ks, 3),
        "離散比 T_obs/T_rep": round(float(t_obs.mean() / max(t_rep.mean(), 1e-9)), 2),
        "p（離散）": round(p_disp, 3),
        "零的比例 真實/模型": f"{zeros_obs:.3f} / {zeros_rep:.3f}",
        "p（最大值）": round(p_max, 3),
    }


def verdict(row: dict) -> str:
    out = []
    if row["p（離散）"] < 0.05 and row["離散比 T_obs/T_rep"] > 1.1:  # 人多時 p 很嚴，離散比超過 1.1 才算實際上有差
        out.append(f"真實比模型散 {row['離散比 T_obs/T_rep']} 倍（過度離散）")
    elif row["p（離散）"] > 0.95 and row["離散比 T_obs/T_rep"] < 0.9:
        out.append("真實比模型集中")
    if row["p（最大值）"] < 0.05:
        out.append("真實的極端值模型生不出來")
    if row["KS"] > 0.05:
        out.append(f"跨人分布形狀差 KS {row['KS']}")
    return "；".join(out) if out else "合"


def plot(results: list, path: Path, title: str) -> None:
    n = len(results)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        obs = r["y"] / r["d"] if r["kind"] != "normal" else r["y"]
        reps = r["rep"] / r["d"] if r["kind"] != "normal" else r["rep"]
        lo, hi = np.quantile(np.concatenate([obs, reps[:20].ravel()]), [0.001, 0.999])
        bins = np.linspace(lo, hi, 40)
        for rr in reps[:20]:
            ax.hist(rr, bins=bins, density=True, histtype="step", color="#c0392b", alpha=0.25, lw=0.8)
        ax.hist(obs, bins=bins, density=True, histtype="step", color="black", lw=1.6)
        ax.set_title(f"{r['name']}（{ {'binom': '二項', 'negbin': '負二項', 'normal': '常態'}[r['kind']] }）", fontsize=10)
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title + "　黑 = 真實跨人分布，紅 = 20 份模型生的", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def structural_checks(post: xr.Dataset, m: pd.DataFrame, s_idx: np.ndarray, outcome: str, idx: np.ndarray, rng: np.random.Generator) -> list:
    """結構段：升級的校準；WAR tag 的 asinh(WAR) 分布與幾年登上的分布。回傳 markdown 段落。"""
    md = []
    if outcome == "promoted" and "lin" in post.data_vars:
        p = expit(flat(post, "lin", idx))  # (S, n_s)
        y = m["promoted"].to_numpy()[s_idx].astype(float)
        pm = p.mean(axis=0)
        dec = pd.qcut(pd.Series(pm).rank(method="first"), 10, labels=False)
        rows = []
        for q in range(10):
            sel = dec.to_numpy() == q
            rep_rate = rng.binomial(1, p[:, sel]).mean(axis=1)
            rows.append({"預測十分位": q + 1, "人數": int(sel.sum()), "模型預測": round(float(pm[sel].mean()), 3), "真實升級率": round(float(y[sel].mean()), 3), "模型生的 5%–95%": f"{np.quantile(rep_rate, 0.05):.3f}–{np.quantile(rep_rate, 0.95):.3f}"})
        md += ["## 結構段：升級的校準（真實升級率要落在模型生的 5%–95% 之間）\n", pd.DataFrame(rows).to_markdown(index=False), ""]
    if outcome == "war" and "w_lin" in post.data_vars:
        reached = m["reached_fwd"].to_numpy()[s_idx].astype(bool)
        r_idx = np.where(reached)[0]
        lin_w = flat(post, "w_lin", idx)[:, r_idx]
        nu = flat(post, "nu", idx)[:, None]
        sw = flat(post, "sigma_w", idx)[:, None]
        rep = lin_w + sw * rng.standard_t(nu * np.ones_like(lin_w))
        obs = np.arcsinh(m["war_fwd"].to_numpy(dtype=float)[s_idx][r_idx])
        rows = [{"統計量": f"第 {int(q * 100)} 百分位", "真實": round(float(np.quantile(obs, q)), 2), "模型生的（平均）": round(float(np.quantile(rep, q, axis=1).mean()), 2)} for q in QS]
        rows.append({"統計量": "WAR ≤ 0 的比例", "真實": round(float((obs <= 0).mean()), 3), "模型生的（平均）": round(float((rep <= 0).mean()), 3)})
        rows.append({"統計量": "最大值", "真實": round(float(obs.max()), 2), "模型生的（平均）": round(float(rep.max(axis=1).mean()), 2)})
        md += ["## 結構段：登上者 asinh(WAR) 的分布（StudentT）\n", pd.DataFrame(rows).to_markdown(index=False), ""]
        lin_t = flat(post, "t_lin", idx)
        st = flat(post, "sigma_t", idx)[:, None]
        T_rep = np.exp(lin_t + st * rng.standard_normal(lin_t.shape))
        T_obs = m["T_debut"].to_numpy(dtype=float)[s_idx]
        rows = []
        for yr in range(1, 7):
            rows.append({"幾年登上": yr, "真實比例": round(float(((T_obs >= yr) & (T_obs < yr + 1)).mean()), 3), "模型生的": round(float(((T_rep >= yr) & (T_rep < yr + 1)).mean()), 3)})
        rows.append({"幾年登上": "六季內沒登上（設限）", "真實比例": round(float((T_obs >= fy.DEBUT_CENSOR).mean()), 3), "模型生的": round(float((T_rep >= 7).mean()), 3)})
        md += ["## 結構段：幾年登上（log-normal 加速失敗時間）\n", pd.DataFrame(rows).to_markdown(index=False), ""]
    return md


def run(post: xr.Dataset, m: pd.DataFrame, factors: int, statcast: str, n_draws: int, seed: int = 0) -> tuple:
    """測量模型的 PPC：回傳 (summary DataFrame, results list)。"""
    rng = np.random.default_rng(seed)
    total = post.sizes["chain"] * post.sizes["draw"]
    idx = rng.choice(total, size=min(n_draws, total), replace=False)
    results = []
    for name, num, den, kind, k in fy.assign_indicators(factors, statcast):
        r = replicate_indicator(post, m, name, num, den, kind, k, idx, rng)
        if r is not None:
            results.append(r)
    rows = [summarize(r) for r in results]
    for row in rows:
        row["判定"] = verdict(row)
    return pd.DataFrame(rows), results, idx, rng


def main(out_dir: str, data_dir: str, season: int, tag: str, statcast: str, n_draws: int) -> str:
    import arviz as az

    out = Path(out_dir)
    d = out / str(season) / tag
    idata = az.from_netcdf(d / "idata_M1.nc")
    post = idata.posterior
    factors = post["eta"].shape[-1]
    outcome = "war" if tag.startswith("war") else "promoted"
    m = fy.load(Path(data_dir), season, club=False, statcast=statcast)
    mask = (m["structural"] if outcome == "promoted" else m["war_structural"]).to_numpy()
    s_idx = np.where(mask)[0]
    if post["eta"].shape[-2] != len(m):
        raise SystemExit(f"後驗的人數 {post['eta'].shape[-2]} 跟建模表 {len(m)} 不合：--statcast 要跟 fit 一樣")
    tab, results, idx, rng = run(post, m, factors, statcast, n_draws)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig = figdir / f"ppc_{season}_{tag}.png"
    plot(results, fig, f"{season} {tag}")
    md = [
        f"# 後驗預測檢查 · {season} · {tag}\n",
        f"從 M1 後驗抽 {len(idx)} 組參數，照模型把每個指標重新生一次，跟真實比。圖：`{fig.relative_to(out)}`。\n",
        "讀法：**p（離散）** 接近 0 = 真實比二項／負二項假設的更散（該換 beta-binomial 或放寬 φ）；離散比 = 真實卡方 ÷ 模型生的卡方，1 是剛好；人多時 p 很嚴，判定以離散比超過 1.1 為準。"
        "**p（最大值）** 接近 0 = 真實有模型生不出來的極端人。**KS** = 跨人比率分布的最大落差（20 份平均）。零的比例看零膨脹。\n",
        "## 測量模型：十五個指標（＋Statcast）\n",
        tab.to_markdown(index=False),
        "",
    ]
    md += structural_checks(post, m, s_idx, outcome, idx, rng)
    text = "\n".join(md)
    (d / "ppc.md").write_text(text)
    print(text, flush=True)
    print(f"-> {d / 'ppc.md'}", flush=True)
    return text


def selftest() -> None:
    """合成：三振數真的從二項生 → p（離散）在 0.05–0.95；另一個指標從 beta-binomial 生（過度離散）→ p < 0.05。"""
    rng = np.random.default_rng(5)
    n, S = 600, 150
    lvl = rng.integers(3, 7, n)
    eta = rng.normal(size=(n, 4))  # 四把尺：K% 載在第 0 把、BB% 載在第 3 把
    gamma_true = np.array([0, 0, 0, -1.2, -1.3, -1.4, -1.5])
    pa = rng.integers(150, 500, n)
    p_k = expit(gamma_true[lvl] - 0.4 * eta[:, 0])
    so = rng.binomial(pa, p_k)
    # 過度離散的保送：每人的 p 再加 beta 噪音
    p_bb = expit(-2.0 + 0.4 * eta[:, 3])
    p_bb_noisy = rng.beta(p_bb * 8, (1 - p_bb) * 8)
    bb = rng.binomial(pa, p_bb_noisy)
    m = pd.DataFrame({"SO": so, "PA": pa, "BB": bb, "lvl": lvl, "promoted": rng.random(n) < 0.3, "structural": True})
    chains, draws = 2, S // 2
    post = xr.Dataset(
        {
            "eta": (("chain", "draw", "n", "k"), eta[None, None] + rng.normal(0, 0.01, (chains, draws, n, 4))),
            "gamma_Kpct": (("chain", "draw", "l"), gamma_true[None, None] + rng.normal(0, 0.005, (chains, draws, 7))),
            "lambda_Kpct": (("chain", "draw"), -0.4 + rng.normal(0, 0.005, (chains, draws))),
            "gamma_BBpct": (("chain", "draw", "l"), np.full((chains, draws, 7), -2.0) + rng.normal(0, 0.02, (chains, draws, 7))),
            "lambda_BBpct": (("chain", "draw"), 0.4 + rng.normal(0, 0.02, (chains, draws))),
        }
    )
    tab, results, _, _ = run(post, m, 4, "none", 100)
    print(tab[["指標", "離散比 T_obs/T_rep", "p（離散）", "p（最大值）", "KS", "判定"]].to_string(index=False))
    k = tab.set_index("指標")
    # 合成的「後驗」沒有對資料做條件化，所以真實資料的卡方比真正的後驗預測稍大；看離散比與判定，不看嚴格的 p
    assert k.loc["K%", "離散比 T_obs/T_rep"] < 1.15 and k.loc["K%", "判定"] == "合", f"二項生的資料被判不合：{k.loc['K%'].to_dict()}"
    assert k.loc["BB%", "p（離散）"] < 0.05 and k.loc["BB%", "離散比 T_obs/T_rep"] > 1.5, "過度離散沒抓到"
    root = Path(tempfile.mkdtemp())
    plot(results, root / "ppc.png", "selftest")
    assert (root / "ppc.png").exists()
    print("selftest OK", flush=True)


def main_all(out_dir: str, data_dir: str, tag: str, statcast: str, n_draws: int) -> str:
    """該 tag 有 idata_M1.nc 的每一季各跑一次，再彙整：每個指標的離散比、KS、p（最大值）逐季列成一張表。"""
    import re

    out = Path(out_dir)
    seasons = sorted(int(p.parent.parent.name) for p in out.glob(f"20*/{tag}/idata_M1.nc"))
    rows = []
    for season in seasons:
        print(f"== {season}", flush=True)
        try:
            text = main(out_dir, data_dir, season, tag, statcast, n_draws)
        except SystemExit as e:
            print(f"  跳過：{e}", flush=True)
            continue
        table = text.split("## 測量模型")[1].split("## 結構段")[0] if "## 結構段" in text else text.split("## 測量模型")[1]
        for line in table.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 10 and cells[0] not in ("指標", "") and not cells[0].startswith(":"):
                rows.append({"季": season, "指標": cells[0], "離散比": float(cells[6]), "KS": float(cells[5]), "p（最大值）": float(cells[9])})
    long = pd.DataFrame(rows)
    disp = long.pivot(index="指標", columns="季", values="離散比")
    disp["中位數"] = disp.median(axis=1).round(2)
    disp["幾季 > 1.1"] = (disp.drop(columns="中位數") > 1.1).sum(axis=1)
    ks = long.pivot(index="指標", columns="季", values="KS")
    ks["中位數"] = ks.median(axis=1).round(3)
    md = [
        f"# 後驗預測檢查彙整 · {tag} · {len(seasons)} 季\n",
        "## 離散比（真實卡方 ÷ 模型生的；> 1.1 = 真實比似然假設的更散）\n",
        disp.round(2).to_markdown(),
        "",
        "## KS（跨人比率分布的最大落差）\n",
        ks.round(3).to_markdown(),
        "",
    ]
    text = "\n".join(md)
    path = out / f"ppc_all_{tag}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="後驗預測檢查：模型假設的分布跟實際資料合不合")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("season", nargs="?", type=int)
    p.add_argument("tag", nargs="?")
    p.add_argument("--all", dest="all_seasons", default=None, help="給 tag：該 tag 有 idata 的每一季都跑，並彙整 outputs/ppc_all_<tag>.md")
    p.add_argument("--statcast", choices=["none", "add", "only"], default="none", help="跟那個 fit 用的一樣")
    p.add_argument("--draws", type=int, default=200, help="從後驗抽幾組參數")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.all_seasons:
        main_all(a.out_dir, a.data_dir, a.all_seasons, a.statcast, a.draws)
    elif not (a.season and a.tag):
        p.error("要 season 與 tag（或 --all <tag>，或 --selftest）")
    else:
        main(a.out_dir, a.data_dir, a.season, a.tag, a.statcast, a.draws)
