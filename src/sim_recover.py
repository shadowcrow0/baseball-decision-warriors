"""參數還原：從模型自己的生成式造一個假農場（真值已知），用 fit_year 的 M1 去配，看估不估得回來。每一種跑過的模型都要有。
也可以故意讓真相偏離模型假設（beta-binomial 的過度離散），看現在的模型會偏多少、區間涵蓋率掉到幾成。

    python src/sim_recover.py data/ outputs/ 2015 --rep 0                          # 升級段四把尺（主模型）
    python src/sim_recover.py data/ outputs/ 2015 --rep 0 --truth betabinom          # 同上，真相過度離散
    python src/sim_recover.py data/ outputs/ 2015 --rep 0 --outcome war              # WAR 段（幾年登上 + WAR|登上）
    python src/sim_recover.py data/ outputs/ 2015 --rep 0 --factors 5                # 五把尺
    python src/sim_recover.py data/ outputs/ 2024 --rep 0 --factors 5 --statcast same|only|both   # 進階量三種（只有 3A 人）
    python src/sim_recover.py data/ outputs/ 2015 --rep 0 --org-slopes               # 球團各自的斜率
    python src/sim_recover.py data/ outputs/ 2015 --rep 0 --club-type                # 母隊強弱 × 型態（要 club 資料）
    python src/sim_recover.py --summarize outputs/          # 彙整所有 outputs/sim/recover_*.json
    python src/sim_recover.py --selftest

一次一個複本（--rep k 換種子），一個複本一個 Arc job。做法：
  1. 拿真實那一季的共變數與分母（打席、打數、進場球…），其餘全部重生：
     η ~ MVN(球團效應 + 年齡效應, R)（R 對角外 0.3），負荷量 λ 用該季真實 fit 的後驗平均當真值（沒有就用常識符號 × 0.4），
     層級截距 γ 用真實比率的 logit；指標按模型似然生（二項 / 負二項 / 常態，常態的 sd = σ·√(100/進場球)）；
     球探 ζ ~ N(0,1)，順位與簽約金當它的兩個指標。
     升級段：升級 ~ Bernoulli(層級截距 + 共變數 + u_org + β_scout ζ + Σ β_k η_k)，β = (+0.7, +0.4, +0.1, −0.3[, +0.3])。
     WAR 段：log T ~ N(t_lin, σ_t = 0.6) 設限 7.5；asinh(WAR) ~ StudentT(ν = 5, w_lin, σ_w = 1.0) 只對登上的人；
             β_t = (−0.4, −0.2, −0.1, +0.2[, −0.2])，β_w = (+0.3, +0.4, 0.0, +0.1[, +0.3])。
     --org-slopes：每隊每把尺的斜率偏離 ~ N(0, 0.15²)；--club-type：β_type×club = −0.3、每隊型態斜率 ~ N(0, 0.15²)。
  2. --truth betabinom：每個人的比率先從 Beta(p·κ, (1−p)·κ) 抽再進二項；κ = 800 對 400 個打席約是真實卡方 1.5 倍（PPC 看到的量級）。
  3. 配 M1（跟該模型一樣的旗標；不加 --extra --direct，生成式裡沒有那些），鏡像對齊後比：
     每個 λ、β、β_scout、τ_org、σ_t、σ_w、σ_org 的後驗平均與 89% 區間 vs 真值（偏差、是否涵蓋）；η 與真值的相關；u_org、斜率偏離與真值的相關。
  4. --summarize 把所有複本按模型 × 真相彙整：每個參數的平均偏差、RMSE、89% 區間涵蓋率（名目 89%）。
寫 outputs/sim/recover_<season>_<model>_<truth>_rep<k>.json；彙整寫 outputs/sim_recover_summary.md。
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import fit_year as fy  # noqa: E402

BETA_TRUE = {"": [0.7, 0.4, 0.1, -0.3, 0.3], "t_": [-0.4, -0.2, -0.1, 0.2, -0.2], "w_": [0.3, 0.4, 0.0, 0.1, 0.3]}
BETA_SCOUT_TRUE = {"": 0.4, "t_": -0.3, "w_": 0.1}
TAU_ORG_TRUE = 0.3
SIGMA_T_TRUE, SIGMA_W_TRUE, NU_TRUE = 0.6, 1.0, 5.0
SIGMA_ORG_SLOPE_TRUE = 0.15  # --org-slopes：每隊斜率偏離的 sd
BETA_TYPE_CLUB_TRUE, SIGMA_TYPE_ORG_TRUE = -0.3, 0.15  # --club-type
SIGMA_NORMAL = 0.8  # 常態指標（EV90 等）在 100 顆進場球時的 sd
CORR_OFF = 0.3


def expit(x):
    return 1 / (1 + np.exp(-x))


def model_tag(outcome: str, factors: int, statcast: str, org_slopes: bool, club_type: bool) -> str:
    """跟 fit_year 的 tag 字母一致：promoted_f4s、war_f4s、promoted_f5sN、promoted_f4so、promoted_f4sk。"""
    return f"{outcome}_f{factors}s{'o' if org_slopes else ''}{'k' if club_type else ''}{fy.STATCAST_LETTER[statcast]}"


def true_lambdas(out: Path, season: int, tag: str, inds: list) -> dict:
    """真值負荷量：該季真實 fit 的後驗平均（lambda_M1.csv）；沒有就用常識符號 × 0.4。"""
    p = out / str(season) / tag / "lambda_M1.csv"
    lam = {}
    real = pd.read_csv(p, index_col=0)["mean"] if p.exists() else None
    for name, *_ in inds:
        t = fy.indicator_tag(name)
        lam[name] = float(real[t]) if real is not None and t in real.index else 0.4 * fy.INIT_SIGN[name]
    return lam


def level_logits(m: pd.DataFrame, s_idx: np.ndarray, col: str) -> np.ndarray:
    """結構段每個層級的基礎 logit（用真實的升級率）。"""
    lvl = m["lvl"].to_numpy()
    rate = m.loc[s_idx, col].astype(float).groupby(lvl[s_idx]).mean()
    g = np.zeros(fy.N_LEVELS)
    for l, r in rate.items():
        r = min(max(r, 0.05), 0.95)
        g[int(l)] = np.log(r / (1 - r))
    return g


def simulate(m: pd.DataFrame, out: Path, season: int, tag: str, truth: str, kappa: float, rng: np.random.Generator, outcome: str = "promoted", factors: int = 4, statcast: str = "none", org_slopes: bool = False, club_type: bool = False) -> tuple:
    """在真實的共變數與分母上重生指標與結果。回傳 (m 改好的, 真值 dict)。"""
    m = m.copy()
    n, K = len(m), factors
    inds = fy.assign_indicators(K, statcast)
    lam = true_lambdas(out, season, tag, inds)
    lvl = m["lvl"].to_numpy()
    org = m["org_i"].to_numpy()
    n_org = int(org.max()) + 1
    # η：球團效應 + 年齡 + 相關的殘差
    theta_org = rng.normal(0, 0.3, (n_org, K))
    delta = np.array([0.2, 0.1, -0.1, 0.1, 0.1])[:K]
    R = np.full((K, K), CORR_OFF) + np.eye(K) * (1 - CORR_OFF)
    eta = theta_org[org] + delta[None, :] * m["age_rel"].to_numpy()[:, None] + rng.multivariate_normal(np.zeros(K), R, n)
    # 球探 ζ 與它的兩個指標（順位、簽約金）；模型內部會各自標準化，所以尺度隨便
    drafted = m["drafted"].to_numpy().astype(bool)
    zeta = np.zeros(n)
    zeta[drafted] = rng.normal(0, 1, drafted.sum())
    m.loc[drafted, "log_pick"] = 5.0 - 0.8 * zeta[drafted] + rng.normal(0, 0.6, drafted.sum())
    m.loc[drafted, "pick_overall"] = np.exp(m.loc[drafted, "log_pick"])
    kb = drafted & m["bonus_known"].to_numpy().astype(bool)
    m.loc[kb, "log_bonus"] = 13.0 + 0.8 * zeta[kb] + rng.normal(0, 0.6, kb.sum())
    m["log_bonus_f"] = m["log_bonus"].fillna(0.0)
    # 指標。有些指標的分子是別的指標的分母（安打 → 長打比例、三壘打率；盜壘嘗試 → 盜壘成功率；投球數 → 揮空率、壞球率），
    # 要先生分母那一個，後生的才用新的分母，否則分子分母對不上（第一版就是這樣把 SB% 的符號弄反）。
    dens = {den for _, _, den, _, _ in inds}
    ordered = [t for t in inds if t[1] in dens] + [t for t in inds if t[1] not in dens]
    for name, num, den, kind, k in ordered:
        d = m[den].to_numpy(dtype=float)
        y_real = m[num].to_numpy(dtype=float)
        ok = np.isfinite(d) & (d > 0) & np.isfinite(y_real)
        if kind == "normal":  # 已標準化的連續量：層級平均 0，λη 加量測雜訊
            y = np.full(n, np.nan)
            y[ok] = lam[name] * eta[ok, k] + rng.normal(0, SIGMA_NORMAL * np.sqrt(100.0 / d[ok]))
            m[num] = y
            continue
        g = np.zeros(fy.N_LEVELS)
        for l in np.unique(lvl):
            sel = ok & (lvl == l)
            rate = y_real[sel].sum() / max(d[sel].sum(), 1)
            rate = min(max(rate, 1e-3), 1 - 1e-3) if kind == "binom" else max(rate, 1e-3)
            g[l] = np.log(rate / (1 - rate)) if kind == "binom" else np.log(rate)
        lin = g[lvl] + lam[name] * eta[:, k]
        y = np.full(n, np.nan)
        if kind == "binom":
            p = expit(lin[ok])
            if truth == "betabinom":
                p = rng.beta(np.clip(p * kappa, 1e-3, None), np.clip((1 - p) * kappa, 1e-3, None))
            y[ok] = rng.binomial(d[ok].astype(int), p)
        else:
            mu = np.exp(lin[ok]) * d[ok]
            phi = 5.0
            y[ok] = rng.poisson(rng.gamma(phi, mu / phi))
        m[num] = y
    # 結構段：共變數、球團截距、球探、能力；按結果種類生
    mask_col = "structural" if outcome == "promoted" else "war_structural"
    s_idx = np.where(m[mask_col].to_numpy())[0]
    X, _ = fy.design(m, 2, covariates=["age_rel", "log_pa"], dummies=("undrafted", "lvl_HS", "lvl_JC"), linear=list(fy.LINEAR_COVS))
    u_org = rng.normal(0, TAU_ORG_TRUE, n_org)
    slope_dev = rng.normal(0, SIGMA_ORG_SLOPE_TRUE, (n_org, K)) if org_slopes else np.zeros((n_org, K))
    type_dev = rng.normal(0, SIGMA_TYPE_ORG_TRUE, n_org) if club_type else np.zeros(n_org)
    type_ = eta[:, 1] - eta[:, 0]
    club_c = m["club_war_c"].to_numpy(dtype=float) if club_type else np.zeros(n)

    def predictor(prefix: str) -> np.ndarray:
        beta = np.array(BETA_TRUE[prefix][:K])
        b_cov = rng.normal(0, 0.2, X.shape[1])
        lin = X @ b_cov + u_org[org] + BETA_SCOUT_TRUE[prefix] * zeta + eta @ beta
        if org_slopes:
            lin = lin + (slope_dev[org] * eta).sum(1)
        if club_type:
            lin = lin + BETA_TYPE_CLUB_TRUE * club_c * type_ + type_dev[org] * type_
        return lin

    truth_d = {
        "lambda": {name: lam[name] for name in lam},
        "beta_scout": {p: BETA_SCOUT_TRUE[p] for p in (("",) if outcome == "promoted" else ("t_", "w_"))},
        "beta": {p: {fy.FACTOR_NAMES[k]: BETA_TRUE[p][k] for k in range(K)} for p in (("",) if outcome == "promoted" else ("t_", "w_"))},
        "tau_org": TAU_ORG_TRUE,
        "eta": eta,
        "u_org": u_org,
        "corr_off": CORR_OFF,
        "slope_dev": slope_dev,
        "type_dev": type_dev,
    }
    if outcome == "promoted":
        gamma_s = level_logits(m, s_idx, "promoted")
        lin = gamma_s[lvl] + predictor("")
        prom = np.zeros(n, dtype=bool)
        prom[s_idx] = rng.random(len(s_idx)) < expit(lin[s_idx])
        m["promoted"] = prom
    else:
        # 幾年登上：log T 常態，六季內沒登上就設限；登上者的 WAR：asinh 尺度 t 分布
        base_t = float(np.log(np.clip(m.loc[s_idx, "T_debut"].to_numpy(dtype=float), 1, fy.DEBUT_CENSOR)).mean())
        lin_t = base_t + predictor("t_")
        T = np.exp(lin_t + rng.normal(0, SIGMA_T_TRUE, n))
        reached = np.zeros(n, dtype=bool)
        reached[s_idx] = T[s_idx] < fy.DEBUT_CENSOR
        m["reached_fwd"] = reached
        m["T_debut"] = np.where(reached, np.clip(np.round(T), 1, 7), fy.DEBUT_CENSOR)
        lin_w = 0.5 + predictor("w_")
        war_asinh = lin_w + SIGMA_W_TRUE * rng.standard_t(NU_TRUE, n)
        m["war_fwd"] = np.where(reached, np.sinh(war_asinh), 0.0)
        truth_d.update({"sigma_t": SIGMA_T_TRUE, "sigma_w": SIGMA_W_TRUE, "nu": NU_TRUE})
    return m, truth_d


def fit_and_compare(m: pd.DataFrame, truth: dict, draws: int, tune: int, seed: int, outcome: str = "promoted", factors: int = 4, statcast: str = "none", org_slopes: bool = False, club_type: bool = False) -> dict:
    import pymc as pm

    model, info = fy.build(m, outcome, factors, "linear", 2, "M1", scout=True, org_slopes=org_slopes, club=club_type, statcast=statcast, club_type=club_type)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=2, init="jitter+adapt_diag", target_accept=0.9, random_seed=seed, progressbar=False, initvals=info["initvals"], compute_convergence_checks=False)
    fy.align_signs(idata, factors, "linear", False, statcast=statcast)
    post = idata.posterior
    rows = []

    def add(name, true, draws_arr):
        x = np.asarray(draws_arr).ravel()
        lo, hi = np.quantile(x, 0.055), np.quantile(x, 0.945)
        rows.append({"參數": name, "真值": float(true), "估計": float(x.mean()), "lo": float(lo), "hi": float(hi), "涵蓋": bool(lo <= true <= hi)})

    for name, true in truth["lambda"].items():
        v = f"lambda_{fy.indicator_tag(name)}"
        if v in post.data_vars:
            add(f"λ {name}", true, post[v].values)
    for prefix, betas in truth["beta"].items():
        for k, fname in fy.FACTOR_NAMES.items():
            if k < factors and f"{prefix}beta_{fname}" in post.data_vars:
                add(f"{prefix}β {fname}", betas[fname], post[f"{prefix}beta_{fname}"].values[..., 0])
        add(f"{prefix}β_scout", truth["beta_scout"][prefix], post[f"{prefix}beta_scout"].values)
        if org_slopes:
            for k in range(factors):
                v = f"{prefix}sigma_org_{fy.FACTOR_NAMES[k]}"
                if v in post.data_vars:
                    add(f"{prefix}σ_org {fy.FACTOR_NAMES[k]}", SIGMA_ORG_SLOPE_TRUE, post[v].values)
        if club_type and f"{prefix}beta_type_x_club" in post.data_vars:
            add(f"{prefix}β 型態×母隊強弱", BETA_TYPE_CLUB_TRUE, post[f"{prefix}beta_type_x_club"].values)
            add(f"{prefix}σ_org 型態", SIGMA_TYPE_ORG_TRUE, post[f"{prefix}sigma_org_type"].values)
    for prefix in truth["beta"]:  # WAR 段的球團截距是 t_／w_ 各一組，節點名有前綴
        add(f"{prefix}τ_org", truth["tau_org"], post[f"{prefix}tau_org"].values)
    if outcome == "war":
        add("σ_t", truth["sigma_t"], post["sigma_t"].values)
        add("σ_w", truth["sigma_w"], post["sigma_w"].values)
        add("ν", truth["nu"], post["nu"].values)
    if "factor_corr" in post.data_vars:
        add("因子相關（對角外平均）", truth["corr_off"], post["factor_corr"].values[..., 0, 1])
    eta_hat = post["eta"].mean(("chain", "draw")).values
    corr_eta = {fy.FACTOR_NAMES[k]: float(np.corrcoef(eta_hat[:, k], truth["eta"][:, k])[0, 1]) for k in range(factors)}
    prefixes = list(truth["beta"])
    corr_u_all = {pre: float(np.corrcoef(post[f"{pre}u_org"].mean(("chain", "draw")).values, truth["u_org"])[0, 1]) for pre in prefixes}
    corr_u = corr_u_all[prefixes[0]]  # 升級段就是 u_org；WAR 段取 t_（登上時間）那段，w_ 另記
    res = {"params": rows, "corr_eta": corr_eta, "corr_u_org": corr_u, **({"corr_u_org_w": corr_u_all["w_"]} if "w_" in corr_u_all else {}), "rhat_max": float(fy.rhat_max(idata)[0]), "n": int(len(m)), "n_struct": int(len(info["s_idx"]))}
    if org_slopes:  # 每隊斜率偏離的還原：三十隊 × K 把尺攤平算相關
        pre = "" if outcome == "promoted" else "t_"
        dev_hat = np.column_stack([post[f"{pre}beta_{fy.FACTOR_NAMES[k]}_org"].mean(("chain", "draw")).values for k in range(factors)])
        res["corr_slope_dev"] = float(np.corrcoef(dev_hat.ravel(), truth["slope_dev"].ravel())[0, 1])
    if club_type:
        pre = "" if outcome == "promoted" else "t_"
        res["corr_type_dev"] = float(np.corrcoef(post[f"{pre}beta_type_org"].mean(("chain", "draw")).values, truth["type_dev"])[0, 1])
    return res


def main(data_dir: str, out_dir: str, season: int, rep: int, truth: str, kappa: float, sub: int | None, quick: bool, tag: str, outcome: str = "promoted", factors: int = 4, statcast: str = "none", org_slopes: bool = False, club_type: bool = False) -> dict:
    out = Path(out_dir)
    rng = np.random.default_rng(1000 * season + rep)
    m = fy.load(Path(data_dir), season, club=club_type, statcast=statcast)
    if club_type and "club_war_c" not in m.columns:
        raise SystemExit("--club-type 需要 club 資料（club_features），這台沒有")
    if sub and sub < len(m):
        m = m.sample(sub, random_state=rep).reset_index(drop=True)
        m["org_i"] = pd.factorize(m["org"].fillna("UNK"))[0]
    mtag = model_tag(outcome, factors, statcast, org_slopes, club_type)
    m_sim, truth_d = simulate(m, out, season, tag, truth, kappa, rng, outcome, factors, statcast, org_slopes, club_type)
    draws, tune = (300, 300) if quick else (1000, 1000)
    res = fit_and_compare(m_sim, truth_d, draws, tune, rep, outcome, factors, statcast, org_slopes, club_type)
    res.update({"season": season, "rep": rep, "truth": truth, "kappa": kappa if truth == "betabinom" else None, "model": mtag})
    (out / "sim").mkdir(parents=True, exist_ok=True)
    path = out / "sim" / f"recover_{season}_{mtag}_{truth}_rep{rep}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    tab = pd.DataFrame(res["params"]).round(3)
    print(tab.to_string(index=False))
    print(f"η 相關 {res['corr_eta']}；u_org 相關 {res['corr_u_org']:.2f}；r-hat {res['rhat_max']:.2f}")
    print(f"-> {path}", flush=True)
    return res


def summarize(out_dir: str) -> str:
    out = Path(out_dir)
    files = sorted((out / "sim").glob("recover_*.json"))
    if not files:
        raise SystemExit("outputs/sim/ 沒有 recover_*.json")
    rows, extra = [], []
    for f in files:
        r = json.loads(f.read_text())
        if r["rep"] >= 90:  # rep ≥ 90 是 --quick 的煙霧測試，不進彙整
            continue
        mt = r.get("model", "promoted_f4s")  # 第一批檔沒有 model 欄 = 主模型
        for p in r["params"]:
            rows.append({"模型": mt, "truth": r["truth"], "季": r["season"], "rep": r["rep"], **p})
        e = {"模型": mt, "truth": r["truth"], "季": r["season"], "rep": r["rep"], **{f"η 相關 {k}": v for k, v in r["corr_eta"].items()}, "u_org 相關": r["corr_u_org"], "r-hat": r["rhat_max"]}
        for k in ("corr_slope_dev", "corr_type_dev"):
            if k in r:
                e[{"corr_slope_dev": "斜率偏離相關", "corr_type_dev": "型態偏離相關"}[k]] = r[k]
        extra.append(e)
    d = pd.DataFrame(rows)
    d["偏差"] = d["估計"] - d["真值"]
    g = d.groupby(["模型", "truth", "參數"], sort=False).agg(真值=("真值", "mean"), 平均估計=("估計", "mean"), 平均偏差=("偏差", "mean"), RMSE=("偏差", lambda x: float(np.sqrt((x**2).mean()))), 涵蓋率89=("涵蓋", "mean"), 複本=("rep", "count")).round(3).reset_index()
    e = pd.DataFrame(extra).groupby(["模型", "truth"]).mean(numeric_only=True).round(3).drop(columns=["季", "rep"])
    n_rep = pd.DataFrame(extra).groupby(["模型", "truth"]).size().rename("複本")
    overall = g.groupby(["模型", "truth"]).agg(參數數=("參數", "count"), 平均絕對偏差=("平均偏差", lambda x: float(np.abs(x).mean())), 涵蓋率89=("涵蓋率89", "mean")).round(3).join(n_rep)
    md = [
        "# 參數還原彙整\n",
        f"{len(files)} 個複本。涵蓋率名目 89%；真相 = binom 時涵蓋率明顯低於 0.89 或偏差明顯非零，是模型自己的問題；真相 = betabinom 時的落差，是過度離散的代價。\n",
        "## 每個模型一行\n",
        overall.to_markdown(),
        "",
        "## 每個參數\n",
        g.to_markdown(index=False),
        "",
        "## η、u_org（與斜率偏離）與真值的相關，r-hat\n",
        e.to_markdown(),
        "",
    ]
    text = "\n".join(md)
    (out / "sim_recover_summary.md").write_text(text)
    print(text)
    return text


def selftest() -> None:
    """只測生成式（不取樣）：每種模型的假資料形狀對、分子 ≤ 分母、真值能從指標算回來（λ 的符號對）。"""
    import tempfile

    data = Path("data")
    if not (data / "model_bat.parquet").exists():
        print("沒有 data/model_bat.parquet，略過")
        return
    rng = np.random.default_rng(0)
    try:
        fy.load(data, 2015)
    except KeyError as e:  # 本機的 model_bat.parquet 是舊面板（share 版），缺欄；生成式只能在 Arc 用 --quick --sub 500 試
        print(f"本機面板缺 {e}，生成式測試在 Arc 跑（每種模型 --quick --sub 500 各一次）；略過")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for outcome, factors, statcast, org_slopes in (("promoted", 4, "none", False), ("war", 4, "none", False), ("promoted", 5, "none", True)):
            m = fy.load(data, 2015).sample(600, random_state=0).reset_index(drop=True)
            m["org_i"] = pd.factorize(m["org"].fillna("UNK"))[0]
            ms, tr = simulate(m, Path(tmp), 2015, "none", "binom", 800.0, rng, outcome, factors, statcast, org_slopes)
            for name, num, den, kind, k in fy.assign_indicators(factors, statcast):
                if kind == "binom":
                    ok = ms[den].notna() & (ms[den] > 0)
                    assert (ms.loc[ok, num] <= ms.loc[ok, den]).all(), name
                # λ 的符號：指標比率跟 η 的相關要跟真值 λ 同號
                rate = (ms[num] / ms[den].replace(0, np.nan)) if kind != "normal" else ms[num]
                ok = rate.notna() & np.isfinite(rate)
                if ok.sum() > 100:
                    c = np.corrcoef(rate[ok], tr["eta"][ok.to_numpy(), k])[0, 1]
                    assert np.sign(c) == np.sign(tr["lambda"][name]) or abs(c) < 0.05, (name, c, tr["lambda"][name])
            if outcome == "promoted":
                assert 0.05 < ms["promoted"].mean() < 0.95
                assert set(tr["beta"]) == {""}
            else:
                assert 0.02 < ms["reached_fwd"].mean() < 0.9, ms["reached_fwd"].mean()
                assert (ms.loc[~ms["reached_fwd"].astype(bool), "war_fwd"] == 0).all()
                assert set(tr["beta"]) == {"t_", "w_"}
            if org_slopes:
                assert tr["slope_dev"].std() > 0.1
            print(f"  {model_tag(outcome, factors, statcast, org_slopes, False)} 生成式 OK（{len(ms)} 人季）")
        # 進階量模式（要 2024 的 3A 逐球）
        if (data / "savant_minors_2024.parquet").exists():
            m = fy.load(data, 2024, statcast="same")
            m["org_i"] = pd.factorize(m["org"].fillna("UNK"))[0]
            for sc in ("same", "only", "both"):
                ms, tr = simulate(m, Path(tmp), 2024, "none", "binom", 800.0, rng, "promoted", 5, sc)
                names = [n for n, *_ in fy.assign_indicators(5, sc)]
                assert all(n in tr["lambda"] for n in names)
                if sc != "same":
                    assert ms["sc_ev90"].notna().sum() > 50
                print(f"  promoted_f5s{fy.STATCAST_LETTER[sc]} 生成式 OK（{len(ms)} 人季，{len(names)} 個指標）")
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="參數還原：假農場 → fit_year M1 → 跟真值比；每種模型都能跑")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("season", nargs="?", type=int)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--truth", choices=["binom", "betabinom"], default="binom")
    p.add_argument("--kappa", type=float, default=800.0, help="betabinom 的集中度；800 對 400 打席約是卡方 1.5 倍")
    p.add_argument("--sub", type=int, default=None, help="隨機抽多少人季（加速；None = 整季）")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tag", default="promoted_f4sed_linear_p2", help="真值負荷量取自哪個真實 fit")
    p.add_argument("--outcome", choices=["promoted", "war"], default="promoted")
    p.add_argument("--factors", type=int, default=4)
    p.add_argument("--statcast", choices=["none", "same", "only", "both"], default="none")
    p.add_argument("--org-slopes", dest="org_slopes", action="store_true")
    p.add_argument("--club-type", dest="club_type", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.summarize:
        summarize(a.out_dir)
    else:
        if a.season is None:
            p.error("要給 season")
        main(a.data_dir, a.out_dir, a.season, a.rep, a.truth, a.kappa, a.sub, a.quick, a.tag, a.outcome, a.factors, a.statcast, a.org_slopes, a.club_type)
