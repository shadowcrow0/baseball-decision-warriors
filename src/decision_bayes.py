"""用貝氏模型比五種「球隊做決定的程序」，用 LOO／WAIC 選：哪一種程序最接近球隊實際的升級決定。

    python src/decision_bayes.py outputs/ --promoted promoted_f4sed_linear_p2 --seasons 2015 2019 2024 [--quick]
    python src/decision_bayes.py --selftest

每季一組資料：每個人季的七個輸入（四把尺、球探評價、年齡、打席；季 × 層級內標準化）、結果 = 隔年升不升；層級截距每個模型都有。
五種程序，每一種是一個 PyMC 模型，結果都是 Bernoulli(logit p)，差別只在 p 怎麼算：
  單一規則   p = γ_層級 + b · 1[x_j > c]            一個數字、一個門檻、過了就升（門檻用陡的 sigmoid 近似，坡度固定 10）
  單一線性   p = γ_層級 + b · x_j                   同一個數字、沒有門檻、越高越升
  兩個規則   p = γ_層級 + 先看 x_j 過不過 c1，過了再看 x_k 過不過 c2a、沒過也再看 x_k 過不過 c2b（深度二的樹，「中了就中」）
  加權加總   p = γ_層級 + Σ_j b_j x_j                每個數字一個權重加起來（WADD, weighted additive；Payne–Bettman–Johnson 1993，Brunswik lens model 的判斷端）
  Exemplar   p = Σ_e s_ie y_e / Σ_e s_ie，s_ie = exp(−c Σ_k w_k |x_ik − x_ek|)   跟過去被升的人像不像（注意力 w 是 Dirichlet，範例 = 同季隨機抽的 M 個人，不含自己）
單一規則、單一線性對七個數字各配一個；兩個規則對單一規則 LOO 前三名的數字兩兩配對（6 個）。全部用 az.compare（PSIS-LOO）排名，
同時給 WAIC；elpd 差 > 2 × dse 才算分得出來。輸出 outputs/decision_bayes_<tag>_<起季>_<迄季>.md（拆幾個 job 跑不同季不會互相覆蓋）。
正式版 2 鏈 × 1,000；--quick 2 × 300。一季約 20 個小模型，Arc 上半小時到一小時。
"""

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

warnings.filterwarnings("ignore")

# 2026-10-26 前的輸出把加權加總叫「資訊整合」（借自分類學習的 information-integration，名字不對）；讀舊檔時換成新名
LEGACY_NAMES = {"資訊整合": "加權加總"}


def canon(name: str) -> str:
    """舊輸出的家族名換成現在的名字。"""
    return LEGACY_NAMES.get(name, name)
sys.path.insert(0, str(Path(__file__).parent))
from decision_style import FEATURE_NAMES, FEATURES, LEVELS, load_scores  # noqa: E402
from fit_year import unround_compare  # noqa: E402

STEEP = 10.0  # 門檻的坡度：標準化尺度上 0.1 個標準差內從 0 變 1
M_EXEMPLAR = 500


def level_index(d: pd.DataFrame) -> np.ndarray:
    return d["level_main"].map({l: i for i, l in enumerate(LEVELS)}).to_numpy()


def step(x, c):
    return pm.math.sigmoid(STEEP * (x - c))


def build(kind: str, X: np.ndarray, y: np.ndarray, lvl: np.ndarray, dims: tuple = (), rng=None) -> pm.Model:
    """一個候選程序的 PyMC 模型。X (n, K) 標準化輸入；dims 是這個程序用到的欄位索引。"""
    n_lvl = int(lvl.max()) + 1
    with pm.Model() as model:
        gamma = pm.Normal("gamma", 0, 2, shape=n_lvl)
        lin = gamma[lvl]
        if kind == "rule1":
            j = dims[0]
            c = pm.Normal("c", 0, 1)
            b = pm.Normal("b", 0, 2)
            lin = lin + b * step(X[:, j], c)
        elif kind == "lin1":
            j = dims[0]
            b = pm.Normal("b", 0, 2)
            lin = lin + b * X[:, j]
        elif kind == "rule2":
            j, k = dims
            c1 = pm.Normal("c1", 0, 1)
            c2 = pm.Normal("c2", 0, 1, shape=2)  # 過了第一關／沒過第一關，各自的第二個門檻
            b = pm.Normal("b", 0, 2, shape=3)  # 第一關的效果、過了之後第二關的效果、沒過之後第二關的效果
            s1 = step(X[:, j], c1)
            lin = lin + b[0] * s1 + s1 * b[1] * step(X[:, k], c2[0]) + (1 - s1) * b[2] * step(X[:, k], c2[1])
        elif kind == "integrate":
            b = pm.Normal("b", 0, 1, shape=X.shape[1])
            lin = lin + pt.dot(X, b)
        elif kind == "exemplar":
            idx = rng.choice(len(X), size=min(M_EXEMPLAR, len(X)), replace=False)
            D = np.abs(X[:, None, :] - X[idx][None, :, :])  # (n, M, K) 常數
            self_mask = (np.arange(len(X))[:, None] == idx[None, :]).astype(float)  # 自己不當自己的範例
            w = pm.Dirichlet("w", a=np.ones(X.shape[1]))
            c = pm.HalfNormal("c", 2.0)
            s = pt.exp(-c * pt.tensordot(pt.as_tensor(D), w, axes=[[2], [0]])) * (1 - self_mask)
            prior = float(y.mean())
            p = (pt.dot(s, y[idx]) + 2.0 * prior) / (s.sum(axis=1) + 2.0)
            lin = lin + pt.log(p / (1 - p))  # 層級截距當成對 exemplar 機率的修正
        else:
            raise ValueError(kind)
        pm.Bernoulli("y", logit_p=lin, observed=y)
    return model


def fit(model: pm.Model, draws: int, tune: int, seed: int) -> az.InferenceData:
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=2, target_accept=0.9, random_seed=seed, progressbar=False, compute_convergence_checks=False)
        pm.compute_log_likelihood(idata, progressbar=False)
    return idata


def run_season(d: pd.DataFrame, draws: int, tune: int, seed: int = 0) -> tuple:
    """一季：配全部候選，回傳 (az.compare 表, 參數摘要 dict)。"""
    X = d[FEATURES].to_numpy(dtype=float)
    y = d["y"].to_numpy(dtype=float)
    lvl = level_index(d)
    rng = np.random.default_rng(seed)
    idatas, notes = {}, {}
    for j, f in enumerate(FEATURES):
        idatas[f"單一規則：{FEATURE_NAMES[f]}"] = fit(build("rule1", X, y, lvl, (j,)), draws, tune, seed)
        idatas[f"單一線性：{FEATURE_NAMES[f]}"] = fit(build("lin1", X, y, lvl, (j,)), draws, tune, seed)
    loo1 = {name: float(getattr(az.loo(idatas[name]), "elpd", getattr(az.loo(idatas[name]), "elpd_loo", np.nan))) for name in idatas if name.startswith("單一規則")}
    top = sorted(loo1, key=loo1.get, reverse=True)[:3]
    top_idx = [FEATURES.index(next(f for f in FEATURES if FEATURE_NAMES[f] == name.split("：")[1])) for name in top]
    for a in top_idx:
        for b in top_idx:
            if a != b:
                name = f"兩個規則：{FEATURE_NAMES[FEATURES[a]]} → {FEATURE_NAMES[FEATURES[b]]}"
                idatas[name] = fit(build("rule2", X, y, lvl, (a, b)), draws, tune, seed)
                post = idatas[name].posterior
                notes[name] = f"c1 {float(post['c1'].mean()):+.2f}；過了再看 c2 {float(post['c2'][..., 0].mean()):+.2f}，沒過再看 c2 {float(post['c2'][..., 1].mean()):+.2f}"
    idatas["加權加總"] = fit(build("integrate", X, y, lvl), draws, tune, seed)
    bm = idatas["加權加總"].posterior["b"].mean(("chain", "draw")).values
    notes["加權加總"] = "、".join(f"{FEATURE_NAMES[f]} {v:+.2f}" for f, v in zip(FEATURES, bm))
    idatas["Exemplar"] = fit(build("exemplar", X, y, lvl, rng=rng), draws, tune, seed)
    wm = idatas["Exemplar"].posterior["w"].mean(("chain", "draw")).values
    notes["Exemplar"] = "注意力 " + "、".join(f"{FEATURE_NAMES[f]} {v:.2f}" for f, v in zip(FEATURES, wm)) + f"；c {float(idatas['Exemplar'].posterior['c'].mean()):.2f}"
    for name in idatas:
        if name.startswith("單一規則"):
            post = idatas[name].posterior
            notes[name] = f"門檻 c {float(post['c'].mean()):+.2f}（標準差單位），過門檻 b {float(post['b'].mean()):+.2f}"
    # arviz 1.x：先各自算 ELPD 物件再比，compare 不收 ic 參數
    loos = {k: az.loo(v) for k, v in idatas.items()}
    cmp_loo = unround_compare(az.compare(loos), loos)
    try:
        waics = {k: az.waic(v) for k, v in idatas.items()}
        cmp_waic = unround_compare(az.compare(waics), waics)
    except Exception:  # noqa: BLE001  WAIC 在某些版本沒有，就用 LOO 代
        cmp_waic = cmp_loo
    return cmp_loo, cmp_waic, notes


def family(name: str) -> str:
    return name.split("：")[0]


def elpd_col(tab: pd.DataFrame) -> str:
    """arviz 1.x 的 compare 表欄名是 elpd，0.x 是 elpd_loo / elpd_waic。"""
    return "elpd" if "elpd" in tab.columns else next(c for c in tab.columns if c.startswith("elpd_") and c != "elpd_diff")


def main(out_dir: str, tag: str, seasons: list | None, quick: bool) -> str:
    out = Path(out_dir)
    d_all = load_scores(out, tag)
    seasons = list(seasons) if seasons else sorted(d_all["season"].unique())
    draws, tune = (300, 300) if quick else (1000, 1000)
    md = [
        f"# 用貝氏模型比五種決定程序，LOO／WAIC 選 · {tag}\n",
        "每季一組，候選：單一規則（七個數字各一）、單一線性（七個各一）、兩個規則（單一規則前三名兩兩配對）、加權加總（WADD）、Exemplar。"
        "elpd 越大越好；elpd_diff 是跟第一名的差，dse 是差的標準誤；log10 BF = Δelpd ÷ ln 10 是 pseudo-Bayes factor（Geisser–Eddy），−1 = 第一名的證據是它的十倍，−2 = 一百倍。weight 是 stacking 權重。\n",
    ]
    tally = {}
    for season in seasons:
        d = d_all[d_all["season"] == season].reset_index(drop=True)
        if len(d) < 200:
            continue
        print(f"{season}: {len(d):,} 人季", flush=True)
        cmp_loo, cmp_waic, notes = run_season(d, draws, tune)
        col = elpd_col(cmp_loo)
        best_by_family = cmp_loo.groupby(cmp_loo.index.map(family))[col].max().sort_values(ascending=False)
        winner = best_by_family.index[0]
        tally[winner] = tally.get(winner, 0) + 1
        show = cmp_loo[["rank", col, "se", "elpd_diff", "dse", "weight"]].round(1).rename(columns={col: "elpd_loo"})
        show.insert(4, "log10 BF vs 第一名", (-cmp_loo["elpd_diff"].abs() / np.log(10)).round(1))  # pseudo-Bayes factor：Δelpd / ln 10，負 = 輸第一名
        show["waic"] = cmp_waic[elpd_col(cmp_waic)].reindex(show.index).round(1)
        show["參數"] = [notes.get(i, "") for i in show.index]
        md += [f"## {int(season)}（{len(d):,} 人季，升級率 {d['y'].mean():.2f}）\n", f"每個家族最好的 elpd_loo：{'；'.join(f'{k} {v:.1f}' for k, v in best_by_family.items())}\n", show.to_markdown(), ""]
    md += ["## 哪一種程序最接近球隊（各季 LOO 第一名的家族）\n", pd.Series(tally).rename("季數").to_markdown(), ""]
    text = "\n".join(md)
    span = f"_{int(min(seasons))}_{int(max(seasons))}" if seasons else ""
    path = out / f"decision_bayes_{tag}{span}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成兩季：一季真相是加權加總 → 加權加總要贏；一季真相是兩個門檻的樹 → 兩個規則要贏（LOO 家族最佳）。"""
    rng = np.random.default_rng(3)
    n = 700
    lvls = rng.choice(LEVELS[2:], n)
    X = rng.normal(size=(n, len(FEATURES)))
    frames = []
    for season, truth in ((2001, "linear"), (2002, "tree")):
        if truth == "linear":
            lin = -0.3 + 0.9 * X[:, 0] + 0.6 * X[:, 1] + 0.4 * X[:, 4] - 0.3 * X[:, 3]
        else:
            first = X[:, 0] > 0.3
            lin = -2.0 + 2.5 * first + np.where(first, 1.5 * (X[:, 4] > -0.2), 0.8 * (X[:, 1] > 0.8))
        y = rng.random(n) < 1 / (1 + np.exp(-lin))
        f = pd.DataFrame(X, columns=FEATURES)
        f["season"], f["level_main"], f["y"] = season, lvls, y.astype(float)
        frames.append(f)
    d_all = pd.concat(frames, ignore_index=True)
    wins = {}
    for season in (2001, 2002):
        d = d_all[d_all["season"] == season].reset_index(drop=True)
        cmp_loo, _, notes = run_season(d, 200, 200)
        best = cmp_loo.groupby(cmp_loo.index.map(family))[elpd_col(cmp_loo)].max().sort_values(ascending=False)
        print(season, best.round(1).to_dict())
        wins[season] = best.index[0]
    assert wins[2001] == "加權加總", f"加總的真相沒被選到：{wins}"
    assert wins[2002] == "兩個規則", f"樹的真相沒被選到：{wins}"
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="貝氏版：五種決定程序，LOO／WAIC 比")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("--promoted", default=None, help="升級模型的 tag")
    p.add_argument("--seasons", type=int, nargs="*", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.promoted is None:
        p.error("--promoted 是必要的（或用 --selftest）")
    else:
        main(a.out_dir, a.promoted, a.seasons, a.quick)
