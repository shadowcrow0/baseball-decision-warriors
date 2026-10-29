"""Bayes factor 工具：證據強度用 BF 講，不用 p < .05。

    python src/bf.py --selftest

三種算法：
  log10_bf10(b, se, tau)        係數的 Savage–Dickey 近似：H1 的先驗 β ~ N(0, τ²)，似然用 N(b̂, se²) 近似（大樣本），
                                後驗是兩個常態的乘積；BF10 = 先驗在 0 的密度 ÷ 後驗在 0 的密度。τ 預設 0.5（標準化尺度上的「中等」效果），
                                表上另附 τ = 0.2 與 1.0 的敏感度；三個都同方向才算穩。
  log10_bf10_draws(draws, sd)   有後驗抽樣時的 Savage–Dickey：後驗在 0 的密度用核密度估，先驗 N(0, sd²)。
  log10_pseudo_bf(delta_elpd)   模型比較：LOO 的 elpd 差換成 pseudo-Bayes factor（Geisser–Eddy），log10 BF = Δelpd / ln 10。
讀法（Jeffreys 的級距，log10）：> 2 極強、1–2 強、0.5–1 中等、−0.5–0.5 分不出來、< −0.5 支持「沒有效果」（負號是 H0 那邊）。
"""

import argparse

import numpy as np
from scipy.stats import gaussian_kde, norm

TAU_DEFAULT = 0.5
TAUS = (0.2, 0.5, 1.0)


def log10_bf10(b: float, se: float, tau: float = TAU_DEFAULT) -> float:
    """Savage–Dickey，常態先驗 × 常態似然近似。回傳 log10 BF10（正 = 支持有效果）。"""
    if not (np.isfinite(b) and np.isfinite(se)) or se <= 0:
        return np.nan
    post_prec = 1 / se**2 + 1 / tau**2
    post_mean = (b / se**2) / post_prec
    post_sd = np.sqrt(1 / post_prec)
    return float((norm.logpdf(0, 0, tau) - norm.logpdf(0, post_mean, post_sd)) / np.log(10))


def log10_bf10_draws(draws: np.ndarray, prior_sd: float) -> float:
    """有後驗抽樣：後驗在 0 的密度用 KDE。抽樣全在同一側、0 落在 KDE 範圍外時密度趨近 0，BF 很大，上限記成 log10 = 6。"""
    x = np.asarray(draws, dtype=float).ravel()
    if len(x) < 50 or x.std() == 0:
        return np.nan
    post0 = float(gaussian_kde(x)(0.0)[0])
    prior0 = norm.pdf(0, 0, prior_sd)
    if post0 <= 1e-300:
        return 6.0
    return float(min(np.log10(prior0 / post0), 6.0))


def log10_pseudo_bf(delta_elpd: float) -> float:
    return float(delta_elpd / np.log(10))


def bf_str(l10: float) -> str:
    """log10 BF → 好讀的字串：10^5、3.2、1/4.0、1/10^3。"""
    if not np.isfinite(l10):
        return "—"
    if l10 >= 3:
        return f"10^{l10:.0f}"
    if l10 >= 0:
        return f"{10**l10:.1f}"
    if l10 > -3:
        return f"1/{10**(-l10):.1f}"
    return f"1/10^{-l10:.0f}"


def label(l10: float) -> str:
    if not np.isfinite(l10):
        return "—"
    if l10 >= 2:
        return "極強"
    if l10 >= 1:
        return "強"
    if l10 >= 0.5:
        return "中等"
    if l10 > -0.5:
        return "分不出來"
    if l10 > -1:
        return "中等支持零"
    return "強支持零"


def cell(b: float, se: float, tau: float = TAU_DEFAULT, digits: int = 2) -> str:
    """表格格子：係數 [BF]，例如 "+0.65 [BF 10^4]"。"""
    return f"{b:+.{digits}f} [BF {bf_str(log10_bf10(b, se, tau))}]"


def sensitivity(b: float, se: float) -> str:
    """三個 τ 的 log10 BF，看方向穩不穩。"""
    return " / ".join(f"τ{t}: {log10_bf10(b, se, t):+.1f}" for t in TAUS)


def selftest() -> None:
    # 大效果、小標準誤 → 大 BF；效果 0、小標準誤 → 支持零；效果 0、大標準誤 → 分不出來
    big = log10_bf10(0.6, 0.1)
    null_tight = log10_bf10(0.0, 0.05)
    null_loose = log10_bf10(0.0, 1.0)
    assert big > 2 and null_tight < -0.5 and -0.5 < null_loose < 0.5, (big, null_tight, null_loose)
    rng = np.random.default_rng(0)
    assert log10_bf10_draws(rng.normal(0.6, 0.1, 4000), 1.0) > 2
    assert log10_bf10_draws(rng.normal(0.0, 0.05, 4000), 1.0) < -0.5
    assert abs(log10_pseudo_bf(np.log(10) * 3) - 3) < 1e-9
    assert bf_str(4.2) == "10^4" and bf_str(0.5).startswith("3.") and bf_str(-0.7).startswith("1/5")
    print(f"大效果 {big:.1f}，零且準 {null_tight:.1f}，零且不準 {null_loose:.1f}；{sensitivity(0.3, 0.1)}")
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    if p.parse_args().selftest:
        selftest()
