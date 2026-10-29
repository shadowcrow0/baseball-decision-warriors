"""決策程序的模型還原：用五種程序各自造一組「球隊的升級決定」（真程序已知），再用 decision_bayes 的五個候選去配、LOO 選，
看 LOO 認不認得出真的那一個。這是「加權加總（WADD）12/12 季第一名」這句話的前提：如果真相是規則、LOO 也會說整合，那句話就沒意義。

    python src/sim_decision.py outputs/ 2015 --truth integrate --rep 0 [--quick] [--sub 1500]
    python src/sim_decision.py --summarize outputs/
    python src/sim_decision.py --selftest

輸入 = 真實那一季每個人季的七個數字（四把尺、球探、年齡、打席，季 × 層級內標準化，跟 decision_bayes 一樣）與層級；只有 y 重生：
  rule1     p = γ_層級 + 2.0 · 1[打得到球 > 0.5]
  lin1      p = γ_層級 + 1.5 · 打得到球
  rule2     p = γ_層級 + 1.5 · 1[打得到球 > 0.3] + 過了 · 1.5 · 1[打得遠 > 0] + 沒過 · 1.0 · 1[球探 > 0.5]
  integrate p = γ_層級 + Σ b_j x_j，b = (0.8, 0.5, 0.2, −0.3, 0.6, −0.3, 0.5)
  exemplar  p = Σ_e s_ie y_e / Σ_e s_ie，範例 = 同季隨機 500 個真實升級結果，s = exp(−2 Σ_k w_k |x_ik − x_ek|)，w ∝ (3,2,1,1,2,1,1)
γ_層級 = 真實升級率的 logit。每個複本一個 Arc job（一季 22 個小模型，--quick 約半小時）。
輸出 outputs/sim/decision_<season>_<truth>_rep<k>.json：每個家族最好的 elpd、LOO 第一名家族、第一名跟真程序家族的 elpd 差。
--summarize：真程序 × LOO 第一名的混淆表、認對的比例、認對時贏多少（elpd）、認錯時輸多少。
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
from decision_style import FEATURES, LEVELS, load_scores  # noqa: E402

TRUTHS = ["rule1", "lin1", "rule2", "integrate", "exemplar"]
FAMILY_OF = {"rule1": "單一規則", "lin1": "單一線性", "rule2": "兩個規則", "integrate": "加權加總", "exemplar": "Exemplar"}
COARSE = {"單一規則": "規則", "兩個規則": "規則", "單一線性": "線性／整合", "加權加總": "線性／整合", "Exemplar": "Exemplar"}  # 兩個規則包含單一規則、整合包含單一線性，大類認對就算認對
B_INTEGRATE = np.array([0.8, 0.5, 0.2, -0.3, 0.6, -0.3, 0.5])
W_EXEMPLAR = np.array([3, 2, 1, 1, 2, 1, 1], dtype=float) / 11.0
C_EXEMPLAR = 2.0
M_EXEMPLAR = 500


def expit(x):
    return 1 / (1 + np.exp(-x))


def level_logits(d: pd.DataFrame) -> np.ndarray:
    """每個層級真實升級率的 logit（照 LEVELS 順序）。"""
    g = np.zeros(len(LEVELS))
    for i, l in enumerate(LEVELS):
        r = d.loc[d["level_main"] == l, "y"].mean() if (d["level_main"] == l).any() else 0.4
        r = min(max(float(r), 0.05), 0.95)
        g[i] = np.log(r / (1 - r))
    return g


def generate(d: pd.DataFrame, truth: str, rng: np.random.Generator) -> np.ndarray:
    """照一種程序生 y（0/1）。X 是標準化後的七個數字。"""
    X = d[FEATURES].to_numpy(dtype=float)
    lvl = d["level_main"].map({l: i for i, l in enumerate(LEVELS)}).to_numpy()
    gamma = level_logits(d)[lvl]
    contact, power, scout = X[:, 0], X[:, 1], X[:, 4]
    if truth == "rule1":
        lin = gamma + 2.0 * (contact > 0.5)
    elif truth == "lin1":
        lin = gamma + 1.5 * contact
    elif truth == "rule2":
        s1 = (contact > 0.3).astype(float)
        lin = gamma + 1.5 * s1 + s1 * 1.5 * (power > 0) + (1 - s1) * 1.0 * (scout > 0.5)
    elif truth == "integrate":
        lin = gamma + X @ B_INTEGRATE
    elif truth == "exemplar":
        idx = rng.choice(len(X), size=min(M_EXEMPLAR, len(X)), replace=False)
        y_ex = d["y"].to_numpy(dtype=float)[idx]  # 範例用真實的升級結果：「跟過去被升的人像不像」
        D = np.abs(X[:, None, :] - X[idx][None, :, :])
        s = np.exp(-C_EXEMPLAR * (D * W_EXEMPLAR).sum(2))
        s[np.arange(len(X))[:, None] == idx[None, :]] = 0.0
        prior = float(y_ex.mean())
        p = (s @ y_ex + 2.0 * prior) / (s.sum(1) + 2.0)
        lin = gamma - np.log(1 / prior - 1) + np.log(p / (1 - p))  # 層級截距只當修正，整體率由範例決定
    else:
        raise ValueError(truth)
    return (rng.random(len(X)) < expit(lin)).astype(float)


def main(out_dir: str, season: int, truth: str, rep: int, quick: bool, sub: int | None, tag: str) -> dict:
    from decision_bayes import elpd_col, family, run_season, canon

    out = Path(out_dir)
    d = load_scores(out, tag)
    d = d[d["season"] == season].reset_index(drop=True)
    if len(d) < 200:
        raise SystemExit(f"{season} 只有 {len(d)} 人季")
    rng = np.random.default_rng(100 * season + 10 * TRUTHS.index(truth) + rep)
    if sub and sub < len(d):
        d = d.sample(sub, random_state=rep).reset_index(drop=True)
    d["y"] = generate(d, truth, rng)
    print(f"{season} 真程序 {truth}：{len(d):,} 人季，升級率 {d['y'].mean():.2f}", flush=True)
    draws, tune = (300, 300) if quick else (1000, 1000)
    cmp_loo, _, notes = run_season(d, draws, tune, seed=rep)
    col = elpd_col(cmp_loo)
    best = cmp_loo.groupby(cmp_loo.index.map(family))[col].max().sort_values(ascending=False)
    winner = best.index[0]
    true_family = FAMILY_OF[truth]
    res = {
        "season": season,
        "truth": truth,
        "true_family": true_family,
        "rep": rep,
        "n": int(len(d)),
        "rate": float(d["y"].mean()),
        "winner": winner,
        "correct": bool(winner == true_family),
        "correct_coarse": bool(COARSE[winner] == COARSE[true_family]),
        "best_by_family": {k: float(v) for k, v in best.items()},
        "margin_true_vs_best_other": float(best[true_family] - best.drop(true_family).max()),
        "winner_note": notes.get(cmp_loo.index[0], ""),
    }
    (out / "sim").mkdir(parents=True, exist_ok=True)
    path = out / "sim" / f"decision_{season}_{truth}_rep{rep}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(pd.Series(best).round(1).to_string())
    print(f"真程序 {true_family}，LOO 第一名 {winner}，{'認對' if res['correct'] else '認錯'}；真 − 其他最好 = {res['margin_true_vs_best_other']:+.1f} elpd")
    print(f"-> {path}", flush=True)
    return res


def summarize(out_dir: str) -> str:
    out = Path(out_dir)
    files = sorted((out / "sim").glob("decision_*.json"))
    if not files:
        raise SystemExit("outputs/sim/ 沒有 decision_*.json")
    rows = [r for r in (json.loads(f.read_text()) for f in files) if r["rep"] < 90]  # rep ≥ 90 是煙霧測試
    d = pd.DataFrame(rows)
    from decision_bayes import canon  # 舊檔的「資訊整合」→ 加權加總（main 裡的 import 是函式區域的，這裡要自己 import）

    d["winner"] = d["winner"].map(canon)
    d["true_family"] = d["true_family"].map(canon)
    conf = pd.crosstab(d["true_family"], d["winner"]).reindex(index=[FAMILY_OF[t] for t in TRUTHS if FAMILY_OF[t] in set(d["true_family"])])
    if "correct_coarse" not in d.columns:
        d["correct_coarse"] = [COARSE[w] == COARSE[t] for w, t in zip(d["winner"], d["true_family"])]
    acc = d.groupby("true_family").agg(複本=("rep", "count"), 認對率=("correct", "mean"), 大類認對率=("correct_coarse", "mean"), 真程序減其他最好_elpd=("margin_true_vs_best_other", "mean"), 升級率=("rate", "mean")).round(2)
    md = [
        "# 決策程序的模型還原\n",
        f"{len(files)} 個複本：用五種程序各自生球隊的升級決定，再讓 decision_bayes 的五個候選配、LOO 選。"
        "認對率 = LOO 第一名的家族就是生資料的那個；大類認對率把單一規則／兩個規則算同一類（規則）、單一線性／加權加總算同一類（線性／整合）；「真程序 − 其他最好」是真程序家族的 elpd 減掉其他家族裡最好的，正 = 認對且贏多少，負 = 認錯輸多少。\n",
        "## 混淆表（列 = 真程序，欄 = LOO 第一名）\n",
        conf.to_markdown(),
        "",
        "## 每種真程序\n",
        acc.to_markdown(),
        "",
        "讀法：真相是規則而 LOO 說整合 → 整合贏只是因為它彈性大，真實資料的 13/13 不能當證據；"
        "各種真相都認得對、而且規則為真時規則贏整合幾十 elpd → 真實資料裡整合贏規則 50–70 elpd 是可信的。\n",
    ]
    text = "\n".join(md)
    (out / "sim_decision_summary.md").write_text(text)
    print(text)
    return text


def selftest() -> None:
    """只測生成式：每種程序的 y 率合理、跟它該依賴的數字有關、不該依賴的沒關。"""
    rng = np.random.default_rng(0)
    n = 3000
    d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    d["level_main"] = rng.choice(LEVELS, n)
    d["y"] = (rng.random(n) < 0.4).astype(float)
    for t in TRUTHS:
        y = generate(d, t, rng)
        assert 0.1 < y.mean() < 0.9, (t, y.mean())
        c_contact = np.corrcoef(y, d["eta_contact"])[0, 1]
        c_speed = np.corrcoef(y, d["eta_speed"])[0, 1]
        if t in ("rule1", "lin1", "rule2"):
            assert c_contact > 0.15 and abs(c_speed) < 0.06, (t, c_contact, c_speed)
        if t == "integrate":
            assert c_contact > 0.15 and c_speed > 0.03, (t, c_contact, c_speed)
        if t == "rule1":  # 門檻：門檻兩側的率差大，門檻同側內跟 contact 無關
            above = d["eta_contact"] > 0.5
            assert y[above].mean() - y[~above].mean() > 0.25
            assert abs(np.corrcoef(y[~above], d.loc[~above, "eta_contact"])[0, 1]) < 0.06
        print(f"  {t}: 率 {y.mean():.2f}，跟打得到球 {c_contact:+.2f}，跟跑得快 {c_speed:+.2f}")
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="決策程序的模型還原：五種真程序 × decision_bayes 的 LOO 選擇")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("season", nargs="?", type=int)
    p.add_argument("--truth", choices=TRUTHS, default="integrate")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--sub", type=int, default=None)
    p.add_argument("--tag", default="promoted_f4sed_linear_p2")
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
        main(a.out_dir, a.season, a.truth, a.rep, a.quick, a.sub, a.tag)
