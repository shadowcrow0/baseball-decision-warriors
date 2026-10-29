"""反事實政策模擬：如果球隊照「真值尺」升人（WAR 段學到的能力 → 登上後六年 WAR），而不是照自己的尺，會差多少。

    python src/sim_policy.py outputs/ --promoted promoted_f4sed_linear_p2 --war war_f4sed_linear_p2 [--boot 500]
    python src/sim_policy.py --selftest

每季每個層級（A- … AAA）三個名單，人數都等於該季該層級實際升上去的人數 n：
  A = 實際升上去的人（球隊的決定，含雜訊）
  C = 球隊自己的尺（升級段的線性預測 score_promoted）排最前面的 n 個（球隊的偏好，去掉雜訊）
  B = 真值尺（WAR 段的 score_war_asinh = E[asinh(登上後六年 WAR) | 能力]）排最前面的 n 個
比三件事：
  1. 名單重疊：|A∩B|/n、|C∩B|/n（球隊的尺跟真值尺挑的人有多少一樣）。
  2. 模型帳面：Σ sinh(score_war_asinh) 在 A、C、B 三個名單的總和 = 「照這個名單升人，六年內預期拿到多少 WAR」；B−A 就是反事實的差。
     不確定性用該層級該季的人重抽（bootstrap）；證據用 Bayes factor（Savage–Dickey，先驗 N(0, τ²)，τ = 名單帳面 WAR 的 10%）。
  3. 可觀測的檢查（不靠模型）：實際升上去的人裡，真值尺也要的（A∩B）和真值尺不要的（A∖B），之後六年實際 WAR 平均差多少；
     以及球隊沒升、真值尺要的（B∖A）之後有沒有登上、拿了多少 WAR（下界：他們那季沒被升，之後才有機會）。
跨季：總和相加、差用反變異數合併、幾季同號。要兩段 fit 都有的季（2007–2019 共 13 季）。
輸出 outputs/sim_policy_<promoted tag>.md。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bf import bf_str, log10_bf10  # noqa: E402

LEVELS = ["A-", "A", "A+", "AA", "AAA"]


def load_season(out: Path, season: int, promoted_tag: str, war_tag: str) -> pd.DataFrame:
    """一季：升級段的分數與去向，接上 WAR 段的真值分數。只留兩個分數都有、六年 WAR 窗口完整的人季。"""
    p = pd.read_parquet(out / str(season) / promoted_tag / "eta_M1.parquet")
    w = pd.read_parquet(out / str(season) / war_tag / "eta_M1.parquet")
    cols = ["key_mlbam", "season", "level_main", "score_promoted", "promoted", "war_fwd", "reached_fwd", "war_observable", "PA"]
    d = p[cols].merge(w[["key_mlbam", "score_war_asinh", "score_preach"]], on="key_mlbam", how="inner")
    d = d[d["score_promoted"].notna() & d["score_war_asinh"].notna() & d["war_observable"].astype(bool)]
    d = d[d["level_main"].isin(LEVELS)].reset_index(drop=True)
    d["promoted"] = d["promoted"].astype(int)
    d["reached_fwd"] = d["reached_fwd"].astype(int)
    d["truth"] = np.sinh(d["score_war_asinh"].to_numpy(dtype=float))  # 帳面 WAR（期望值，WAR 單位）
    return d


def pick_top(score: np.ndarray, n: int) -> np.ndarray:
    """分數最高的 n 個人（布林遮罩）。"""
    mask = np.zeros(len(score), dtype=bool)
    if n > 0:
        mask[np.argsort(-score, kind="stable")[:n]] = True
    return mask


def lists_one(g: pd.DataFrame) -> dict:
    """一季一個層級：A / C / B 三個名單與帳面總和、重疊、可觀測檢查。"""
    n = int(g["promoted"].sum())
    a = g["promoted"].to_numpy() == 1
    c = pick_top(g["score_promoted"].to_numpy(dtype=float), n)
    b = pick_top(g["score_war_asinh"].to_numpy(dtype=float), n)
    truth = g["truth"].to_numpy(dtype=float)
    war = g["war_fwd"].to_numpy(dtype=float)
    reached = g["reached_fwd"].to_numpy(dtype=float)
    out = {
        "n_pool": len(g),
        "n": n,
        "overlap_AB": float((a & b).sum() / n) if n else np.nan,
        "overlap_CB": float((c & b).sum() / n) if n else np.nan,
        "overlap_AC": float((a & c).sum() / n) if n else np.nan,
        "war_A": float(truth[a].sum()),
        "war_C": float(truth[c].sum()),
        "war_B": float(truth[b].sum()),
        "obs_AandB_mean": float(war[a & b].mean()) if (a & b).any() else np.nan,
        "obs_AnotB_mean": float(war[a & ~b].mean()) if (a & ~b).any() else np.nan,
        "obs_AandB_n": int((a & b).sum()),
        "obs_AnotB_n": int((a & ~b).sum()),
        "obs_AandB_sd": float(war[a & b].std(ddof=1)) if (a & b).sum() > 1 else np.nan,
        "obs_AnotB_sd": float(war[a & ~b].std(ddof=1)) if (a & ~b).sum() > 1 else np.nan,
        "passed_n": int((b & ~a).sum()),
        "passed_reached": float(reached[b & ~a].mean()) if (b & ~a).any() else np.nan,
        "passed_war": float(war[b & ~a].mean()) if (b & ~a).any() else np.nan,
        "picked_reached": float(reached[a & ~b].mean()) if (a & ~b).any() else np.nan,
        "picked_war": float(war[a & ~b].mean()) if (a & ~b).any() else np.nan,
    }
    return out


def boot_delta(g: pd.DataFrame, n_boot: int, rng) -> tuple:
    """重抽該層級該季的人，算 B−A 與 B−C 帳面差的標準誤。"""
    db, dc = [], []
    idx = np.arange(len(g))
    for _ in range(n_boot):
        s = g.iloc[rng.choice(idx, len(idx), replace=True)].reset_index(drop=True)
        r = lists_one(s)
        db.append(r["war_B"] - r["war_A"])
        dc.append(r["war_B"] - r["war_C"])
    return float(np.std(db, ddof=1)), float(np.std(dc, ddof=1))


def season_level_table(d: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for (season, lvl), g in d.groupby(["season", "level_main"]):
        g = g.reset_index(drop=True)
        if g["promoted"].sum() < 5 or len(g) < 20:
            continue
        r = lists_one(g)
        se_ba, se_bc = boot_delta(g, n_boot, rng)
        r.update({"season": int(season), "level": lvl, "d_BA": r["war_B"] - r["war_A"], "se_BA": se_ba, "d_BC": r["war_B"] - r["war_C"], "se_BC": se_bc})
        rows.append(r)
    return pd.DataFrame(rows)


def pool(b: np.ndarray, se: np.ndarray) -> tuple:
    """反變異數合併：(合併值, 標準誤)。"""
    ok = np.isfinite(b) & np.isfinite(se) & (se > 0)
    if not ok.any():
        return np.nan, np.nan
    w = 1.0 / se[ok] ** 2
    return float((b[ok] * w).sum() / w.sum()), float(np.sqrt(1.0 / w.sum()))


def welch(m1, s1, n1, m2, s2, n2) -> tuple:
    """兩組平均的差與標準誤。"""
    if n1 < 2 or n2 < 2 or not (np.isfinite(s1) and np.isfinite(s2)):
        return np.nan, np.nan
    return float(m1 - m2), float(np.sqrt(s1**2 / n1 + s2**2 / n2))


def summarize(t: pd.DataFrame) -> tuple:
    """跨季、按層級彙整：帳面差（模型）與可觀測差。"""
    rows = []
    for lvl, g in t.groupby("level", sort=False):
        d_ba, se_ba = pool(g["d_BA"].to_numpy(), g["se_BA"].to_numpy())
        d_bc, se_bc = pool(g["d_BC"].to_numpy(), g["se_BC"].to_numpy())
        tau = max(0.1 * float((g["war_A"].abs() + g["war_B"].abs()).mean() / 2), 1.0)  # 先驗寬度：名單帳面 WAR 的 10%（至少 1 WAR）
        obs = [welch(r.obs_AandB_mean, r.obs_AandB_sd, r.obs_AandB_n, r.obs_AnotB_mean, r.obs_AnotB_sd, r.obs_AnotB_n) for r in g.itertuples()]
        o_b, o_se = pool(np.array([x[0] for x in obs]), np.array([x[1] for x in obs]))
        rows.append(
            {
                "層級": lvl,
                "季數": len(g),
                "每季升級人數": round(float(g["n"].mean()), 1),
                "名單重疊 A∩B": round(float(g["overlap_AB"].mean()), 2),
                "名單重疊 C∩B": round(float(g["overlap_CB"].mean()), 2),
                "實際名單帳面 WAR（季均）": round(float(g["war_A"].mean()), 1),
                "真值名單帳面 WAR（季均）": round(float(g["war_B"].mean()), 1),
                "B−A（季均 ± se）": f"{g['d_BA'].mean():+.1f} ± {se_ba * np.sqrt(len(g)):.1f}",
                "B−A 幾季為正": f"{int((g['d_BA'] > 0).sum())}/{len(g)}",
                "BF10（B−A ≠ 0）": bf_str(log10_bf10(g["d_BA"].mean(), se_ba * np.sqrt(len(g)), tau)),
                "B−C（季均 ± se）": f"{g['d_BC'].mean():+.1f} ± {se_bc * np.sqrt(len(g)):.1f}",
                "BF10（B−C ≠ 0）": bf_str(log10_bf10(g["d_BC"].mean(), se_bc * np.sqrt(len(g)), tau)),
                "實際升上去：真值要 − 真值不要（人均六年 WAR）": f"{o_b:+.2f} ({o_se:.2f})" if np.isfinite(o_b) else "—",
                "BF10（可觀測差 ≠ 0）": bf_str(log10_bf10(o_b, o_se, 1.0)) if np.isfinite(o_b) else "—",
                "球隊沒升、真值要的：之後登上率 / 人均 WAR": f"{g['passed_reached'].mean():.2f} / {g['passed_war'].mean():+.2f}",
                "球隊升了、真值不要的：之後登上率 / 人均 WAR": f"{g['picked_reached'].mean():.2f} / {g['picked_war'].mean():+.2f}",
            }
        )
    summ = pd.DataFrame(rows)
    aaa = t[t["level"] == "AAA"].sort_values("season")
    per = pd.DataFrame(
        {
            "季": aaa["season"],
            "3A 人數": aaa["n_pool"],
            "升級 n": aaa["n"],
            "A∩B": aaa["overlap_AB"].round(2),
            "C∩B": aaa["overlap_CB"].round(2),
            "帳面 A": aaa["war_A"].round(1),
            "帳面 C": aaa["war_C"].round(1),
            "帳面 B": aaa["war_B"].round(1),
            "B−A (se)": [f"{x:+.1f} ({s:.1f})" for x, s in zip(aaa["d_BA"], aaa["se_BA"])],
            "實際升：真值要 vs 不要（六年 WAR 人均）": [f"{x:+.2f} vs {y:+.2f}" for x, y in zip(aaa["obs_AandB_mean"], aaa["obs_AnotB_mean"])],
            "沒升但真值要：登上率 / WAR": [f"{x:.2f} / {y:+.2f}" for x, y in zip(aaa["passed_reached"], aaa["passed_war"])],
        }
    )
    return summ, per


def write_md(summ: pd.DataFrame, per: pd.DataFrame, tags: tuple, path: Path) -> str:
    md = [
        f"# 反事實政策模擬：照真值尺升人會差多少 · {tags[0]} × {tags[1]}\n",
        "A = 實際升上去的人；C = 球隊自己的尺挑的同樣人數；B = 真值尺（WAR 段的 E[asinh WAR | 能力]）挑的同樣人數。"
        "帳面 WAR = 名單上每個人 sinh(score_war_asinh) 的總和（模型的期望，WAR 單位，六年）。"
        "B−A 的標準誤來自該季該層級的人重抽；BF 用 Savage–Dickey，先驗 N(0, τ²)，τ = 兩個名單帳面 WAR 平均的 10%（至少 1 WAR）。"
        "可觀測檢查不靠模型帳面：實際升上去的人裡，真值尺也要的和不要的，之後六年實際 fWAR 平均差多少（先驗 τ = 1 WAR）。\n",
        "## 跨季彙整（按層級）\n",
        summ.to_markdown(index=False),
        "",
        "## 3A 逐季\n",
        per.to_markdown(index=False),
        "",
        "讀法：A∩B 低 = 球隊挑的人跟真值尺挑的人不同；B−A 大且 BF 大 = 照真值尺升人帳面上拿更多 WAR；"
        "但帳面是同一個模型算的，真正的證據在可觀測檢查：真值尺要的人升上去之後是不是真的賺比較多。"
        "「球隊沒升、真值要的」那欄是下界——他們那季沒被升，之後才有機會。\n",
        "警語：真值尺（WAR 段）是用同一批人之後的 WAR 估的，可觀測檢查有一部分是樣本內；"
        "乾淨的版本要用 forward_check 的參考季權重（別季學的尺）來排名單，見 docs/findings.md。\n",
    ]
    text = "\n".join(md)
    path.write_text(text)
    return text


def main(out_dir: str, promoted_tag: str, war_tag: str, n_boot: int, seed: int) -> str:
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    frames = []
    for p in sorted(out.glob(f"20*/{war_tag}/eta_M1.parquet")):
        season = int(p.parent.parent.name)
        if not (out / str(season) / promoted_tag / "eta_M1.parquet").exists():
            continue
        frames.append(load_season(out, season, promoted_tag, war_tag))
        print(f"  {season}: {len(frames[-1]):,} 個人季", flush=True)
    if not frames:
        raise SystemExit("找不到同時有升級段與 WAR 段的季")
    d = pd.concat(frames, ignore_index=True)
    t = season_level_table(d, n_boot, rng)
    t["level"] = pd.Categorical(t["level"], LEVELS, ordered=True)
    t = t.sort_values(["level", "season"])
    summ, per = summarize(t)
    t.to_csv(out / f"sim_policy_{promoted_tag}_by_season_level.csv", index=False)
    text = write_md(summ, per, (promoted_tag, war_tag), out / f"sim_policy_{promoted_tag}.md")
    print(text, flush=True)
    return text


def fake_season(rng, season: int, n: int = 400) -> tuple:
    """一季假資料：真值 v，球隊分數跟 v 相關 0.5，升不升照球隊分數；之後 WAR = 登上 × (v + 雜訊)。"""
    v = rng.normal(0, 1, n)
    club = 0.5 * v + rng.normal(0, np.sqrt(1 - 0.25), n)
    promoted = (rng.random(n) < 1 / (1 + np.exp(-(club - 0.3)))).astype(int)
    reached = (rng.random(n) < 0.2 + 0.6 * promoted).astype(int)
    war = reached * (2.0 * v + rng.normal(0, 1.0, n))
    lvl = rng.choice(["AA", "AAA"], n)
    p = pd.DataFrame({"key_mlbam": np.arange(n) + season * 10000, "season": season, "level_main": lvl, "score_promoted": club, "promoted": promoted, "war_fwd": war, "reached_fwd": reached, "war_observable": True, "PA": 400})
    w = pd.DataFrame({"key_mlbam": p["key_mlbam"], "score_war_asinh": np.arcsinh(2.0 * v), "score_preach": 0.3})
    return p, w


def selftest() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for s in (2011, 2012):
            p, w = fake_season(rng, s)
            (out / str(s) / "P").mkdir(parents=True)
            (out / str(s) / "W").mkdir(parents=True)
            p.to_parquet(out / str(s) / "P" / "eta_M1.parquet", index=False)
            w.assign(season=s).to_parquet(out / str(s) / "W" / "eta_M1.parquet", index=False)
        main(tmp, "P", "W", n_boot=50, seed=1)
        t = pd.read_csv(out / "sim_policy_P_by_season_level.csv")
    assert len(t) == 4, t
    assert (t["d_BA"] > 0).all(), t[["season", "level", "d_BA"]]  # 真值名單帳面一定不低於實際名單
    assert (t["overlap_AB"] > 0.3).all() and (t["overlap_AB"] < 1.0).all(), t["overlap_AB"]
    assert (t["obs_AandB_mean"] > t["obs_AnotB_mean"]).all(), t[["obs_AandB_mean", "obs_AnotB_mean"]]  # 真值要的人之後 WAR 比較高
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="反事實政策模擬：照真值尺升人 vs 實際升人，帳面 WAR 與可觀測檢查")
    ap.add_argument("out_dir", nargs="?", default="outputs")
    ap.add_argument("--promoted", default="promoted_f4sed_linear_p2")
    ap.add_argument("--war", default="war_f4sed_linear_p2")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        main(a.out_dir, a.promoted, a.war, a.boot, a.seed)
