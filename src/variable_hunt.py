"""四把尺之外還有什麼：候選變數一個一個加進線性整合，用前三季估、第 t 季測，看對下一季的預測有沒有變準（不取樣）。

    python src/variable_hunt.py outputs/ data/ --promoted promoted_f4sed_linear_p2 --war war_f4sed_linear_p2 [--workers 16]

基底 = 四種能力 + 球探評價 + 年齡 + 打席（層級內各自配邏輯迴歸，跟 decision_style 一樣）。
三個結果：隔年升不升（球隊）；六季內登上、登上者六季 WAR（真相）。
每個候選：加進基底後，測試季的 log-loss（或 WAR 的均方誤差）比基底少多少；幾季變準、合併起來值多少。
前向選擇：一次加一個合併最有用的，加到不再變準為止；留下的就是四把尺之外真的多帶資訊的東西。
候選：十五個指標在四種能力之外的殘差、跨季進步、年資、使用方式、犧牲觸擊、守位、守幾個位置。
輸出 outputs/variable_hunt_<promoted tag>.md。
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent))
from decision_style import LEVELS, fit_logit, logit_predict, log_loss
from fit_year import INDICATORS, add_extra, indicator_tag
from club_features import add_position

BASE = ["eta_contact", "eta_power", "eta_speed", "eta_discipline", "zeta_scout", "age", "log_pa"]
CANDIDATES = {
    "d_k": "三振率進步",
    "d_hr": "全壘打率進步",
    "d_bb": "保送率進步",
    "years_in": "年資",
    "pa_share": "打席佔比",
    "has_prev": "有上一季",
    "repeat_level": "重複層級",
    "multi_level": "跨層級",
    "sh_rate_f": "犧牲觸擊率",
    "undrafted": "非選秀",
    "lvl_HS": "高中出身",
    "C": "守位：捕手",
    "SS2B": "守位：游擊/二壘",
    "OF": "守位：外野",
    "1BDH": "守位：一壘/指定",
    "3B": "守位：三壘",
    "n_positions": "守幾個位置",
}
SHARED = {}
COVERAGE = {}  # 每個指標被四種能力與層級解釋的 R²（每季一個，之後平均）


def load(out: Path, data: Path, tag_p: str, tag_w: str, min_pa: int = 100) -> pd.DataFrame:
    """每人季：基底、候選、三個結果。指標殘差 = 層級內標準化的比率扣掉四種能力與層級能解釋的部分（每季各自算）。"""
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = add_position(m, data)
    m["sh_rate_f"] = m["sh_rate"].fillna(0.0)
    m["undrafted"] = m["pick_overall"].isna().astype(float)
    m["lvl_HS"] = (m["draft_level"] == "HS").astype(float)
    frames = []
    for p in sorted(out.glob(f"20*/{tag_p}/eta_M1.parquet")):
        season = int(p.parts[-3])
        e = pd.read_parquet(p)[["key_mlbam", "season"] + BASE[:5] + ["score_promoted"]]
        w_path = out / str(season) / tag_w / "eta_M1.parquet"
        d = m[m["season"] == season].merge(e, on=["key_mlbam", "season"], how="inner")
        if w_path.exists():
            w = pd.read_parquet(w_path)[["key_mlbam", "score_preach", "score_war_asinh"]]
            d = d.merge(w, on="key_mlbam", how="left")
        else:
            d["score_preach"] = np.nan
            d["score_war_asinh"] = np.nan
        d = d[(d["PA"] >= min_pa) & d["level_main"].isin(LEVELS)].copy()
        d["zeta_scout"] = d["zeta_scout"].fillna(0.0)
        d["log_pa"] = np.log(d["PA"])
        # 指標殘差
        levels = sorted(d["level_main"].unique())
        X = np.column_stack(
            [np.ones(len(d))]
            + [d[f].to_numpy(dtype=float) for f in BASE[:4]]
            + [(d["level_main"] == lv).to_numpy(dtype=float) for lv in levels[1:]]
        )
        for name, num, den, _, _ in INDICATORS:
            rate = d[num] / d[den].where(d[den] >= 30)
            z = rate.groupby(d["level_main"]).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            ok = z.notna().to_numpy()
            r = np.zeros(len(d))
            if ok.sum() > 50:
                b = np.linalg.lstsq(X[ok], z[ok].to_numpy(dtype=float), rcond=None)[0]
                r[ok] = z[ok].to_numpy(dtype=float) - X[ok] @ b
                COVERAGE.setdefault("r_" + indicator_tag(name), []).append(1.0 - r[ok].var() / z[ok].var())
            d["r_" + indicator_tag(name)] = r
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    cols = BASE + list(CANDIDATES) + [c for c in d.columns if c.startswith("r_") and c not in ("r_k", "r_hr", "r_bb")]
    for c in cols:
        d[c] = d[c].astype(float)
        d[c] = d.groupby(["season", "level_main"])[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
    d["y_prom"] = d["promoted"].astype(float)
    d["y_reach"] = np.where(d["war_observable"].astype(bool), d["reached_fwd"].astype(float), np.nan)
    d["y_war"] = np.where(
        d["war_observable"].astype(bool) & d["reached_fwd"].astype(bool), np.arcsinh(d["war_fwd"].astype(float)), np.nan
    )
    return d


def residual_persistence(d: pd.DataFrame) -> dict:
    """每個指標殘差：同一個人相鄰兩季的 Spearman 相關（合併所有相鄰配對）。"""
    out = {}
    seasons = sorted(d["season"].unique())
    for c in [c for c in d.columns if c.startswith("r_") and c not in ("r_k", "r_hr", "r_bb")]:
        xs, ys = [], []
        for s0 in seasons:
            if s0 + 1 not in seasons:
                continue
            a = d.loc[d["season"] == s0, ["key_mlbam", c]]
            b = d.loc[d["season"] == s0 + 1, ["key_mlbam", c]]
            pair = a.merge(b, on="key_mlbam")
            xs.append(pair[c + "_x"])
            ys.append(pair[c + "_y"])
        if xs:
            x = pd.concat(xs)
            y = pd.concat(ys)
            out[c] = float(x.rank().corr(y.rank()))
    return out


def oos_loss(d: pd.DataFrame, season: int, train: list, features: list, outcome: str) -> float:
    """前三季估、第 season 季測：logit 的 log-loss，或 WAR 的均方誤差。層級內各自配。"""
    tr = d[d["season"].isin(train) & d[outcome].notna()]
    te = d[(d["season"] == season) & d[outcome].notna()]
    losses, weights = [], []
    for lvl in LEVELS:
        a = tr[tr["level_main"] == lvl]
        b = te[te["level_main"] == lvl]
        if len(a) < 60 or len(b) < 10:
            continue
        Xa = a[features].to_numpy(dtype=float)
        Xb = b[features].to_numpy(dtype=float)
        ya = a[outcome].to_numpy(dtype=float)
        yb = b[outcome].to_numpy(dtype=float)
        if outcome == "y_war":
            Xa1 = np.column_stack([np.ones(len(Xa)), Xa])
            Xb1 = np.column_stack([np.ones(len(Xb)), Xb])
            coef = np.linalg.solve(Xa1.T @ Xa1 + 1e-2 * np.eye(Xa1.shape[1]), Xa1.T @ ya)
            losses.append(float(((yb - Xb1 @ coef) ** 2).mean()))
        else:
            if ya.sum() < 10 or (1 - ya).sum() < 10:
                continue
            coef = fit_logit(Xa, ya)
            losses.append(log_loss(yb, logit_predict(coef, Xb)))
        weights.append(len(b))
    if not losses:
        return np.nan
    return float(np.average(losses, weights=weights))


def one_season(args):
    """一個測試季：每個候選對三個結果的改善量。"""
    season, train, cands = args
    d = SHARED["d"]
    out = {}
    for outcome in ("y_prom", "y_reach", "y_war"):
        base = oos_loss(d, season, train, BASE, outcome)
        out[(outcome, "base")] = base
        for c in cands:
            out[(outcome, c)] = base - oos_loss(d, season, train, BASE + [c], outcome)
    return season, out


def forward_select(d: pd.DataFrame, jobs: list, cands: list, outcome: str, max_steps: int = 5) -> list:
    """前向選擇：每一步加進「所有測試季合併改善最多」的候選，改善 ≤ 0 就停。"""
    chosen, log = [], []
    current = {s: oos_loss(d, s, tr, BASE, outcome) for s, tr, _ in jobs}
    for _ in range(max_steps):
        best, best_gain, best_losses = None, 0.0, None
        for c in cands:
            if c in chosen:
                continue
            losses = {s: oos_loss(d, s, tr, BASE + chosen + [c], outcome) for s, tr, _ in jobs}
            gain = float(np.nanmean([current[s] - losses[s] for s in losses]))
            if gain > best_gain:
                best, best_gain, best_losses = c, gain, losses
        if best is None:
            break
        chosen.append(best)
        n_better = int(sum(current[s] - best_losses[s] > 0 for s in best_losses))
        log.append(
            {
                "步驟": len(chosen),
                "加進去的": CANDIDATES.get(best, best),
                "平均每季變準": round(best_gain, 4),
                "變準的季數": f"{n_better} / {len(best_losses)}",
            }
        )
        current = best_losses
    return log


def main(out_dir: str, data_dir: str, tag_p: str, tag_w: str, workers: int):
    out, data = Path(out_dir), Path(data_dir)
    d = load(out, data, tag_p, tag_w)
    SHARED["d"] = d
    resid_cols = [
        c for c in d.columns if c.startswith("r_") and c not in ("r_k", "r_hr", "r_bb")
    ]  # 後三個是 add_extra 的原始比率，不是殘差
    cands = list(CANDIDATES) + resid_cols
    names = dict(CANDIDATES)
    names.update({c: "殘差：" + c[2:] for c in resid_cols})
    seasons = sorted(d["season"].unique())
    jobs = [
        (s, [t for t in range(s - 3, s) if t in seasons], cands)
        for s in seasons
        if any(t in seasons for t in range(s - 3, s))
    ]
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            results = pool.map(one_season, jobs)
    else:
        results = [one_season(j) for j in jobs]
    md = [
        f"# 四把尺之外還有什麼 · {tag_p} / {tag_w}\n",
        "基底 = 四種能力 + 球探評價 + 年齡 + 打席（層級內邏輯迴歸）。每個候選加進基底，前三季估、當季測。表中是「加了之後測試季少掉的 log-loss」（WAR 段是均方誤差），正 = 變準。\n",
    ]
    tabs = {}
    for outcome, label in (
        ("y_prom", "球隊：隔年升不升"),
        ("y_reach", "真相：六季內登上"),
        ("y_war", "真相：登上者六季 WAR（asinh，均方誤差）"),
    ):
        rows = []
        for c in cands:
            gains = [r[1].get((outcome, c), np.nan) for r in results]
            gains = [g for g in gains if np.isfinite(g)]
            if not gains:
                continue
            rows.append(
                {
                    "候選": names[c],
                    "平均每季變準": round(float(np.mean(gains)), 4),
                    "變準的季數": f"{int(sum(g > 0 for g in gains))} / {len(gains)}",
                    "最差的一季": round(float(min(gains)), 4),
                }
            )
        tab = pd.DataFrame(rows).sort_values("平均每季變準", ascending=False)
        tabs[outcome] = tab
        bases = [r[1].get((outcome, "base"), np.nan) for r in results]
        md += [
            f"## {label}（基底平均 {np.nanmean(bases):.4f}，{len(results)} 個測試季）\n",
            tab.head(15).to_markdown(index=False),
            "",
            "### 前向選擇\n",
            pd.DataFrame(forward_select(d, jobs, cands, outcome)).to_markdown(index=False),
            "",
        ]
    # 兩邊對照
    cross = tabs["y_prom"][["候選", "平均每季變準", "變準的季數"]].rename(
        columns={"平均每季變準": "球隊 變準", "變準的季數": "球隊 季數"}
    )
    for outcome, lab in (("y_reach", "登上"), ("y_war", "WAR")):
        cross = cross.merge(
            tabs[outcome][["候選", "平均每季變準", "變準的季數"]].rename(
                columns={"平均每季變準": f"{lab} 變準", "變準的季數": f"{lab} 季數"}
            ),
            on="候選",
            how="outer",
        )

    def verdict(r):
        club = r["球隊 變準"] > 0.001
        truth = (r["登上 變準"] > 0.001) or (r["WAR 變準"] > 0.001)
        if club and truth:
            return "兩邊都有：四把尺之外的第五樣"
        if club:
            return "球隊偏好"
        if truth:
            return "被放掉"
        return ""

    cross["判讀"] = cross.apply(verdict, axis=1)
    persist = residual_persistence(d)
    cover_name = {names[c]: round(float(np.mean(v)), 2) for c, v in COVERAGE.items()}
    persist_name = {names[c]: round(v, 2) for c, v in persist.items()}
    cross["四把尺涵蓋（R²）"] = cross["候選"].map(cover_name)
    cross["沒涵蓋那塊的跨季相關"] = cross["候選"].map(persist_name)
    cross = cross.sort_values("球隊 變準", ascending=False)
    md += [
        "## 對照：四把尺涵蓋多少、沒涵蓋的穩不穩、球隊用不用、真相要不要\n",
        "關鍵變數 = 涵蓋低、沒涵蓋那塊跨季相關高、至少一邊變準。門檻：平均每季變準 > 0.001。\n",
        cross.to_markdown(index=False),
        "",
    ]
    text = "\n".join(md)
    (out / f"variable_hunt_{tag_p}.md").write_text(text)
    print(text)
    print(f"-> {out / f'variable_hunt_{tag_p}.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="四把尺之外還有什麼：候選變數的樣本外增量")
    p.add_argument("out_dir")
    p.add_argument("data_dir")
    p.add_argument("--promoted", required=True)
    p.add_argument("--war", required=True)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    main(a.out_dir, a.data_dir, a.promoted, a.war, a.workers)
