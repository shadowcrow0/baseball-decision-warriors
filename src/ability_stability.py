"""個人層次的兩種檢查：校準（拆十組）與能力分數跨季穩不穩。

    python src/ability_stability.py outputs/ --promoted promoted_f4sed_linear_p2 --war war_f4sed_linear_p2

一、校準（每個有結果的球季各一張）
    把人依模型給的機率分成十組（十分位：機率最低的 10%、次低的 10%、……），每組算「模型說的平均機率」和「實際發生的比例」。
    兩者接近 = 模型說三成的人裡真的有三成會發生。這是個人預測對不對的公平問法：不要求知道是哪三成。
    升級用 score_promoted（logit → 機率），登上用 score_preach（六季內登上的機率）。

二、跨季穩定（相鄰兩季都有分數的人）
    同一個人 t 季和 t+1 季各估一次四種能力。WAR 每年會變，能力不該每年變：
    - 兩季分數的 Spearman 相關（特質應該高，運氣應該低）
    - t 季分成十組（十分位），每組看 t+1 季分數的分布：中位數、四分位、留在同一組 / 往上 / 往下的比例
    - 依年齡看 t → t+1 的中位變化：哪種能力會成長、哪種會衰退（成長曲線）
    需要相鄰球季的 eta_M1.parquet（全年份網格跑完才有）；現在有幾對就算幾對。
三、四種能力之外的小特徵（需要 data/model_bat.parquet）
    每個指標在四種能力之外的殘差（層級內標準化的比率，扣掉四種能力與層級能解釋的部分），同一個人兩季的殘差相關：
    年年同號的殘差是特質，單季的是運氣。
四、特質對後來的 WAR
    殘差在模型分數之外對六季 WAR 與登上的係數，逐季同不同號、合併起來多大。小但年年在的，就進小效果目錄。
五、球隊的偏好殘差
    升級殘差 = 實際升不升 − 模型機率。同一個人兩季的殘差相關 = 球隊對這個人有沒有「成績單以外」的固定看法；
    每人多季的平均殘差跟守位、出身的對應，以及它對後來 WAR 的預測：球隊的隱藏判斷是不是真的。
輸出 outputs/<season>/calibration_<tag>.md 與 outputs/stability_<tag>.md。
"""

import argparse
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fit_year import INDICATORS, indicator_tag
from misplaced_profile import logistic_irls

FACTORS = ["contact", "power", "speed", "discipline"]
FACTOR_NAMES = {"contact": "打得到球", "power": "打得遠", "speed": "跑得快", "discipline": "選得到球"}


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def decile_table(prob: pd.Series, y: pd.Series) -> pd.DataFrame:
    """十分位校準表：每組的人數、模型平均機率、實際比例、差。"""
    ok = prob.notna() & y.notna()
    p = prob[ok]
    outcome = y[ok].astype(float)
    group = pd.qcut(p.rank(method="first"), 10, labels=[f"第 {i} 組" for i in range(1, 11)])
    rows = []
    for g in group.cat.categories:
        m = group == g
        rows.append(
            {
                "十分位（機率由低到高）": g,
                "人數": int(m.sum()),
                "模型平均機率": round(float(p[m].mean()), 3),
                "實際比例": round(float(outcome[m].mean()), 3),
                "差（實際 − 模型）": round(float(outcome[m].mean() - p[m].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def calibration(out: Path, season: int, tag_p: str, tag_w: str) -> str | None:
    """一個球季的校準：升級與登上各一張十分位表。回傳 markdown。"""
    fp = out / str(season) / tag_p / "eta_M1.parquet"
    if not fp.exists():
        return None
    e = pd.read_parquet(fp)
    md = [f"# {season} 校準：模型說幾成，實際幾成 · {tag_p}\n"]
    md += [
        "## 隔年升不升（十分位）\n",
        decile_table(expit(e["score_promoted"]), e["promoted"]).to_markdown(index=False),
        "",
    ]
    fw = out / str(season) / tag_w / "eta_M1.parquet"
    if fw.exists():
        w = pd.read_parquet(fw)
        obs = w["war_observable"].astype(bool)
        md += [
            "## 六季內登上（十分位，只算窗口完整的人）\n",
            decile_table(w.loc[obs, "score_preach"], w.loc[obs, "reached_fwd"]).to_markdown(index=False),
            "",
        ]
    md.append("讀法：差接近 0 就是校準好；高機率組的實際比例若偏低，是模型對看好的人太樂觀。")
    text = "\n".join(md)
    (out / str(season) / f"calibration_{tag_p}.md").write_text(text)
    return text


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank().corr(y.rank()))


def transition_table(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """t 季分十組，每組看 t+1 季分數的分布與去向。"""
    ga = pd.qcut(a.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    gb = pd.qcut(b.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    rows = []
    for g in range(1, 11):
        m = ga == g
        nxt = b[m]
        rows.append(
            {
                "t 季十分位": g,
                "人數": int(m.sum()),
                "t+1 季 25%": round(float(nxt.quantile(0.25)), 2),
                "t+1 季中位": round(float(nxt.median()), 2),
                "t+1 季 75%": round(float(nxt.quantile(0.75)), 2),
                "留在同組 ±1": round(float((abs(gb[m] - g) <= 1).mean()), 2),
                "往上超過一組": round(float((gb[m] - g > 1).mean()), 2),
                "往下超過一組": round(float((g - gb[m] > 1).mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def growth_table(pair: pd.DataFrame, factor: str) -> pd.DataFrame:
    """依 t 季年齡分組，t → t+1 的分數變化：中位數與四分位。"""
    d = pair.copy()
    d["change"] = d[f"{factor}_next"] - d[f"{factor}_now"]
    d["年齡"] = pd.cut(d["age"], [0, 20, 22, 24, 26, 99], labels=["≤20", "21–22", "23–24", "25–26", "27+"])
    rows = []
    for g, grp in d.groupby("年齡", observed=True):
        rows.append(
            {
                "t 季年齡": g,
                "人數": len(grp),
                "變化 25%": round(float(grp["change"].quantile(0.25)), 2),
                "變化中位": round(float(grp["change"].median()), 2),
                "變化 75%": round(float(grp["change"].quantile(0.75)), 2),
                "變好的比例": round(float((grp["change"] > 0).mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b


def indicator_residuals(m: pd.DataFrame, eta: pd.DataFrame, min_den: int = 30) -> pd.DataFrame:
    """一個球季：每個指標的比率（層級內標準化）扣掉四種能力與層級能解釋的部分。分母太小的人季留空。"""
    d = m.merge(eta[["key_mlbam"] + [f"eta_{f}" for f in FACTORS]], on="key_mlbam", how="inner")
    levels = sorted(d["level_main"].dropna().unique())
    X = np.column_stack(
        [np.ones(len(d))]
        + [d[f"eta_{f}"].to_numpy(dtype=float) for f in FACTORS]
        + [(d["level_main"] == lv).to_numpy(dtype=float) for lv in levels[1:]]
    )
    out = d[["key_mlbam", "season", "age", "level_main", "PA"]].copy()
    for name, num, den, _, _ in INDICATORS:
        tag = indicator_tag(name)
        if num not in d or den not in d:
            continue
        rate = d[num] / d[den].where(d[den] >= min_den)
        z = rate.groupby(d["level_main"]).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        ok = z.notna().to_numpy()
        resid = np.full(len(d), np.nan)
        if ok.sum() > 50:
            resid[ok] = ols_residual(z[ok].to_numpy(dtype=float), X[ok])
        out[f"r_{tag}"] = resid
    return out


def residual_persistence(res_by_season: dict) -> pd.DataFrame:
    """相鄰兩季、同一個人的指標殘差相關：高 = 特質，低 = 運氣。"""
    seasons = sorted(res_by_season)
    pairs = [(s, s + 1) for s in seasons if s + 1 in res_by_season]
    cols = [c for c in next(iter(res_by_season.values())).columns if c.startswith("r_")]
    rows = []
    for c in cols:
        row = {"指標": c[2:]}
        allx, ally = [], []
        for s, t in pairs:
            a = res_by_season[s][["key_mlbam", c]].dropna()
            b = res_by_season[t][["key_mlbam", c]].dropna()
            pair = a.merge(b, on="key_mlbam", suffixes=("_now", "_next"))
            if len(pair) >= 30:
                row[f"{s}→{t}"] = round(spearman(pair[f"{c}_now"], pair[f"{c}_next"]), 2)
                allx.append(pair[f"{c}_now"])
                ally.append(pair[f"{c}_next"])
        if allx:
            row["全部"] = round(spearman(pd.concat(allx), pd.concat(ally)), 2)
            row["n"] = int(sum(len(x) for x in allx))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("全部", ascending=False) if "全部" in out else out


def residual_vs_truth(res_by_season: dict, war_by_season: dict) -> pd.DataFrame:
    """每個指標的殘差，在模型分數之外對六季 WAR（登上者，asinh）與登上與否（logit）的係數；逐季符號與合併係數。"""
    cols = [c for c in next(iter(res_by_season.values())).columns if c.startswith("r_")]
    rows = []
    for c in cols:
        row = {"指標": c[2:]}
        signs_w, signs_r = [], []
        Xw, yw, Xr, yr = [], [], [], []
        for s, res in sorted(res_by_season.items()):
            if s not in war_by_season:
                continue
            w = war_by_season[s]
            d = res[["key_mlbam", c]].merge(
                w[["key_mlbam", "war_observable", "reached_fwd", "war_fwd", "score_war_asinh", "score_preach"]],
                on="key_mlbam",
            )
            d = d[d["war_observable"].astype(bool) & d[c].notna()]
            r = d[d["reached_fwd"].astype(bool) & d["score_war_asinh"].notna()]
            if len(r) >= 40:
                X = np.column_stack([np.ones(len(r)), r["score_war_asinh"], r[c]])
                y = np.arcsinh(r["war_fwd"].to_numpy(dtype=float))
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                signs_w.append("+" if b[2] > 0 else "−")
                Xw.append(X)
                yw.append(y)
            dd = d[d["score_preach"].notna()]
            if len(dd) >= 100:
                pr = dd["score_preach"].clip(1e-4, 1 - 1e-4)
                X = np.column_stack([np.ones(len(dd)), np.log(pr / (1 - pr)), dd[c]])
                y = dd["reached_fwd"].astype(float).to_numpy()
                b, _ = logistic_irls(X, y)
                signs_r.append("+" if b[2] > 0 else "−")
                Xr.append(X)
                yr.append(y)
        if Xw:
            X = np.vstack(Xw)
            y = np.concatenate(yw)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ b
            se = np.sqrt(np.diag(resid @ resid / max(len(y) - 3, 1) * np.linalg.inv(X.T @ X)))
            row["WAR|登上 合併係數"] = round(float(b[2]), 3)
            row["z"] = round(float(b[2] / se[2]), 1)
            row["逐季符號"] = "".join(signs_w)
        if Xr:
            b, se = logistic_irls(np.vstack(Xr), np.concatenate(yr))
            row["登上 合併係數"] = round(float(b[2]), 3)
            row["z "] = round(float(b[2] / se[2]), 1)
            row["逐季符號 "] = "".join(signs_r)
        rows.append(row)
    out = pd.DataFrame(rows)
    if "z" in out:
        out["_k"] = out["z"].abs().fillna(0)
        out = out.sort_values("_k", ascending=False).drop(columns="_k")
    return out


def promotion_residual_tables(eta_by_season: dict, war_by_season: dict, m_all: pd.DataFrame) -> list:
    """球隊的偏好殘差：跨季相關；每人平均殘差與守位、出身的對應；平均殘差對後來 WAR。"""
    frames = []
    for s, e in eta_by_season.items():
        d = e[["key_mlbam", "season", "score_promoted", "promoted", "PA"]].dropna(subset=["score_promoted"]).copy()
        d = d[d["PA"] >= 100]
        d["resid_prom"] = d["promoted"].astype(float) - expit(d["score_promoted"])
        frames.append(d)
    allr = pd.concat(frames, ignore_index=True)
    seasons = sorted(eta_by_season)
    pairs = [(s, s + 1) for s in seasons if s + 1 in eta_by_season]
    md = []
    rows = []
    for s, t in pairs:
        a = allr[allr["season"] == s][["key_mlbam", "resid_prom"]]
        b = allr[allr["season"] == t][["key_mlbam", "resid_prom"]]
        pair = a.merge(b, on="key_mlbam", suffixes=("_now", "_next"))
        if len(pair) >= 30:
            rows.append(
                {
                    "配對": f"{s}→{t}",
                    "人數": len(pair),
                    "升級殘差的跨季相關": round(spearman(pair["resid_prom_now"], pair["resid_prom_next"]), 2),
                }
            )
    md += [
        "### 同一個人兩季的升級殘差相關（> 0 = 球隊對這個人有成績單以外的固定看法）\n",
        pd.DataFrame(rows).to_markdown(index=False) if rows else "還沒有相鄰球季",
        "",
    ]
    # 每人多季平均殘差
    per = (
        allr.groupby("key_mlbam").agg(n_seasons=("resid_prom", "size"), mean_resid=("resid_prom", "mean")).reset_index()
    )
    per = per[per["n_seasons"] >= 2]
    if len(per) >= 50 and {"positionDB"}.issubset(m_all.columns):
        pos = m_all.sort_values("season").drop_duplicates("key_mlbam", keep="last")[
            ["key_mlbam", "positionDB", "draft_level", "pick_overall"]
        ]
        d = per.merge(pos, on="key_mlbam", how="left")
        p = d["positionDB"].fillna("").astype(str)
        cands = {
            "捕手": p.str.contains("C").astype(float),
            "游擊/二壘": (p.str.contains("SS") | p.str.contains("2B")).astype(float),
            "外野": p.str.contains("OF").astype(float),
            "一壘/指定": (p.str.contains("1B") | p.str.contains("DH")).astype(float),
            "守幾個位置": p.str.count("/").astype(float) + (p != "").astype(float),
            "高中出身": (d["draft_level"] == "HS").astype(float),
            "非選秀": d["pick_overall"].isna().astype(float),
            "順位（log）": np.log(d["pick_overall"].astype(float)),
        }
        rows = [
            {"候選": k, "與平均升級殘差的 Spearman": round(spearman(v, d["mean_resid"]), 3), "n": int(v.notna().sum())}
            for k, v in cands.items()
        ]
        md += [
            f"### 每人多季的平均升級殘差（≥ 2 季，{len(per):,} 人）跟什麼對得上\n",
            pd.DataFrame(rows).to_markdown(index=False),
            "",
        ]
    # 平均殘差對後來 WAR（用最後一季的 WAR 模型分數當控制）
    rows = []
    for s, w in sorted(war_by_season.items()):
        d = per.merge(w[["key_mlbam", "war_observable", "reached_fwd", "war_fwd", "score_war_asinh"]], on="key_mlbam")
        r = d[d["war_observable"].astype(bool) & d["reached_fwd"].astype(bool) & d["score_war_asinh"].notna()]
        if len(r) >= 40:
            X = np.column_stack([np.ones(len(r)), r["score_war_asinh"], r["mean_resid"]])
            y = np.arcsinh(r["war_fwd"].to_numpy(dtype=float))
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ b
            se = np.sqrt(np.diag(resid @ resid / max(len(y) - 3, 1) * np.linalg.inv(X.T @ X)))
            rows.append(
                {
                    "WAR 模型球季": s,
                    "登上者": len(r),
                    "平均升級殘差對 WAR 的係數": round(float(b[2]), 3),
                    "z": round(float(b[2] / se[2]), 1),
                }
            )
    md += [
        "### 球隊多給的機會，後來有沒有變成 WAR（模型分數之外）\n",
        pd.DataFrame(rows).to_markdown(index=False) if rows else "還沒有可用的球季",
        "",
    ]
    return md


def stability(out: Path, tag_p: str, tag_w: str, data_dir: str | None, min_pa: int = 100) -> str:
    """所有相鄰球季配對：同一個人兩季的四種能力；加上指標殘差、特質對 WAR、球隊偏好殘差三張表。"""
    seasons = sorted(int(p.name) for p in out.iterdir() if p.name.isdigit() and (p / tag_p / "eta_M1.parquet").exists())
    pairs = [(s, s + 1) for s in seasons if s + 1 in seasons]
    md = [f"# 能力分數跨季穩不穩 · {tag_p}\n", f"有分數的球季：{seasons}；相鄰配對：{pairs}\n"]
    eta_by_season = {s: pd.read_parquet(out / str(s) / tag_p / "eta_M1.parquet") for s in seasons}
    war_by_season = {
        s: pd.read_parquet(out / str(s) / tag_w / "eta_M1.parquet")
        for s in seasons
        if (out / str(s) / tag_w / "eta_M1.parquet").exists()
    }
    if data_dir:
        m_all = pd.read_parquet(Path(data_dir) / "model_bat.parquet")
        pos = pd.read_parquet(
            Path(data_dir) / "minors_bat_clean.parquet", columns=["key_mlbam", "season", "positionDB"]
        ).drop_duplicates(["key_mlbam", "season"])
        m_all = m_all.merge(pos, on=["key_mlbam", "season"], how="left")
        res_by_season = {s: indicator_residuals(m_all[m_all["season"] == s], eta_by_season[s]) for s in seasons}
        md += [
            "## 三、四種能力之外的小特徵：指標殘差的跨季相關（高 = 特質）\n",
            residual_persistence(res_by_season).to_markdown(index=False),
            "",
        ]
        md += [
            "## 四、特質對後來的 WAR 與登上（模型分數之外；逐季符號 + 合併係數）\n",
            residual_vs_truth(res_by_season, war_by_season).to_markdown(index=False),
            "",
        ]
        md += ["## 五、球隊的偏好殘差\n"] + promotion_residual_tables(eta_by_season, war_by_season, m_all)
    if not pairs:
        md.append("還沒有相鄰的球季，跨季配對等全年份網格跑完。")
        text = "\n".join(md)
        (out / f"stability_{tag_p}.md").write_text(text)
        return text
    frames = []
    for s, t in pairs:
        a = pd.read_parquet(out / str(s) / tag_p / "eta_M1.parquet")
        b = pd.read_parquet(out / str(t) / tag_p / "eta_M1.parquet")
        a = a[a["PA"] >= min_pa]
        b = b[b["PA"] >= min_pa]
        cols = ["key_mlbam", "age"] + [f"eta_{f}" for f in FACTORS]
        pair = a[cols].merge(
            b[["key_mlbam"] + [f"eta_{f}" for f in FACTORS]], on="key_mlbam", suffixes=("_now", "_next")
        )
        pair = pair.rename(
            columns={f"eta_{f}_now": f"{f}_now" for f in FACTORS} | {f"eta_{f}_next": f"{f}_next" for f in FACTORS}
        )
        pair["pair"] = f"{s}→{t}"
        frames.append(pair)
    allp = pd.concat(frames, ignore_index=True)
    rows = []
    for f in FACTORS:
        row = {"能力": FACTOR_NAMES[f]}
        for p in frames:
            row[p["pair"].iloc[0]] = round(spearman(p[f"{f}_now"], p[f"{f}_next"]), 2)
        row["全部"] = round(spearman(allp[f"{f}_now"], allp[f"{f}_next"]), 2)
        rows.append(row)
    md += [
        f"## 同一個人兩季分數的 Spearman 相關（兩季打席都 ≥ {min_pa}，共 {len(allp):,} 人季配對）\n",
        "高 = 穩定的特質；低 = 那一季的運氣或正在變。",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
    ]
    for f in FACTORS:
        md += [
            f"## {FACTOR_NAMES[f]}：t 季分十組，t+1 季去了哪裡（全部配對）\n",
            transition_table(allp[f"{f}_now"], allp[f"{f}_next"]).to_markdown(index=False),
            "",
            f"### {FACTOR_NAMES[f]}：依年齡的變化（成長曲線）\n",
            growth_table(allp, f).to_markdown(index=False),
            "",
        ]
    md.append("讀法：十分位表的對角線（留在同組 ±1）越高越穩；成長表的中位變化隨年齡由正轉負，就是那種能力的高峰年齡。")
    text = "\n".join(md)
    (out / f"stability_{tag_p}.md").write_text(text)
    return text


def main(out_dir: str, tag_p: str, tag_w: str, data_dir: str | None):
    out = Path(out_dir)
    for p in sorted(out.iterdir()):
        if p.name.isdigit():
            text = calibration(out, int(p.name), tag_p, tag_w)
            if text:
                print(text)
                print(f"-> {out / p.name / f'calibration_{tag_p}.md'}\n")
    print(stability(out, tag_p, tag_w, data_dir))
    print(f"-> {out / f'stability_{tag_p}.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="個人層次：校準十分位表、能力分數跨季穩定")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("--promoted", required=True, help="升級模型的 tag")
    p.add_argument("--war", required=True, help="WAR 模型的 tag")
    p.add_argument("--data", default=None, help="資料夾（含 model_bat.parquet）；給了才算三、四、五那三張表")
    args = p.parse_args()
    main(args.out_dir, args.promoted, args.war, args.data)
