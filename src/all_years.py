"""全年份整理：每一季各自估的模型放在一起看（不取樣，只讀每季的係數表、球團表、summary）。

    python src/all_years.py outputs/ --promoted promoted_f4sed_linear_p2 --war war_f4sed_linear_p2

四張表：
    一、跨年同號：每個係數在每一季的估計；幾年正、幾年負；合併估計（各季以 1/sd² 加權平均）與它的標準誤；判定。
        升級段一張，登上時間段與 WAR|登上段各一張。
    二、趨勢：係數對年份的斜率（每年變多少、標準誤）：哪些權重在十幾年間真的在變。
    三、球團截距：每隊每季的 u_org 疊起來：平均、標準誤、幾季為正、相鄰兩季的相關（習慣穩不穩）；
        再用 400 個「球團 × 季」的點解釋截距：人才平均（θ_org）對截距的係數，含球團固定效應（同一隊年年比）。
    四、模型比較：每季 M0 / M1 / Mh 的 LOO 差。
輸出 outputs/all_years_<promoted tag>.md 與 outputs/all_years_<promoted tag>.json（給投影片畫曲線）。
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_NAMES = {"contact": "打得到球", "power": "打得遠", "speed": "跑得快", "discipline": "選得到球"}
# 要追的係數：struct_M1.csv 的列名 → 顯示名
PROMOTED_ROWS = {
    "beta_contact[0]": "打得到球",
    "beta_power[0]": "打得遠",
    "beta_speed[0]": "跑得快",
    "beta_discipline[0]": "選得到球",
    "beta_scout": "球探評價",
    "undrafted": "非選秀",
    "lvl_HS": "高中出身",
    "years_in": "年資",
    "has_prev": "有上一季",
    "repeat_level": "重複層級",
    "d_hr": "全壘打率進步",
    "d_k": "三振率進步",
    "d_bb": "保送率進步",
    "sh_rate_f": "犧牲觸擊率",
    "direct_SBpct": "盜壘成功率（direct）",
    "direct_3B_H": "三壘打率（direct）",
    "direct_GDP_AB": "雙殺率（direct）",
    "direct_P_PA": "每打席看球數（direct）",
    "direct_BBpct": "保送率（direct）",
    "tau_org": "球團截距 τ",
}


def read_struct(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def coef_table(structs: dict, rows: dict, prefix: str = "") -> tuple[pd.DataFrame, dict]:
    """structs: {season: struct df}。回傳跨年表與（給 json 的）每季數值。"""
    seasons = sorted(structs)
    out_rows, curves = [], {}
    for key, name in rows.items():
        full = prefix + key
        means, sds = {}, {}
        for s in seasons:
            df = structs[s]
            if full in df.index:
                means[s] = float(df.loc[full, "mean"])
                sds[s] = float(df.loc[full, "sd"])
        if not means:
            continue
        m = np.array([means[s] for s in means])
        sd = np.array([sds[s] for s in means])
        w = 1.0 / np.maximum(sd, 1e-6) ** 2
        pooled = float((w * m).sum() / w.sum())
        pooled_se = float(np.sqrt(1.0 / w.sum()))
        n_pos, n_neg = int((m > 0).sum()), int((m < 0).sum())
        # 趨勢：係數對年份的加權最小平方斜率
        yrs = np.array(list(means), dtype=float)
        if len(yrs) >= 4:
            X = np.column_stack([np.ones(len(yrs)), yrs - yrs.mean()])
            W = np.diag(w)
            cov = np.linalg.inv(X.T @ W @ X)
            b = cov @ X.T @ W @ m
            slope, slope_se = float(b[1]), float(np.sqrt(cov[1, 1]))
        else:
            slope, slope_se = np.nan, np.nan
        if n_pos == len(m) or n_neg == len(m):
            verdict = "全部同號"
        elif max(n_pos, n_neg) >= 0.8 * len(m):
            verdict = "多數同號"
        else:
            verdict = "不穩"
        row = {"係數": name}
        for s in seasons:
            row[str(s)] = round(means[s], 2) if s in means else ""
        row["正 / 負"] = f"{n_pos} / {n_neg}"
        row["合併"] = round(pooled, 3)
        row["合併 se"] = round(pooled_se, 3)
        row["每年變化"] = f"{slope:+.3f} ({slope_se:.3f})" if np.isfinite(slope) else ""
        row["判定"] = verdict + ("，有趨勢" if np.isfinite(slope) and abs(slope) > 2 * slope_se else "")
        out_rows.append(row)
        curves[name] = {
            "seasons": list(means),
            "mean": [round(v, 3) for v in means.values()],
            "sd": [round(v, 3) for v in sds.values()],
        }
    return pd.DataFrame(out_rows), curves


def club_tables(out: Path, tag: str, col: str, label: str) -> list:
    """每隊每季的 u_org 疊起來。col = org_M1.csv 裡的欄名。"""
    frames = []
    for p in sorted(out.glob(f"20*/{tag}/org_M1.csv")):
        season = int(p.parts[-3])
        o = pd.read_csv(p)
        if col not in o.columns:
            continue
        o = o[o["org"] != "UNK"]
        f = o[["org", col]].rename(columns={col: "u"})
        f["season"] = season
        for k in FACTOR_NAMES:
            if f"能力平均 {k}" in o.columns:
                f[f"theta_{k}"] = o[f"能力平均 {k}"].to_numpy()
        frames.append(f)
    if not frames:
        return [f"### {label}\n", "還沒有球團表。", ""]
    d = pd.concat(frames, ignore_index=True)
    seasons = sorted(d["season"].unique())
    per = d.groupby("org")["u"].agg(["mean", "std", "size", lambda x: (x > 0).mean()])
    per.columns = ["平均", "sd", "季數", "為正的比例"]
    per["平均的 se"] = per["sd"] / np.sqrt(per["季數"])
    per["|平均| / se"] = (per["平均"] / per["平均的 se"]).abs()
    per = per.sort_values("平均", ascending=False).round(3)
    # 相鄰兩季的相關
    wide = d.pivot(index="org", columns="season", values="u")
    cors = []
    for a, b in zip(seasons[:-1], seasons[1:]):
        if a in wide and b in wide:
            pair = wide[[a, b]].dropna()
            if len(pair) >= 15:
                cors.append(
                    {
                        "配對": f"{a}→{b}",
                        "隊數": len(pair),
                        "Spearman": round(float(pair[a].rank().corr(pair[b].rank())), 2),
                    }
                )
    # 截距 ← 人才平均：合併 OLS 與球團固定效應（同一隊年年比）
    md = [
        f"### {label}：每隊疊 {len(seasons)} 季（{seasons[0]}–{seasons[-1]}）\n",
        "平均 = 這隊跨季的截距平均；|平均| / se > 2 的隊才算分得出來。為正的比例 = 幾成的球季這隊比聯盟多升（或早上、WAR 高）。\n",
        per.head(8).to_markdown(),
        "",
        "……",
        "",
        per.tail(8).to_markdown(),
        "",
        f"分得出來的隊（|平均| / se > 2）：{int((per['|平均| / se'] > 2).sum())} / {len(per)}。\n",
        "相鄰兩季截距的相關（高 = 習慣穩定）：",
        "",
        pd.DataFrame(cors).to_markdown(index=False) if cors else "不足",
        "",
    ]
    theta_cols = [c for c in d.columns if c.startswith("theta_")]
    if theta_cols and len(d) >= 60:
        y = d["u"].to_numpy(dtype=float)
        Xp = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(dtype=float) for c in theta_cols])
        bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        rp = y - Xp @ bp
        sep = np.sqrt(np.diag(rp @ rp / max(len(y) - Xp.shape[1], 1) * np.linalg.inv(Xp.T @ Xp)))
        dm = d.copy()
        for c in ["u"] + theta_cols:
            dm[c] = dm[c] - dm.groupby("org")[c].transform("mean")
        Xw = np.column_stack([dm[c].to_numpy(dtype=float) for c in theta_cols])
        yw = dm["u"].to_numpy(dtype=float)
        bw, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        rw = yw - Xw @ bw
        sew = np.sqrt(np.diag(rw @ rw / max(len(yw) - Xw.shape[1] - d["org"].nunique(), 1) * np.linalg.inv(Xw.T @ Xw)))
        rows = []
        for i, c in enumerate(theta_cols):
            rows.append(
                {
                    "農場人才平均": FACTOR_NAMES[c[6:]],
                    "跨隊合併（係數 (se)）": f"{bp[i + 1]:+.2f} ({sep[i + 1]:.2f})",
                    "同一隊年年比（係數 (se)）": f"{bw[i]:+.2f} ({sew[i]:.2f})",
                }
            )
        md += [
            f"截距 ← 農場人才平均（{len(d)} 個球團 × 季）。負 = 人才越多越不升（塞車）。「同一隊年年比」扣掉每隊的固定習慣，只看同一隊人才多的年份是不是升得少。\n",
            pd.DataFrame(rows).to_markdown(index=False),
            "",
        ]
    return md


def loo_table(out: Path, tag: str) -> pd.DataFrame:
    rows = []
    for p in sorted(out.glob(f"20*/{tag}/summary.md")):
        season = int(p.parts[-3])
        text = p.read_text()
        m = re.search(r"### PSIS-LOO(.*?)###", text, re.S)
        if not m:
            continue
        vals = {}
        for line in m.group(1).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] in ("M0", "M1", "Mh") and len(cells) >= 5:
                vals[cells[0]] = float(cells[4])
        if vals:
            best = max(vals, key=vals.get)
            rows.append(
                {
                    "球季": season,
                    "最好的": best,
                    "M0 相對最好": vals.get("M0"),
                    "M1 相對最好": vals.get("M1"),
                    "Mh 相對最好": vals.get("Mh"),
                }
            )
    return pd.DataFrame(rows)


def main(out_dir: str, tag_p: str, tag_w: str):
    out = Path(out_dir)
    structs_p = {int(p.parts[-3]): read_struct(p) for p in sorted(out.glob(f"20*/{tag_p}/struct_M1.csv"))}
    structs_w = {int(p.parts[-3]): read_struct(p) for p in sorted(out.glob(f"20*/{tag_w}/struct_M1.csv"))}
    md = [
        f"# 全年份整理 · {tag_p} / {tag_w}\n",
        f"升級段有 {len(structs_p)} 季：{sorted(structs_p)}；WAR 段有 {len(structs_w)} 季：{sorted(structs_w)}。每季各自估，沒有共用參數；這裡只是把結果排在一起。\n",
    ]
    curves = {}
    tab, cv = coef_table(structs_p, PROMOTED_ROWS)
    curves["promoted"] = cv
    md += [
        "## 一、跨年同號：升級段（球隊隔年升不升）\n",
        "每格是該季的係數（高一個標準差的對數勝算變化）。合併 = 各季以 1/sd² 加權平均。每年變化 = 係數對年份的斜率。\n",
        tab.to_markdown(index=False),
        "",
    ]
    rows_t = {k: v for k, v in PROMOTED_ROWS.items() if k != "tau_org"}
    tab_t, cv_t = coef_table(structs_w, rows_t, prefix="t_")
    curves["debut"] = cv_t
    md += ["## 一之二、登上時間段（負 = 上得快）\n", tab_t.to_markdown(index=False), ""]
    tab_w, cv_w = coef_table(structs_w, rows_t, prefix="w_")
    curves["war"] = cv_w
    md += ["## 一之三、WAR|登上段（asinh WAR）\n", tab_w.to_markdown(index=False), ""]
    md += ["## 二、趨勢\n", "上面三張表的「每年變化」欄：|斜率| > 2 se 的標了「有趨勢」。投影片的曲線用 json 畫。\n"]
    md += ["## 三、球團截距\n"]
    md += club_tables(out, tag_p, "升級 u_org", "升級截距（同樣的人，這隊多升多少）")
    md += club_tables(out, tag_w, "登上時間 t_u_org（負 = 快）", "登上時間截距（負 = 這隊叫人上得快）")
    md += club_tables(out, tag_w, "WAR|登上 w_u_org", "WAR 截距（這隊上去的人打得比能力預期好多少）")
    md += [
        "## 四、模型比較（PSIS-LOO，相對該季最好的模型；0 = 最好）\n",
        "升級段：",
        "",
        loo_table(out, tag_p).to_markdown(index=False),
        "",
        "WAR 段：",
        "",
        loo_table(out, tag_w).to_markdown(index=False),
        "",
    ]
    text = "\n".join(md)
    (out / f"all_years_{tag_p}.md").write_text(text)
    (out / f"all_years_{tag_p}.json").write_text(json.dumps(curves, ensure_ascii=False, indent=1))
    print(text)
    print(f"-> {out / f'all_years_{tag_p}.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="全年份整理：跨年同號、趨勢、球團截距、模型比較")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("--promoted", required=True)
    p.add_argument("--war", required=True)
    a = p.parse_args()
    main(a.out_dir, a.promoted, a.war)
