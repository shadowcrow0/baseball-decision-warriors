"""四種能力對 Statcast 物理量的外部驗證。

    python src/validate_abilities.py data/ outputs/ <season> <tag>  [--years 3]

拿 <season> 年 fit 的四個分數（outputs/<season>/<tag>/eta_M1.parquet），對同一個人登上大聯盟後前幾年（預設 3 年）的
Savant 排行榜平均值算 Spearman 相關。預期：跑得快 ↔ 衝刺速度、打得遠 ↔ 擊球初速 / barrel%、
打得到球 ↔ 揮空率（負）/ 區內接觸、選得到球 ↔ 追打壞球率（負）。輸出 reports/validate_abilities_<season>_<tag>.md。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ABIL = {"eta_contact": "打得到球", "eta_power": "打得遠", "eta_speed": "跑得快", "eta_discipline": "選得到球"}
# 候選欄位（用子字串找，Savant 欄名偶爾改）
TARGETS = {
    "衝刺速度 ft/s": ("sprint_speed", ["hp_to_1b", "sprint_speed"]),
    "平均擊球初速": ("exitvelo", ["avg_hit_speed", "exit_velocity"]),
    "barrel %": ("exitvelo", ["brl_percent", "barrel"]),
    "硬擊球 %": ("exitvelo", ["ev95percent", "hard_hit"]),
    "揮空率（百分位）": ("percentile", ["whiff_percent", "whiff"]),
    "追打壞球（百分位）": ("percentile", ["chase_percent", "chase"]),
    "三振率（百分位）": ("percentile", ["k_percent"]),
    "保送率（百分位）": ("percentile", ["bb_percent"]),
    "衝刺速度（百分位）": ("percentile", ["sprint_speed"]),
    "xwOBA": ("expected", ["est_woba", "xwoba"]),
}


def pick(df, cands):
    for c in cands:
        hit = [x for x in df.columns if c in x]
        if hit:
            return hit[0]
    return None


def main(data_dir, out_dir, season, tag, years=3):
    data, out = Path(data_dir), Path(out_dir)
    eta = pd.read_parquet(out / str(season) / tag / "eta_M1.parquet")
    m = pd.read_parquet(data / "model_bat.parquet", columns=["key_mlbam", "season", "mlb_played_first", "PA"])
    m = m[m.season == season]
    eta = eta.merge(m, on=["key_mlbam", "season"], how="left", suffixes=("", "_m"))
    sv = pd.read_parquet(data / "savant_leaderboards.parquet")
    md = [
        f"# 四種能力 × Statcast 物理量：{season} 年 {tag}\n",
        f"分數來自 {season} 年小聯盟成績單；物理量是同一人登上大聯盟後前 {years} 年的 Savant 排行榜平均。Spearman 相關，n = 同時有兩者的人數。\n",
    ]
    rows = []
    for label, (board, cands) in TARGETS.items():
        b = sv[sv.board == board]
        col = pick(b, cands)
        if col is None:
            rows.append({"物理量": label, "欄位": "找不到", "n": 0})
            continue
        b = b[["key_mlbam", "season", col]].copy()
        b[col] = pd.to_numeric(b[col], errors="coerce")
        b = b.merge(eta[["key_mlbam", "mlb_played_first"]].drop_duplicates(), on="key_mlbam")
        b = b[(b.season >= b.mlb_played_first) & (b.season < b.mlb_played_first + years)]
        agg = b.groupby("key_mlbam")[col].mean().rename("phys")
        d = eta.merge(agg, on="key_mlbam")
        d = d[d.PA >= 100]
        r = {"物理量": label, "欄位": col, "n": len(d)}
        for k, name in ABIL.items():
            r[name] = round(float(d[[k, "phys"]].corr(method="spearman").iloc[0, 1]), 2) if len(d) >= 20 else np.nan
        rows.append(r)
    tab = pd.DataFrame(rows)
    md += [
        tab.to_markdown(index=False),
        "",
        "讀法：每一列是一個物理量，四欄是四種能力各自的相關。對角線（跑得快×衝刺速度、打得遠×初速、打得到球×揮空率為負、選得到球×追打為負）應該最大，非對角線接近零。",
    ]
    rp = Path("reports") / f"validate_abilities_{season}_{tag}.md"
    rp.write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {rp}")
    return {"四種能力 × 物理量（Spearman）": (tab, None)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="四種能力對 Statcast 物理量的外部驗證")
    p.add_argument("data_dir")
    p.add_argument("out_dir")
    p.add_argument("season", type=int)
    p.add_argument("tag", help="例如 promoted_f4sed_linear_p2")
    p.add_argument("--years", type=int, default=3, help="登上大聯盟後看前幾年（預設 3）")
    args = p.parse_args()
    main(args.data_dir, args.out_dir, args.season, args.tag, args.years)
