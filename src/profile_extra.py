"""補充剖析：缺值的年份分布、投手打擊污染、加權層級平均、年齡分布。

    python src/profile_extra.py data/ reports/
輸出 reports/profile_extra.md
"""

import sys
from pathlib import Path

import pandas as pd

LEVEL_ORDER = ["DSL", "R", "CPX", "A-", "A", "A+", "AA", "AAA"]


def by_season_coverage(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return (df.groupby("season")[cols].apply(lambda g: g.notna().mean().round(2) * 100)).astype(int)


def main(data_dir: str, out_dir: str):
    data, out = Path(data_dir), Path(out_dir)
    bat = pd.read_parquet(data / "minors_bat.parquet")
    pit = pd.read_parquet(data / "minors_pit.parquet")
    md = ["# 補充剖析\n"]

    # 1. 缺值的年份分布
    md.append("## 1. 各年份有值比例 %（打者）\n")
    md.append(by_season_coverage(bat, ["GB%", "SwStr%", "Pitches", "Pull%", "wRC+", "Spd"]).to_markdown())
    md.append("\n## 1b. 各年份有值比例 %（投手）\n")
    md.append(by_season_coverage(pit, ["GB%", "SwStr%", "Pitches", "xFIP", "HLD"]).to_markdown())

    # 2. 投手打擊污染：打者表中同季也在投手表的人
    pit_keys = pit[["key_mlbam", "season"]].drop_duplicates().assign(is_pitcher=True)
    b = bat.merge(pit_keys, on=["key_mlbam", "season"], how="left")
    b["is_pitcher"] = b["is_pitcher"].fillna(False).astype(bool)
    md.append("\n## 2. 打者表裡同季也出現在投手表的比例與其 wRC+\n")
    t = (
        b.groupby("level")
        .apply(
            lambda g: pd.Series(
                {
                    "列數": len(g),
                    "投手列%": round(100 * g["is_pitcher"].mean(), 1),
                    "投手列中位PA": g.loc[g["is_pitcher"], "PA"].median(),
                    "投手列wRC+": round(g.loc[g["is_pitcher"], "wRC+"].mean(), 1),
                    "非投手wRC+": round(g.loc[~g["is_pitcher"], "wRC+"].mean(), 1),
                    "非投手PA加權wRC+": round(
                        (g.loc[~g["is_pitcher"], "wRC+"] * g.loc[~g["is_pitcher"], "PA"]).sum()
                        / g.loc[~g["is_pitcher"], "PA"].sum(),
                        1,
                    ),
                }
            )
        )
        .reindex(LEVEL_ORDER)
        .dropna(how="all")
    )
    md.append(t.to_markdown())
    md.append("\n### 2b. 投手列比例 × 年份（AA 與 AAA）\n")
    t2 = b[b["level"].isin(["AA", "AAA"])].groupby(["season", "level"])["is_pitcher"].mean().unstack().round(2)
    md.append(t2.to_markdown())

    # 3. 非投手、PA>=50 的層級平均（加權）
    nb = b[~b["is_pitcher"] & (b["PA"] >= 50)]
    md.append("\n## 3. 非投手且 PA≥50 的層級平均（PA 加權）\n")
    cols = ["AVG", "OBP", "SLG", "ISO", "BB%", "K%", "wOBA", "wRC+", "SwStr%"]
    w = nb.groupby("level").apply(
        lambda g: pd.Series({c: (g[c] * g["PA"]).sum() / g["PA"].sum() for c in cols if c in g})
    )
    w["n"] = nb.groupby("level").size()
    md.append(w.round(3).reindex(LEVEL_ORDER).dropna(how="all").to_markdown())
    md.append("\n### 3b. PA 分布（非投手）：各層級 PA<20 的比例\n")
    md.append(
        b[~b["is_pitcher"]]
        .groupby("level")["PA"]
        .apply(lambda s: round(100 * (s < 20).mean(), 1))
        .reindex(LEVEL_ORDER)
        .to_frame("PA<20 %")
        .to_markdown()
    )

    # 4. 年齡分布
    md.append("\n## 4. 年齡分位數（非投手打者 / 投手）\n")
    q = lambda s: s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(1)
    a1 = b[~b["is_pitcher"]].groupby("level")["Age"].apply(q).unstack().reindex(LEVEL_ORDER)
    a2 = pit.groupby("level")["Age"].apply(q).unstack().reindex(LEVEL_ORDER)
    md.append("打者\n\n" + a1.to_markdown() + "\n\n投手\n\n" + a2.to_markdown())

    # 5. 一人一季跨幾個層級
    md.append("\n## 5. 一人一季出現在幾個層級（打者）\n")
    md.append(bat.groupby(["key_mlbam", "season"]).size().value_counts().sort_index().to_frame("人季數").to_markdown())

    (out / "profile_extra.md").write_text("\n".join(md))
    print(f"-> {out / 'profile_extra.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data", sys.argv[2] if len(sys.argv) > 2 else "reports")
