"""清洗小聯盟逐季逐層級資料，產出建模用的三張表。

    python src/clean_minors.py data/ reports/

規則（與討論一致：分母進模型、尾巴留著、只切母體）：
  1. 投手判定：同季「面對打者數 TBF > 打席 PA」者視為投手，其打者列移除；
     反之 PA > TBF 的野手，其投手列移除（大比分上去投一局的野手）。
  2. 大聯盟首登年之後的小聯盟列標記 post_debut（復健／下放），不刪。
  3. 其餘全部保留，不做任何出賽數門檻。

輸出：
  data/minors_bat_clean.parquet   打者，一列 = 一人 × 一季 × 一層級，加 post_debut、level_rank
  data/minors_pit_clean.parquet   投手，同上
  data/player_season.parquet      一人一季：該季最高/最低層級、層級數、總 PA/IP、年齡、
                                  下一季去向 next_status ∈ {promoted, same, demoted, mlb, gone, censored}
  reports/clean_summary.md        各步驟列數與去向分布
注意：2020 年小聯盟停賽，2019 的「下一季」是 2021；next_gap 欄記錄間隔年數。
"""

import sys
from pathlib import Path

import pandas as pd

LEVEL_RANK = {"DSL": 0, "R": 1, "CPX": 1, "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6}


def main(data_dir: str, out_dir: str):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bat = pd.read_parquet(data / "minors_bat.parquet")
    pit = pd.read_parquet(data / "minors_pit.parquet")
    ids = pd.read_parquet(data / "ids.parquet")[["key_mlbam", "mlb_played_first"]].dropna().drop_duplicates("key_mlbam")
    ids["key_mlbam"] = ids["key_mlbam"].astype("Int64")
    md = ["# 清洗摘要\n", f"原始：打者 {len(bat):,} 列，投手 {len(pit):,} 列\n"]

    # 1. 投手 / 野手判定（以人季為單位）
    pa = bat.groupby(["key_mlbam", "season"])["PA"].sum().rename("season_PA")
    tbf = pit.groupby(["key_mlbam", "season"])["TBF"].sum().rename("season_TBF")
    role = pd.concat([pa, tbf], axis=1).fillna(0)
    role["is_pitcher"] = role["season_TBF"] > role["season_PA"]
    role = role.reset_index()
    bat = bat.merge(role[["key_mlbam", "season", "is_pitcher"]], on=["key_mlbam", "season"], how="left")
    pit = pit.merge(role[["key_mlbam", "season", "is_pitcher"]], on=["key_mlbam", "season"], how="left")
    n_bat_drop, n_pit_drop = int(bat["is_pitcher"].sum()), int((~pit["is_pitcher"]).sum())
    bat = bat[~bat["is_pitcher"]].drop(columns="is_pitcher")
    pit = pit[pit["is_pitcher"]].drop(columns="is_pitcher")
    md.append(f"規則 1：移除打者表中的投手列 {n_bat_drop:,}，移除投手表中的野手列 {n_pit_drop:,}\n")
    md.append(
        f"清洗後：打者 {len(bat):,} 列（{bat['key_mlbam'].nunique():,} 人），投手 {len(pit):,} 列（{pit['key_mlbam'].nunique():,} 人）\n"
    )

    # 2. post_debut 標記
    for df in (bat, pit):
        df.drop(columns=[c for c in ("mlb_played_first",) if c in df.columns], inplace=True)
    bat = bat.merge(ids, on="key_mlbam", how="left")
    pit = pit.merge(ids, on="key_mlbam", how="left")
    for df in (bat, pit):
        df["post_debut"] = df["mlb_played_first"].notna() & (df["season"] > df["mlb_played_first"])
    md.append(
        f"規則 2：post_debut 標記 打者 {int(bat['post_debut'].sum()):,} 列，投手 {int(pit['post_debut'].sum()):,} 列（保留）\n"
    )
    bat.to_parquet(data / "minors_bat_clean.parquet", index=False)
    pit.to_parquet(data / "minors_pit_clean.parquet", index=False)

    # 3. 一人一季路徑表
    def season_table(df: pd.DataFrame, role: str, expo: str) -> pd.DataFrame:
        g = df.groupby(["key_mlbam", "season"])
        t = pd.DataFrame(
            {
                "role": role,
                "max_rank": g["level_rank"].max(),
                "min_rank": g["level_rank"].min(),
                "n_levels": g["level"].nunique(),
                "exposure": g[expo].sum(),
                "age": g["Age"].min(),
                "org": g["org"].first(),
                "post_debut": g["post_debut"].any(),
                "mlb_played_first": g["mlb_played_first"].first(),
            }
        ).reset_index()
        return t

    ps = pd.concat([season_table(bat, "bat", "PA"), season_table(pit, "pit", "IP")], ignore_index=True)
    seasons = sorted(ps["season"].unique())
    next_season = {s: next((t for t in seasons if t > s), None) for s in seasons}
    ps["next_season"] = ps["season"].map(next_season)
    ps["next_gap"] = ps["next_season"] - ps["season"]
    nxt = ps[["key_mlbam", "role", "season", "max_rank"]].rename(
        columns={"season": "next_season", "max_rank": "next_max_rank"}
    )
    ps = ps.merge(nxt, on=["key_mlbam", "role", "next_season"], how="left")
    last_season = max(seasons)

    def status(r):
        if pd.notna(r["mlb_played_first"]) and r["mlb_played_first"] <= r["season"] + 1:
            return "mlb"  # 本季或下一季登上大聯盟，優先於小聯盟層級比較
        if r["season"] == last_season:
            return "censored"
        if pd.notna(r["next_max_rank"]):
            if r["next_max_rank"] > r["max_rank"]:
                return "promoted"
            if r["next_max_rank"] < r["max_rank"]:
                return "demoted"
            return "same"
        return "gone"

    ps["next_status"] = ps.apply(status, axis=1)
    # 消失拆成「之後有回來」與「再也沒出現」；回來的記間隔年數（獨立聯盟／海外／傷停）
    last_seen = ps.groupby(["key_mlbam", "role"])["season"].max().rename("last_seen")
    ps = ps.merge(last_seen, on=["key_mlbam", "role"], how="left")
    gone = ps["next_status"] == "gone"
    ps.loc[gone & (ps["last_seen"] > ps["season"]), "next_status"] = "gone_return"
    ps.loc[gone & (ps["last_seen"] <= ps["season"]), "next_status"] = "gone_exit"
    # 回來的人：下一次出現的球季
    nxt_seen = ps[["key_mlbam", "role", "season"]].sort_values("season")
    nxt_seen["return_season"] = nxt_seen.groupby(["key_mlbam", "role"])["season"].shift(-1)
    ps = ps.merge(nxt_seen, on=["key_mlbam", "role", "season"], how="left")
    ps["gap_years"] = (ps["return_season"] - ps["season"]).where(ps["next_status"] == "gone_return")
    ps = ps.drop(columns=["last_seen"])
    ps["promoted"] = ps["next_status"].isin(["promoted", "mlb"])
    ps.to_parquet(data / "player_season.parquet", index=False)

    md.append(
        f"\n## 一人一季路徑表：{len(ps):,} 人季（打者 {int((ps['role']=='bat').sum()):,}，投手 {int((ps['role']=='pit').sum()):,}）\n"
    )
    md.append("### 下一季去向 × 最高層級（打者，排除 post_debut）\n")
    rank_name = {0: "DSL", 1: "R/CPX", 2: "A-", 3: "A", 4: "A+", 5: "AA", 6: "AAA"}

    def crosstab(role: str) -> str:
        sub = ps[(ps["role"] == role) & ~ps["post_debut"]]
        ct = pd.crosstab(sub["max_rank"], sub["next_status"], normalize="index").round(3) * 100
        ct["n"] = sub.groupby("max_rank").size()
        ct.index = ct.index.map(rank_name)
        return ct.to_markdown()

    md.append(crosstab("bat"))
    md.append("\n### 下一季去向 × 最高層級（投手，排除 post_debut）\n")
    md.append(crosstab("pit"))
    md.append("\n### 消失：回來 vs 離開（排除 post_debut 與最後一季）\n")
    sub = ps[~ps["post_debut"] & (ps["season"] < last_season)]
    g = sub.groupby("role")["next_status"].value_counts().unstack().fillna(0).astype(int)
    md.append(g[[c for c in ["gone_exit", "gone_return"] if c in g]].to_markdown())
    md.append(
        "\n間隔年數分布（gone_return）："
        + sub.loc[sub["next_status"] == "gone_return", "gap_years"].value_counts().sort_index().to_dict().__repr__()
        + "\n"
    )
    md.append("\n### 離開率 gone_exit % × 年份 × 層級（打者）\n")
    sub_b = sub[sub["role"] == "bat"]
    tbl = (
        pd.crosstab(
            sub_b["season"],
            sub_b["max_rank"].map(rank_name),
            values=(sub_b["next_status"] == "gone_exit"),
            aggfunc="mean",
        )
        * 100
    ).round(1)
    md.append(tbl.to_markdown())
    md.append("\n### 升級率（promoted 或 mlb）× 年份（打者）\n")
    sub = ps[
        (ps["role"] == "bat") & ~ps["post_debut"] & (ps["next_status"] != "censored") & (ps["season"] < last_season)
    ]
    md.append((sub.groupby("season")["promoted"].mean().round(3) * 100).to_frame("升級率 %").to_markdown())
    (out / "clean_summary.md").write_text("\n".join(md))
    print("\n".join(md[:5]))
    print(f"-> {out / 'clean_summary.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data", sys.argv[2] if len(sys.argv) > 2 else "reports")
