"""建模表（打者）：一人一季一列，指標以「分子 + 分母」形式放，對應 docs/model_spec.md。

    python src/build_model_table.py data/ reports/

讀：data/minors_bat_clean.parquet、data/player_season.parquet、data/draft_linked.parquet、data/ids.parquet
寫：data/model_bat.parquet、reports/model_table_summary.md

跨層級的季：分子分母跨層加總，level 取打席最多的層級（主要層級），n_levels 記跨幾層。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RANK_NAME = {0: "DSL", 1: "R/CPX", 2: "A-", 3: "A", 4: "A+", 5: "AA", 6: "AAA"}


def main(data_dir: str, out_dir: str):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bat = pd.read_parquet(data / "minors_bat_clean.parquet")
    ps = pd.read_parquet(data / "player_season.parquet")
    ps = ps[ps["role"] == "bat"]

    # ---- 指標的分子與分母（先從比例欄還原成計數，再跨層加總）----
    b = bat.copy()
    for c in ("PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "HBP", "SO", "SB", "CS", "GDP", "SH", "Pitches", "Balls"):
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0)
    b["XBH"] = b["2B"] + b["3B"] + b["HR"]
    b["OB"] = b["H"] + b["BB"] + b["HBP"]
    # 揮棒數與揮空數：FanGraphs 給 SwStr%（揮空/投球數）與 Pitches；揮棒數用 Strikes 近似不可靠，改用 whiff/pitches
    b["whiff"] = (pd.to_numeric(b["SwStr%"], errors="coerce") * b["Pitches"]).round()
    # 擊球類型：GB%、Pull% 的分母是擊球數 BIP ≈ AB − SO + SF（FanGraphs 用 batted balls）
    b["BIP"] = (b["AB"] - b["SO"] + pd.to_numeric(b.get("SF", 0), errors="coerce").fillna(0)).clip(lower=0)
    b["GB"] = (pd.to_numeric(b["GB%"], errors="coerce") * b["BIP"]).round()
    b["PULL"] = (pd.to_numeric(b["Pull%"], errors="coerce") * b["BIP"]).round()
    # 擊球品質（第五把尺）的指標：平飛球數 / 擊球數、內野高飛數 / 高飛球數、場內安打 / 場內擊球（BABIP）
    b["LD"] = (pd.to_numeric(b["LD%"], errors="coerce") * b["BIP"]).round()
    b["FB"] = (pd.to_numeric(b["FB%"], errors="coerce") * b["BIP"]).round()
    b["IFFB"] = (pd.to_numeric(b["IFFB%"], errors="coerce") * b["FB"]).round()
    b["HIP"] = (b["H"] - b["HR"]).clip(lower=0)  # 場內安打
    b["BIPH"] = (b["AB"] - b["SO"] - b["HR"] + pd.to_numeric(b.get("SF", 0), errors="coerce").fillna(0)).clip(
        lower=0
    )  # BABIP 的分母
    b["SBA"] = b["SB"] + b["CS"]
    b["OBT"] = b["1B"] + b["BB"] + b["HBP"]  # 上壘次數（盜壘嘗試的分母）

    b["BALLS"] = b["Balls"]
    num_cols = [
        "PA",
        "AB",
        "H",
        "OB",
        "BB",
        "SO",
        "XBH",
        "HR",
        "SB",
        "SBA",
        "OBT",
        "3B",
        "GDP",
        "SH",
        "whiff",
        "Pitches",
        "BALLS",
        "GB",
        "PULL",
        "BIP",
        "LD",
        "FB",
        "IFFB",
        "HIP",
        "BIPH",
    ]
    g = b.groupby(["key_mlbam", "season"])
    agg = g[num_cols].sum(min_count=1)
    # 主要層級 = 打席最多的那層
    main_lvl = (
        b.sort_values("PA", ascending=False)
        .drop_duplicates(["key_mlbam", "season"])
        .set_index(["key_mlbam", "season"])["level_rank"]
        .rename("level_rank_main")
    )
    agg = agg.join(main_lvl).reset_index()
    # 擊球類型資料 2006 年沒有 → 該年 whiff/GB/PULL 設為缺值（分母也設 0，模型裡跳過）
    for c in ("whiff", "GB", "PULL", "BALLS", "Pitches", "LD", "FB", "IFFB"):
        agg.loc[agg["season"] == 2006, c] = np.nan

    # ---- 路徑表：結果與其他欄 ----
    keep = [
        "key_mlbam",
        "season",
        "max_rank",
        "min_rank",
        "n_levels",
        "age",
        "org",
        "post_debut",
        "mlb_played_first",
        "next_status",
        "promoted",
        "gap_years",
    ]
    m = ps[keep].merge(agg, on=["key_mlbam", "season"], how="inner")

    # ---- 選秀共變數 ----
    dr = pd.read_parquet(data / "draft_linked.parquet")
    dr = dr[dr["key_mlbam"].notna() & dr["signed"]].sort_values("year").drop_duplicates("key_mlbam", keep="last")
    dr = dr[["key_mlbam", "year", "pick_overall", "round", "bonus", "level"]].rename(
        columns={"year": "draft_year", "level": "draft_level"}
    )
    m = m.merge(dr, on="key_mlbam", how="left")
    m["drafted"] = m["draft_year"].notna()
    m["draft_level"] = m["draft_level"].fillna("UNDRAFTED")
    m["log_pick"] = np.log(m["pick_overall"].fillna(1500))  # 非選秀給一個比最後順位更大的值
    m["log_bonus"] = np.log1p(m["bonus"])  # 缺值保留為 NaN，模型用 bonus_known 指標
    m["bonus_known"] = m["bonus"].notna()

    # ---- 球團怎麼用你（間接選擇訊號，結構模型共變數）----
    m["log_pa"] = np.log1p(m["PA"])
    m["pa_share"] = m["PA"] / m.groupby(["season", "level_rank_main"])["PA"].transform(
        "median"
    )  # 相對同層級同年中位打席
    m["sh_rate"] = m["SH"] / m["PA"].replace(0, np.nan)

    # ---- 年齡相對層級 ----
    med_age = m.groupby(["season", "level_rank_main"])["age"].transform("median")
    m["age_rel"] = m["age"] - med_age

    # ---- 球員在體系內第幾季、索引 ----
    m = m.sort_values(["key_mlbam", "season"])
    m["season_idx_in_career"] = m.groupby("key_mlbam").cumcount()
    m["player_idx"] = pd.factorize(m["key_mlbam"])[0]
    m["org_idx"] = pd.factorize(m["org"].fillna("UNK"))[0]
    m["level_main"] = m["level_rank_main"].map(RANK_NAME)
    m["censored"] = m["next_status"].isin(["censored"]) | (m["next_status"] == "gone_return")
    # 結構模型只用「尚未登上大聯盟、且下一季去向已觀測」的人季；post_debut 列仍進測量模型
    m["structural"] = ~m["post_debut"] & ~m["censored"]

    # ---- 結果變數 B：往後 H 季的大聯盟 fWAR（打擊 + 投球）；沒登上 = 0 ----
    H = 6
    fg = pd.concat(
        [
            pd.read_parquet(data / "fg_batting.parquet", columns=["key_mlbam", "Season", "WAR"]),
            pd.read_parquet(data / "fg_pitching.parquet", columns=["key_mlbam", "Season", "WAR"]),
        ]
    )
    fg = fg.dropna(subset=["key_mlbam"]).astype({"key_mlbam": "int64", "Season": "int64"})
    fg_last = int(fg["Season"].max())
    war_by = fg.groupby(["key_mlbam", "Season"])["WAR"].sum()

    def war_window(pid, s):
        tot = 0.0
        for y in range(s + 1, s + H + 1):
            tot += war_by.get((pid, y), 0.0)
        return tot

    m["war_fwd"] = [
        war_window(int(pid), int(s)) if pd.notna(pid) else 0.0 for pid, s in zip(m["key_mlbam"], m["season"])
    ]
    m["reached_fwd"] = (
        m["mlb_played_first"].notna()
        & (m["mlb_played_first"] <= m["season"] + H)
        & (m["mlb_played_first"] >= m["season"])
    )
    m["war_observable"] = (m["season"] + H) <= fg_last  # 六季窗口完整落在資料內
    m["war_structural"] = m["structural"] & m["war_observable"] & ~m["post_debut"]
    m.to_parquet(data / "model_bat.parquet", index=False)

    # ---- 摘要 ----
    md = [
        f"# 建模表（打者）\n",
        f"{len(m):,} 人季，{m['key_mlbam'].nunique():,} 人，{m['season'].min()}–{m['season'].max()}\n",
        f"選秀出身 {int(m['drafted'].sum()):,} 人季；簽約金已知 {int(m['bonus_known'].sum()):,}；post_debut {int(m['post_debut'].sum()):,}；censored {int(m['censored'].sum()):,}\n",
        "### 指標分母覆蓋（分母 > 0 的人季比例 %）\n",
    ]
    den = {
        "AB": "擊球/長打/全壘打/雙殺打",
        "PA": "上壘/保送/三振",
        "H": "長打(XBH/H)/三壘打(3B/H)",
        "SBA": "盜壘成功率",
        "OBT": "盜壘嘗試率(SBA/OBT)",
        "Pitches": "揮空",
        "BIP": "滾地/拉打",
    }
    md.append(
        pd.Series({f"{k}（{v}）": round(100 * (m[k] > 0).mean(), 1) for k, v in den.items()})
        .to_frame("%")
        .to_markdown()
    )
    md.append("\n### 各指標比率的中位數 × 主要層級（粗看單調性）\n")
    r = pd.DataFrame(
        {
            "AVG": m["H"] / m["AB"],
            "OBP": m["OB"] / m["PA"],
            "BB%": m["BB"] / m["PA"],
            "K%": m["SO"] / m["PA"],
            "XBH/H": m["XBH"] / m["H"],
            "HR/AB": m["HR"] / m["AB"],
            "SB%": m["SB"] / m["SBA"],
            "SBA/OBT": m["SBA"] / m["OBT"],
            "3B/H": m["3B"] / m["H"],
            "GDP/AB": m["GDP"] / m["AB"],
            "Ball%": m["BALLS"] / m["Pitches"],
            "P/PA": m["Pitches"] / m["PA"],
            "whiff%": m["whiff"] / m["Pitches"],
            "GB%": m["GB"] / m["BIP"],
            "Pull%": m["PULL"] / m["BIP"],
            "level": m["level_main"],
        }
    )
    big = m["PA"] >= 100
    md.append(
        r[big].groupby("level").median().round(3).reindex(list(RANK_NAME.values())).dropna(how="all").to_markdown()
    )
    md.append(f"\n### 結果分布 × 主要層級（structural 遮罩內，{int(m['structural'].sum()):,} 人季）\n")
    ms = m[m["structural"]]
    md.append(
        (pd.crosstab(ms["level_main"], ms["next_status"], normalize="index").round(3) * 100)
        .reindex(list(RANK_NAME.values()))
        .dropna(how="all")
        .to_markdown()
    )
    md.append(
        f"\n升級率（promoted ∪ mlb）：{100 * ms['promoted'].mean():.1f}%；遮罩外：post_debut {int(m['post_debut'].sum()):,}、censored {int(m['censored'].sum()):,}"
    )
    mw = m[m["war_structural"]]
    md.append(
        f"\n### 結果 B：往後 {H} 季 fWAR（war_structural 內，{len(mw):,} 人季，球季 ≤ {int(mw['season'].max()) if len(mw) else '-'}）\n"
    )
    t = mw.groupby("level_main").agg(
        n=("war_fwd", "size"),
        登上一軍率=("reached_fwd", "mean"),
        WAR均值=("war_fwd", "mean"),
        WAR中位_登上者=(
            "war_fwd",
            lambda x: (
                x[mw.loc[x.index, "reached_fwd"]].median() if mw.loc[x.index, "reached_fwd"].any() else float("nan")
            ),
        ),
        WAR_p90=("war_fwd", lambda x: x.quantile(0.9)),
    )
    t["登上一軍率"] = (t["登上一軍率"] * 100).round(1)
    md.append(t.round(2).reindex(list(RANK_NAME.values())).dropna(how="all").to_markdown())
    md.append("\n### 欄位\n" + ", ".join(m.columns))
    (out / "model_table_summary.md").write_text("\n".join(md))
    print("\n".join(md[:3]))
    print(f"-> {data / 'model_bat.parquet'}, {out / 'model_table_summary.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data", sys.argv[2] if len(sys.argv) > 2 else "reports")
