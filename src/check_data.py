"""資料完整性檢查：每個資料集對照「應該有的範圍」，輸出 PASS / WARN。

    python src/check_data.py data/ reports/
輸出 reports/data_check.md
"""

import sys
from pathlib import Path

import pandas as pd

# 選秀：各年輪數（B-Ref）
DRAFT_ROUNDS = {
    **{y: 50 for y in range(2005, 2012)},
    **{y: 40 for y in range(2012, 2020)},
    2020: 5,
    **{y: 20 for y in range(2021, 2026)},
}
# 小聯盟：各年應有的層級
MINOR_LEVELS_OLD = {"AAA", "AA", "A+", "A", "A-", "R"}  # 2006–2019
MINOR_LEVELS_NEW = {"AAA", "AA", "A+", "A", "CPX", "DSL"}  # 2021–
# Statcast：各季大致列數下限（整季約 70 萬球；2020 縮短賽季約 26 萬）
STATCAST_MIN = {y: 600_000 for y in range(2015, 2027)}
STATCAST_MIN[2020] = 240_000


def main(data_dir: str, out_dir: str):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(ds, item, status, detail=""):
        rows.append({"資料集": ds, "項目": item, "狀態": status, "說明": detail})

    # ---- 選秀 ----
    f = data / "draft.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        for y, exp in DRAFT_ROUNDS.items():
            sub = d[d["year"] == y]
            if sub.empty:
                add("選秀", f"{y}", "WARN", "整年缺")
                continue
            nr = sub["round"].nunique()
            ok = nr >= exp
            add("選秀", f"{y}", "PASS" if ok else "WARN", f"{nr} 輪 / 預期 {exp}，{len(sub):,} picks")
        add(
            "選秀",
            "欄位",
            "PASS" if {"pick_overall", "bonus", "level", "bwar_career", "name"} <= set(d.columns) else "WARN",
            ", ".join(d.columns[:12]),
        )
        add("選秀", "簽約金覆蓋", "INFO", f"{100 * d['bonus'].notna().mean():.1f}%（B-Ref 有簽約金的 pick）")
    else:
        add("選秀", "draft.parquet", "WARN", "檔案不存在")

    # ---- 小聯盟 ----
    for stats in ("bat", "pit"):
        f = data / f"minors_{stats}.parquet"
        if not f.exists():
            add(f"小聯盟 {stats}", "檔案", "WARN", "不存在")
            continue
        d = pd.read_parquet(f, columns=["season", "level", "key_mlbam"])
        for y in list(range(2006, 2020)) + list(range(2021, 2027)):
            sub = d[d["season"] == y]
            exp = MINOR_LEVELS_OLD if y <= 2020 else MINOR_LEVELS_NEW
            have = set(sub["level"].unique())
            missing = exp - have
            add(
                f"小聯盟 {stats}",
                f"{y}",
                "PASS" if not missing and len(sub) > 2000 else "WARN",
                f"{len(sub):,} 列，層級 {sorted(have)}" + (f"，缺 {sorted(missing)}" if missing else ""),
            )
        add(
            f"小聯盟 {stats}",
            "key_mlbam 缺值",
            "PASS" if d["key_mlbam"].isna().mean() < 0.01 else "WARN",
            f"{100 * d['key_mlbam'].isna().mean():.2f}%",
        )

    # ---- FanGraphs 大聯盟 ----
    for stats, lo in (("batting", 1000), ("pitching", 550)):
        f = data / f"fg_{stats}.parquet"
        if not f.exists():
            add(f"FG 大聯盟 {stats}", "檔案", "WARN", "不存在")
            continue
        d = pd.read_parquet(f, columns=["Season", "IDfg", "WAR"])
        for y in range(2005, 2026):
            n = int((d["Season"] == y).sum())
            add(f"FG 大聯盟 {stats}", f"{y}", "PASS" if n >= lo else "WARN", f"{n:,} 人")
        add(f"FG 大聯盟 {stats}", "欄位", "PASS" if "WAR" in d.columns else "WARN", "")

    # ---- Chadwick ----
    f = data / "ids.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        add(
            "Chadwick",
            "列數",
            "PASS" if len(d) > 20000 else "WARN",
            f"{len(d):,} 人，key_mlbam 非空 {d['key_mlbam'].notna().sum():,}",
        )
    else:
        add("Chadwick", "ids.parquet", "WARN", "不存在")

    # ---- Statcast ----
    for y, lo in STATCAST_MIN.items():
        f = data / f"statcast_{y}.parquet"
        if not f.exists():
            add("Statcast", f"{y}", "WARN", "合併檔不存在")
            continue
        d = pd.read_parquet(f, columns=["game_date"])
        d["game_date"] = pd.to_datetime(d["game_date"])
        reg = d[(d["game_date"].dt.month >= 4) & (d["game_date"] < pd.Timestamp(f"{y}-09-28"))]  # 例行賽；10 月是季後賽
        wk = reg.groupby(reg["game_date"].dt.to_period("W")).size()
        full = pd.period_range(wk.index.min(), wk.index.max(), freq="W")
        gaps = [str(p) for p in full if p not in wk.index]
        low = [str(p) for p, n in wk.items() if n < wk.median() / 4 and p not in (wk.index.min(), wk.index.max())]
        status = "PASS" if len(d) >= lo and not gaps and not low else "WARN"
        detail = f"{len(d):,} 球，{d['game_date'].min().date()} ~ {d['game_date'].max().date()}"
        if gaps:
            detail += f"，空週 {gaps}"
        if low:
            detail += f"，異常少的週 {low}"
        if y == 2026:
            detail += "（本季尚未結束，列數可能偏低屬正常）"
        add("Statcast", f"{y}", status, detail)

    df = pd.DataFrame(rows)
    n_warn = int((df["狀態"] == "WARN").sum())
    md = [
        f"# 資料完整性檢查\n",
        f"**{n_warn} 個 WARN**，{int((df['狀態'] == 'PASS').sum())} 個 PASS\n",
        df.to_markdown(index=False),
    ]
    (out / "data_check.md").write_text("\n".join(md))
    print(df[df["狀態"] != "PASS"].to_string(index=False) if n_warn else "全部 PASS")
    print(f"-> {out / 'data_check.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data", sys.argv[2] if len(sys.argv) > 2 else "reports")
