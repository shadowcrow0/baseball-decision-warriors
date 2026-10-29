"""變數字典：對小聯盟表每個數值欄位算分布特徵，並建議似然函數。

    python src/profile_variables.py data/ reports/

輸出（可進 git 的小檔）：
    reports/vardict_bat.csv / vardict_pit.csv   每欄一列
    reports/level_means_bat.csv / _pit.csv      各層級平均，看層級位移
    reports/profile.md                          上面兩張表的 markdown 版

建議規則（人工覆核用）：
    有分母的比例  -> binomial        計數          -> poisson / negbin（var/mean > 1.5）
    0~1 無分母    -> beta 或 logit    正值偏態      -> 改回計數 或 gamma/lognormal
    無界連續      -> normal           零比例 > 30%  -> 加註 zero-inflated
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = {
    "key_mlbam",
    "season",
    "level_rank",
    "playerids",
    "minormasterid",
    "AffId",
    "level",
    "name",
    "team",
    "org",
    "aLevel",
    "MaxAge",
}
# 比例欄位對應的分母（打者 / 投手）
DENOM = {
    "bat": {
        "AVG": "AB",
        "OBP": "PA",
        "SLG": "AB",
        "BB%": "PA",
        "K%": "PA",
        "ISO": "AB",
        "BABIP": "AB",
        "GB%": "PA",
        "LD%": "PA",
        "FB%": "PA",
        "IFFB%": "PA",
        "HR/FB": "HR",
        "Oppo%": "PA",
        "Pull%": "PA",
        "Cent%": "PA",
        "SwStr%": "Pitches",
    },
    "pit": {
        "K%": "TBF",
        "BB%": "TBF",
        "GB%": "TBF",
        "LD%": "TBF",
        "FB%": "TBF",
        "IFFB%": "TBF",
        "HR/FB": "HR",
        "BABIP": "TBF",
        "LOB%": "TBF",
        "SwStr%": "Pitches",
        "K-BB%": "TBF",
    },
}
COUNT_LIKE = {
    "G",
    "AB",
    "PA",
    "H",
    "1B",
    "2B",
    "3B",
    "HR",
    "R",
    "RBI",
    "BB",
    "IBB",
    "SO",
    "HBP",
    "SF",
    "SH",
    "GDP",
    "SB",
    "CS",
    "Balls",
    "Strikes",
    "Pitches",
    "W",
    "L",
    "SV",
    "GS",
    "CG",
    "ShO",
    "HLD",
    "BS",
    "TBF",
    "ER",
    "WP",
    "BK",
    "HR",
    "BB",
    "SO",
    "HBP",
    "IP",
}
RATE_PER9 = {"ERA", "FIP", "xFIP", "K/9", "BB/9", "HR/9", "H/9", "WHIP", "RA9"}


def suggest(col: str, s: pd.Series, stats: str) -> str:
    denom = DENOM[stats].get(col)
    lo, hi = s.min(), s.max()
    zero = (s == 0).mean()
    tag = ""
    if zero > 0.3:
        tag = "；零比例高，考慮 zero-inflated"
    if denom:
        return f"binomial（分母 {denom}）" + tag
    if col in COUNT_LIKE:
        vm = s.var() / s.mean() if s.mean() > 0 else np.nan
        return ("negbin" if vm > 1.5 else "poisson") + "，offset=log(PA 或 IP)" + tag
    if col in RATE_PER9:
        return "改回計數（分子 ~ poisson，offset=IP）；非得用則 gamma/lognormal"
    if 0 <= lo and hi <= 1:
        return "beta 或 logit→normal" + tag
    if lo >= 0 and s.skew() > 1:
        return "lognormal 或取 log→normal" + tag
    if col in {"wRC+", "Age"}:
        return "normal（層級×球季內標準化）"
    return "normal（層級×球季內標準化）" + tag


def profile(df: pd.DataFrame, stats: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in ID_COLS or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        rows.append(
            {
                "欄位": col,
                "n": len(valid),
                "缺值%": round(100 * s.isna().mean(), 1),
                "min": round(valid.min(), 3),
                "中位": round(valid.median(), 3),
                "max": round(valid.max(), 3),
                "零%": round(100 * (valid == 0).mean(), 1),
                "偏態": round(valid.skew(), 2),
                "var/mean": round(valid.var() / valid.mean(), 2) if valid.mean() > 0 else np.nan,
                "建議似然": suggest(col, valid, stats),
            }
        )
    return pd.DataFrame(rows)


def level_means(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return (
        df.groupby("level")[cols]
        .mean()
        .round(3)
        .reindex(["DSL", "R", "CPX", "A-", "A", "A+", "AA", "AAA"])
        .dropna(how="all")
    )


def main(data_dir: str, out_dir: str):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# 小聯盟變數字典\n"]
    keys = {
        "bat": ["Age", "PA", "AVG", "OBP", "SLG", "ISO", "BB%", "K%", "wOBA", "wRC+", "SwStr%", "GB%"],
        "pit": ["Age", "IP", "ERA", "FIP", "K%", "BB%", "K/9", "BB/9", "HR/9", "GB%", "SwStr%"],
    }
    for stats in ("bat", "pit"):
        fn = data / f"minors_{stats}.parquet"
        if not fn.exists():
            print(f"略過 {fn}（不存在）")
            continue
        df = pd.read_parquet(fn)
        vd = profile(df, stats)
        vd.to_csv(out / f"vardict_{stats}.csv", index=False)
        lm = level_means(df, keys[stats])
        lm.to_csv(out / f"level_means_{stats}.csv")
        md += [
            f"## {stats}：{len(df):,} 列，{df['key_mlbam'].nunique():,} 人，{df['season'].min()}–{df['season'].max()}\n",
            "### 各層級平均\n",
            lm.to_markdown(),
            "\n### 變數字典\n",
            vd.to_markdown(index=False),
            "\n",
        ]
        print(f"{stats}: {len(vd)} 個欄位 -> {out}")
    (out / "profile.md").write_text("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data", sys.argv[2] if len(sys.argv) > 2 else "reports")
