"""直接從 Chadwick register 的 GitHub 抓完整球員 id 對照表（pybaseball 的版本會丟掉 key_bbref_minors）。

    python src/fetch_ids.py data/
輸出 data/ids.parquet：key_mlbam, key_bbref, key_bbref_minors, key_fangraphs, 姓名, 出生年, 職業/大聯盟首末年
"""

import io
import sys
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{}.csv"
KEEP = [
    "key_mlbam",
    "key_bbref",
    "key_bbref_minors",
    "key_fangraphs",
    "name_first",
    "name_last",
    "birth_year",
    "pro_played_first",
    "pro_played_last",
    "mlb_played_first",
    "mlb_played_last",
]


def main(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for h in "0123456789abcdef":
        r = requests.get(BASE.format(h), timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), usecols=KEEP, low_memory=False)
        frames.append(df)
        print(f"people-{h}: {len(df):,}", flush=True)
    ids = pd.concat(frames, ignore_index=True)
    ids = ids[ids["key_mlbam"].notna() | ids["key_bbref_minors"].notna()]
    for c in (
        "key_mlbam",
        "key_fangraphs",
        "birth_year",
        "pro_played_first",
        "pro_played_last",
        "mlb_played_first",
        "mlb_played_last",
    ):
        ids[c] = pd.to_numeric(ids[c], errors="coerce").astype("Int64")
    ids.to_parquet(out / "ids.parquet", index=False)
    print(
        f"{len(ids):,} 人 -> {out / 'ids.parquet'}（有 key_mlbam {ids['key_mlbam'].notna().sum():,}，有 key_bbref_minors {ids['key_bbref_minors'].notna().sum():,}）"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
