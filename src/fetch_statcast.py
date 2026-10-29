"""
抓 Statcast 逐球資料，按週分段下載、存成 parquet，可中斷續抓。

用法（在 Arc compute 節點、已 activate pyb_env）：
    python fetch_statcast.py 2025-03-27 2025-09-28 data/statcast_2025
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import statcast, cache

cache.enable()  # 重抓同一段日期時直接用本機快取


def daterange_chunks(start: date, end: date, days: int = 7):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def main(start_str: str, end_str: str, out_dir: str):
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for s, e in daterange_chunks(start, end):
        fn = out / f"{s}_{e}.parquet"
        if fn.exists():
            print(f"skip {fn.name} (已存在)")
            continue
        print(f"fetch {s} ~ {e} ...", flush=True)
        df = statcast(start_dt=str(s), end_dt=str(e))
        if df is None or df.empty:
            print("  無資料")
            continue
        df.to_parquet(fn, index=False)
        print(f"  {len(df):,} 球 -> {fn.name}")

    # 合併成單一檔
    parts = sorted(out.glob("*.parquet"))
    if parts:
        full = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        merged = out.with_suffix(".parquet")
        full.to_parquet(merged, index=False)
        print(f"合併完成：{len(full):,} 球 -> {merged}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
