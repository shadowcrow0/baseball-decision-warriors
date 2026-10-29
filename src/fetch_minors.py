"""抓 FanGraphs 小聯盟逐季逐層級個人數據（打者 + 投手）。

    python src/fetch_minors.py 2006 2026 data/

輸出：
    data/minors_bat.parquet   一列 = 一人 × 一季 × 一層級
    data/minors_pit.parquet
欄位含 key_mlbam（MLBAM id，對 Chadwick / 選秀表 / Statcast）、level（代碼）、level_rank（跨年一致的順序）。
每次請求的 JSON 快取在 cache/minors/，中斷重跑不重抓。
資料從 2006 年起；2005 以前端點無資料。
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.fangraphs.com/api/leaders/minor-league/data"
LG = "2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32,33"
CACHE = Path(__file__).resolve().parent.parent / "cache" / "minors"
# level 參數 -> 名稱；5/6 只到 2020，7/8 從 2021 起（2021 改制：短期 A 與新人聯盟併為複合聯盟）
LEVELS = {1: "AAA", 2: "AA", 3: "A+", 4: "A", 5: "A-", 6: "R", 7: "CPX", 8: "DSL"}
# 跨年一致的層級順序，越大越接近大聯盟
LEVEL_RANK = {"DSL": 0, "R": 1, "CPX": 1, "A-": 2, "A": 3, "A+": 4, "AA": 5, "AAA": 6}
# 注意：不要加瀏覽器 User-Agent，requests 預設即可通過
PAUSE = 4.0  # 每次請求間隔秒數；FanGraphs 約 40 次連續請求後會回 429，同時只能跑一個抓取程式


def fetch(stats: str, season: int, level: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = CACHE / f"{stats}_{season}_L{level}.json"
    if fn.exists():
        rows = json.loads(fn.read_text())
    else:
        params = dict(
            pos="all",
            level=str(level),
            lg=LG,
            stats=stats,
            qual="0",
            type="0",
            team="",
            season=season,
            seasonEnd=season,
            org="",
            ind="0",
            splitTeam="false",
        )
        r = None
        for attempt in range(8):
            r = requests.get(URL, params=params, timeout=120)
            if r.ok:
                break
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0)) or min(30 * 2**attempt, 600)
                print(f"  429 流量限制（{stats} {season} L{level}），等 {wait} 秒", flush=True)
            else:
                wait = 10 * (attempt + 1)
                print(f"  HTTP {r.status_code}（{stats} {season} L{level}），等 {wait} 秒", flush=True)
            time.sleep(wait)
        r.raise_for_status()
        j = r.json()
        rows = j if isinstance(j, list) else j.get("data", [])
        fn.write_text(json.dumps(rows))
        time.sleep(PAUSE)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop(columns=[c for c in ("Name", "Team") if c in df.columns])
    df = df.rename(
        columns={
            "PlayerName": "name",
            "TeamName": "team",
            "xMLBAMID": "key_mlbam",
            "Season": "season",
            "AffAbbName": "org",
        }
    )
    df["level"] = LEVELS[level]
    df["level_rank"] = LEVEL_RANK[LEVELS[level]]
    return df


def main(start: int, end: int, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stats in ("bat", "pit"):
        frames = []
        for season in range(start, end + 1):
            n = 0
            for level in LEVELS:
                df = fetch(stats, season, level)
                if not df.empty:
                    frames.append(df)
                    n += len(df)
            print(f"{stats} {season}: {n:,} 列", flush=True)
        full = pd.concat(frames, ignore_index=True)
        full["key_mlbam"] = pd.to_numeric(full["key_mlbam"], errors="coerce").astype("Int64")
        full.to_parquet(out / f"minors_{stats}.parquet", index=False)
        print(
            f"{stats}: {len(full):,} 列 × {full.shape[1]} 欄, {full['key_mlbam'].nunique():,} 人 -> {out / f'minors_{stats}.parquet'}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
