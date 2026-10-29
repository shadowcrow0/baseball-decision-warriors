"""大聯盟每人每季的主守位（FanGraphs 的 position / positionDB），給「缺什麼補什麼」檢驗用。

    python src/fetch_fg_positions.py 2006 2025 data/

fg_batting.parquet 若是舊端點抓的沒有守位欄，這支用 FanGraphs 的 JSON API（同 fetch_war.fg_api）補一份：
    data/fg_positions.parquet   key_mlbam, Season, Team, position, positionDB, Age, PA, WAR
有 cache/fg/bat_<year>_q0.json 就直接讀，不重抓。
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_war import fg_api

KEEP = ["key_mlbam", "Season", "Team", "position", "positionDB", "Age", "PA", "WAR"]


def main(start: int, end: int, out_dir: str):
    rows = []
    for y in range(start, end + 1):
        d = fg_api("bat", y, qual=0)
        miss = [c for c in KEEP if c not in d.columns]
        if miss:
            print(f"{y}: 缺欄位 {miss}，有的欄位：{sorted(d.columns)[:40]}")
            continue
        d = d[KEEP].copy()
        d["key_mlbam"] = pd.to_numeric(d["key_mlbam"], errors="coerce")
        rows.append(d)
        print(f"{y}: {len(d):,} 人季", flush=True)
    out = pd.concat(rows, ignore_index=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(out_dir) / "fg_positions.parquet", index=False)
    print(out["position"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
