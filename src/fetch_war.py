"""用 pybaseball 抓 FanGraphs 逐季數據：fWAR、wOBA、FIP 等，與 Chadwick 球員 id 對照表。

    python src/fetch_war.py 2005 2025 data/

輸出：
    data/fg_batting.parquet   一列一人一季（qual=0 含所有有出賽的人）
    data/fg_pitching.parquet
    data/ids.parquet          Chadwick：key_mlbam / key_fangraphs / key_bbref / 姓名 / 首末出賽年
"""

import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
import pybaseball as pb  # noqa: E402

pb.cache.enable()
CACHE = Path(__file__).resolve().parent.parent / "cache" / "fg"
FG_API = "https://www.fangraphs.com/api/leaders/major-league/data"
# 注意：不要加瀏覽器 User-Agent，加了反而被 Cloudflare 挑戰；requests 預設 UA 可過（實測 2026-10）


def fg_api(stats: str, season: int, qual: int = 0) -> pd.DataFrame:
    """FanGraphs 新的 JSON 排行榜 API（網站前端用的那個），舊的 leaders-legacy.aspx 已 403。
    stats = 'bat' 或 'pit'。回傳欄位對齊 pybaseball：IDfg, Season, Name, Team, Age, PA/IP, wOBA/FIP, WAR ...
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = CACHE / f"{stats}_{season}_q{qual}.json"
    if fn.exists():
        data = json.loads(fn.read_text())
    else:
        params = {
            "age": "",
            "pos": "all",
            "stats": stats,
            "lg": "all",
            "qual": qual,
            "season": season,
            "season1": season,
            "startdate": "",
            "enddate": "",
            "month": 0,
            "hand": "",
            "team": 0,
            "pageitems": 5000,
            "pagenum": 1,
            "ind": 0,
            "rost": 0,
            "players": "",
            "type": 8,
            "postseason": "",
            "sortdir": "default",
            "sortstat": "WAR",
        }
        r = requests.get(FG_API, params=params, timeout=60)
        if r.status_code != 200 or "Just a moment" in r.text[:2000]:
            raise RuntimeError(f"FanGraphs API HTTP {r.status_code}（Cloudflare 擋頁）")
        data = r.json()
        fn.write_text(json.dumps(data))
    rows = data["data"] if isinstance(data, dict) else data
    df = pd.DataFrame(rows)
    # 對齊 pybaseball 欄位：IDfg、Name（API 的 Name 欄是 HTML 連結，改用 PlayerName）
    df = df.drop(columns=[c for c in ("Name", "Team") if c in df.columns])
    df = df.rename(columns={"playerid": "IDfg", "PlayerName": "Name", "TeamName": "Team", "xMLBAMID": "key_mlbam"})
    if "Season" not in df.columns:
        df["Season"] = season
    return df


def fetch_season(stats: str, year: int) -> pd.DataFrame:
    try:
        return pb.batting_stats(year, qual=0) if stats == "bat" else pb.pitching_stats(year, qual=0)
    except Exception as e:  # noqa: BLE001
        print(f"  舊端點失敗（{type(e).__name__}），改用新 API", flush=True)
        return fg_api(stats, year, qual=0)


def main(start: int, end: int, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bat, pit = [], []
    for y in range(start, end + 1):
        b = fetch_season("bat", y)
        p = fetch_season("pit", y)
        bat.append(b)
        pit.append(p)
        print(f"{y}: 打者 {len(b):,}，投手 {len(p):,}", flush=True)
    pd.concat(bat, ignore_index=True).to_parquet(out / "fg_batting.parquet", index=False)
    pd.concat(pit, ignore_index=True).to_parquet(out / "fg_pitching.parquet", index=False)

    ids = pb.chadwick_register()
    ids = ids[
        ["name_last", "name_first", "key_mlbam", "key_fangraphs", "key_bbref", "mlb_played_first", "mlb_played_last"]
    ].dropna(subset=["key_mlbam"])
    ids.to_parquet(out / "ids.parquet", index=False)
    print(f"id 對照 {len(ids):,} 人 -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
