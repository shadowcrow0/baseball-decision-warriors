"""Baseball Savant 排行榜（大聯盟層級的物理量），用來驗證四種能力：
    衝刺速度 ↔ 跑得快；擊球初速、barrel% ↔ 打得遠；揮空率百分位 ↔ 打得到球；追打壞球百分位 ↔ 選得到球（後兩者在 percentile 排行榜）。

    python src/fetch_savant_leaderboards.py data/ [起始年] [結束年]

每年每個排行榜一個 CSV（快取在 data/savant/），合併成 data/savant_leaderboards.parquet（一列 = 一人一年一排行榜）。
守備排行榜（oaa、arm、framing、poptime、blocking）：大聯盟層級，只有登上的人才有；用來驗證「球探看到的是守備」，不能當小聯盟的預測變數。
端點是從網站網址猜的，打不到的會印「失敗」然後跳過，不會中斷。
鍵 player_id = MLBAM id = 我們的 key_mlbam。端點與 pybaseball 相同，自己用 requests 打，避開 pybaseball 在 pandas 3 的問題。
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BOARDS = {
    # name: (url template, 最低門檻)
    "sprint_speed": (
        "https://baseballsavant.mlb.com/leaderboard/sprint_speed?year={y}&position=&team=&min={m}&csv=true",
        5,
    ),
    "exitvelo": (
        "https://baseballsavant.mlb.com/leaderboard/statcast?type=batter&year={y}&position=&team=&min={m}&csv=true",
        25,
    ),
    "expected": (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=batter&year={y}&position=&team=&min={m}&csv=true",
        25,
    ),
    "percentile": (
        "https://baseballsavant.mlb.com/leaderboard/percentile-rankings?type=batter&year={y}&position=&team=&csv=true",
        None,
    ),
    # 守備（大聯盟層級，只有登上的人才有；用來驗證「球探看到的是守備」與拆解守備價值）
    "oaa": (
        "https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielder&year={y}&team=&range=year&min={m}&pos=&roles=&viz=hide&csv=true",
        1,
    ),
    "arm": (
        "https://baseballsavant.mlb.com/leaderboard/arm-strength?type=player&year={y}&minThrows={m}&pos=&team=&csv=true",
        25,
    ),
    "framing": (
        "https://baseballsavant.mlb.com/catcher_framing?year={y}&team=&min={m}&type=catcher&sort=4,1&csv=true",
        1,
    ),
    "poptime": ("https://baseballsavant.mlb.com/leaderboard/poptime?year={y}&team=&min2b={m}&min3b=0&csv=true", 1),
    "blocking": ("https://baseballsavant.mlb.com/leaderboard/catcher_blocking?year={y}&team=&min={m}&csv=true", 1),
}
FIRST_YEAR = {
    "oaa": 2016,
    "arm": 2020,
    "framing": 2015,
    "poptime": 2015,
    "blocking": 2018,
}  # 排行榜開始的年份，之前不打
UA = {"User-Agent": "Mozilla/5.0 (research; contact via GitHub shadowcrow0/baseball)"}


def fetch(name, y, cache: Path):
    url_t, m = BOARDS[name]
    f = cache / f"{name}_{y}.csv"
    if f.exists() and f.stat().st_size > 200:
        return pd.read_csv(f)
    url = url_t.format(y=y, m=m)
    for attempt in range(4):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200 and len(r.text) > 200 and not r.text.lstrip().startswith("<"):
            f.write_text(r.text)
            return pd.read_csv(io.StringIO(r.text))
        print(f"  {name} {y}: HTTP {r.status_code}, {len(r.text)} bytes, 重試", flush=True)
        time.sleep(5 * (attempt + 1))
    print(f"  {name} {y}: 失敗", flush=True)
    return None


def main(data_dir, y0=2015, y1=2025):
    data = Path(data_dir)
    cache = data / "savant"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in range(y0, y1 + 1):
        for name in BOARDS:
            if y < FIRST_YEAR.get(name, 2015):
                continue
            df = fetch(name, y, cache)
            time.sleep(2)
            if df is None or len(df) == 0:
                continue
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            idcol = next(
                (
                    c
                    for c in df.columns
                    if c in ("player_id", "id", "mlbam_id", "playerid", "entity_id", "fielder_id", "catcher_id")
                ),
                None,
            )
            if idcol is None:
                print(f"  {name} {y}: 找不到 id 欄，欄位 = {list(df.columns)[:12]}", flush=True)
                continue
            df = df.rename(columns={idcol: "key_mlbam"})
            df["key_mlbam"] = pd.to_numeric(df["key_mlbam"], errors="coerce")
            df = df.dropna(subset=["key_mlbam"])
            df["key_mlbam"] = df["key_mlbam"].astype(int)
            df["season"] = y
            df["board"] = name
            frames.append(df)
            print(f"  {name} {y}: {len(df)} 人，欄位 {len(df.columns)}", flush=True)
    out = pd.concat(frames, ignore_index=True, sort=False)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str)
    out.to_parquet(data / "savant_leaderboards.parquet", index=False)
    print(f"-> {data / 'savant_leaderboards.parquet'}：{len(out):,} 列；排行榜 {out.board.value_counts().to_dict()}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "data", int(a[1]) if len(a) > 1 else 2015, int(a[2]) if len(a) > 2 else 2025)
