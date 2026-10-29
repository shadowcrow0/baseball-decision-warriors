"""抓 Baseball Savant 的小聯盟 Statcast 逐球資料（打者視角），按天下載、可中斷續抓。

    python src/fetch_savant_minors.py data/ 2024                 # 3A 與 1A 都抓（有追蹤的層級）
    python src/fetch_savant_minors.py data/ 2023 --level AAA     # 只抓 3A

Savant 的涵蓋範圍（頁面自述）：3A 全部比賽從 2023 起；2022 只有太平洋岸聯盟與夏洛特主場；佛州州聯盟（1A）從 2021 起。
端點 https://baseballsavant.mlb.com/statcast-search-minors/csv，要帶 minors=true 與 hfLevel=AAA| 或 A|，不帶層級會變成大聯盟。
每天每層級一個 parquet 快取在 data/savant_minors/<season>/<level>_<date>.parquet（只留下面那些欄），最後合併成 data/savant_minors_<season>.parquet，
只留打擊相關的欄（結果、打球類型、打到哪個守備員、初速、仰角、落點、barrel 分類、描述文字）加上每一球的進壘點（plate_x、plate_z、zone）、球種、球速，
好算打者的「洞」（哪一格揮空最多）和投手攻那格的比例。舊快取少了這些欄會自動重抓那天。
一天約 5,000 球、一季約 150 天，一季一層級約 75 萬列；請在 Arc 的 compute 節點跑（scripts/arc_run.sh）。
鍵 batter = MLBAM id = 我們的 key_mlbam。
"""

import argparse
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL = "https://baseballsavant.mlb.com/statcast-search-minors/csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}
KEEP = [
    "game_date",
    "game_pk",
    "game_year",
    "home_team",
    "away_team",
    "batter",
    "pitcher",
    "player_name",
    "stand",
    "p_throws",
    "inning",
    "outs_when_up",
    "on_1b",
    "on_2b",
    "on_3b",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "type",
    "pitch_type",
    "release_speed",
    "plate_x",
    "plate_z",
    "zone",
    "sz_top",
    "sz_bot",
    "events",
    "description",
    "des",
    "bb_type",
    "hit_location",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    "launch_speed_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "bat_speed",
    "swing_length",
    "age_bat",
    "delta_run_exp",
]
ROW_CAP = 25000  # Savant 單次查詢的上限；一天超過這個數字就要再切半天


def fetch_day(season: int, level: str, day: date) -> pd.DataFrame:
    """一天、一個層級的逐球；429 或 5xx 退避重試，最多八次。"""
    params = dict(
        all="true",
        minors="true",
        hfLevel=f"{level}|",
        hfGT="R|",
        hfSea=f"{season}|",
        player_type="batter",
        game_date_gt=day.isoformat(),
        game_date_lt=day.isoformat(),
        group_by="name",
        min_pitches="0",
        min_results="0",
        min_pas="0",
        sort_col="pitches",
        player_event_sort="api_p_release_speed",
        sort_order="desc",
        type="details",
    )
    for attempt in range(8):
        r = requests.get(URL, params=params, timeout=180, headers=HEADERS)
        if r.ok:
            if len(r.content) < 50:
                return pd.DataFrame()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            if len(df) >= ROW_CAP:
                print(f"  {day} {level} 到達 {ROW_CAP} 列上限，資料可能被截斷", flush=True)
            return df
        wait = min(30 * 2**attempt, 600)
        print(f"  {day} {level} HTTP {r.status_code}，等 {wait} 秒重試", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{day} {level} 連續失敗")


def main(data_dir: str, season: int, levels: list, start: str, end: str) -> None:
    data = Path(data_dir)
    cache = data / "savant_minors" / str(season)
    cache.mkdir(parents=True, exist_ok=True)
    first = date.fromisoformat(f"{season}-{start}")
    last = date.fromisoformat(f"{season}-{end}")
    frames = []
    day = first
    while day <= last:
        for level in levels:
            fn = cache / f"{level}_{day.isoformat()}.parquet"
            cached = pd.read_parquet(fn) if fn.exists() else None
            if cached is not None and (len(cached) == 0 or "plate_x" in cached.columns):
                df = cached  # 空檔表示那天沒比賽；有資料的要有進壘點欄才算抓齊
            else:
                df = fetch_day(season, level, day)
                df = df[[c for c in KEEP if c in df.columns]].copy()  # 只留打擊相關欄，一天約 1 MB
                df.to_parquet(fn, index=False)  # 空的也存，表示這天抓過了
                time.sleep(0.5)
            if len(df):
                df["level"] = level
                frames.append(df)
        if day.day in (1, 15):
            n = sum(len(f) for f in frames)
            print(f"{day}：累計 {n:,} 球", flush=True)
        day += timedelta(days=1)
    if not frames:
        print("沒有抓到任何資料", flush=True)
        return
    out = pd.concat(frames, ignore_index=True)
    path = data / f"savant_minors_{season}.parquet"
    out.to_parquet(path, index=False)
    pa = out["events"].notna().sum()
    print(
        f"{season}：{len(out):,} 球、{pa:,} 個打席結果、{out['batter'].nunique():,} 位打者；層級 {sorted(out['level'].unique())}",
        flush=True,
    )
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="抓 Savant 小聯盟 Statcast 逐球（3A 2023 起、1A 佛州州聯盟 2021 起）")
    p.add_argument("data_dir")
    p.add_argument("season", type=int)
    p.add_argument("--level", default="all", choices=["AAA", "A", "all"], help="層級；all = AAA 與 A 都抓")
    p.add_argument("--start", default="03-25", help="起日 MM-DD")
    p.add_argument("--end", default="10-05", help="迄日 MM-DD")
    a = p.parse_args()
    main(a.data_dir, a.season, ["AAA", "A"] if a.level == "all" else [a.level], a.start, a.end)
