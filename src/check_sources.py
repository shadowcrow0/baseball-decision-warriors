"""先跑這支：確認 pybaseball 各資料來源從這台機器抓不抓得到。

    python src/check_sources.py

雲端 IP 通常被 Baseball-Reference（Cloudflare）與 FanGraphs（403）擋；
校園網路大多可以。全部 OK 才跑後面的 pipeline。
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pybaseball as pb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_draft import amateur_draft  # noqa: E402  修過 pandas 3 bug 的版本
from fetch_war import fg_api  # noqa: E402  FanGraphs 新 JSON API

TESTS = {
    "Baseball-Reference 選秀 amateur_draft(2018, 1)": lambda: amateur_draft(2018, 1),
    "FanGraphs 舊端點 batting_stats(2024)": lambda: pb.batting_stats(2024, qual=100),
    "FanGraphs 新 API fg_api('bat', 2024)": lambda: fg_api("bat", 2024, qual=100),
    "FanGraphs 新 API fg_api('pit', 2024)": lambda: fg_api("pit", 2024, qual=50),
    "Chadwick 球員 id 對照 chadwick_register()": lambda: pb.chadwick_register(),
    "Baseball Savant 逐球 statcast(1 天)": lambda: pb.statcast("2025-04-01", "2025-04-01"),
}

if __name__ == "__main__":
    ok = 0
    for name, fn in TESTS.items():
        try:
            df = fn()
            print(f"OK    {name}: {df.shape[0]:,} 列 × {df.shape[1]} 欄")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:100]}")
    print(f"\n{ok}/{len(TESTS)} 來源可用")
