"""每隊每年的大聯盟薪資總額（「拮据」的切法）。試兩個來源，抓不到的印「失敗」跳過。

    python src/fetch_payroll.py data/ [起始年] [結束年]

來源一：Spotrac 年度薪資頁（每年一張表，含 30 隊的 payroll）。
來源二：Baseball-Reference 聯盟年度頁的 payroll 表（備援）。
輸出 data/payroll.parquet：Team（B-Ref 三碼縮寫，對得上 model_bat 的 org）、Season、payroll（美元）、source。
快取在 data/payroll/，中斷可接續。守規矩：每個請求之間停 3 秒。
"""

import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research; contact via GitHub shadowcrow0/baseball)"}
SOURCES = {
    "spotrac": "https://www.spotrac.com/mlb/payroll/_/year/{y}",
    "bref": "https://www.baseball-reference.com/leagues/majors/{y}-payroll.shtml",
}
# 球隊全名 → B-Ref 三碼（model_bat 的 org 用的就是這套）
NAME_TO_ORG = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Indians": "CLE",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Florida Marlins": "FLA",
    "Miami Marlins": "MIA",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Angels of Anaheim": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Tampa Bay Devil Rays": "TBD",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
    "Montreal Expos": "MON",
}


def get(url: str, cache: Path) -> str | None:
    if cache.exists() and cache.stat().st_size > 1000:
        return cache.read_text()
    for attempt in range(3):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200 and len(r.text) > 1000:
            cache.write_text(r.text)
            return r.text
        print(f"  {url}: HTTP {r.status_code}, 重試", flush=True)
        time.sleep(5 * (attempt + 1))
    return None


def money(s) -> float:
    s = re.sub(r"[^\d.]", "", str(s))
    return float(s) if s else float("nan")


def parse_tables(html: str, year: int, source: str) -> pd.DataFrame | None:
    """找含球隊名與 payroll 欄的表。"""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as e:
        print(f"  {source} {year}: 解析失敗 {e}", flush=True)
        return None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        team_col = next((c for c, lc in zip(t.columns, cols) if "team" in lc), None)
        pay_col = next((c for c, lc in zip(t.columns, cols) if "payroll" in lc or "total" in lc or "est" in lc), None)
        if team_col is None or pay_col is None:
            continue
        rows = []
        for _, r in t.iterrows():
            name = str(r[team_col]).strip()
            name = re.sub(r"\s+", " ", name)
            org = next((v for k, v in NAME_TO_ORG.items() if k in name), None)
            if org is None:
                continue
            rows.append({"Team": org, "Season": year, "payroll": money(r[pay_col]), "source": source})
        if len(rows) >= 25:
            return pd.DataFrame(rows).dropna(subset=["payroll"])
    return None


def main(data_dir, y0=2006, y1=2025):
    data = Path(data_dir)
    cache = data / "payroll"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in range(y0, y1 + 1):
        got = None
        for source, url_t in SOURCES.items():
            html = get(url_t.format(y=y), cache / f"{source}_{y}.html")
            time.sleep(3)
            if html is None:
                print(f"  {source} {y}: 失敗", flush=True)
                continue
            got = parse_tables(html, y, source)
            if got is not None:
                print(f"  {source} {y}: {len(got)} 隊", flush=True)
                break
            print(f"  {source} {y}: 沒找到薪資表", flush=True)
        if got is not None:
            frames.append(got)
    if not frames:
        print("全部失敗")
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(data / "payroll.parquet", index=False)
    print(f"-> {data / 'payroll.parquet'}：{len(out)} 列，{out.Season.nunique()} 年")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "data", int(a[1]) if len(a) > 1 else 2006, int(a[2]) if len(a) > 2 else 2025)
