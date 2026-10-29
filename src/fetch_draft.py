"""用 pybaseball.amateur_draft 抓 Baseball-Reference 選秀名單（含生涯 WAR、簽約金）。

    python src/fetch_draft.py 2005 2025 data/draft.parquet

一列一個 pick。欄位來自 B-Ref：OvPck 總順位、Tm 球隊、Signed、Bonus 簽約金、
Name、Pos、WAR（B-Ref 生涯 bWAR，截至抓取日）、G/AB/HR/BA/OPS（打擊）、
G.1/W/L/ERA/WHIP/SV（投球）、Type（HS/4Yr/JC）、Drafted Out of 學校。
每年每輪一次請求；B-Ref 限速，整段約 1–2 小時。HTML 快取在 cache/bref/，重跑不重抓。
注意：pybaseball 2.2.7 的 amateur_draft 在 pandas 3 下會壞（read_html 不吃 bytes），這裡自己修。
"""

import io
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from pybaseball.amateur_draft import _URL, postprocess  # noqa: E402
from pybaseball.datasources.bref import BRefSession  # noqa: E402

MAX_ROUNDS = 60  # 2005 年有 50 輪；遇到空輪就停
CACHE = Path(__file__).resolve().parent.parent / "cache" / "bref"
session = BRefSession()  # pybaseball 內建的 B-Ref 限速 session


def amateur_draft(year: int, rnd: int) -> pd.DataFrame:
    """同 pybaseball.amateur_draft，但修掉 pandas 3 的 read_html(bytes) 錯誤，並把 HTML 快取到磁碟。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    fn = CACHE / f"{year}_{rnd}.html"
    if fn.exists():
        html = fn.read_text()
    else:
        for attempt in range(6):
            r = session.get(_URL.format(year=year, draft_round=rnd), timeout=None)
            html = r.text
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  HTTP 429 限流（{year} 第 {rnd} 輪），等 {wait} 秒", flush=True)
                time.sleep(wait)
                continue
            if "Just a moment" in html[:2000] or r.status_code == 403:
                raise RuntimeError(f"Cloudflare 擋頁 HTTP {r.status_code}（此 IP 被 Baseball-Reference 封鎖）")
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")
            break
        else:
            raise ValueError("連續 429，放棄這一輪")
        if "<table" in html:
            fn.write_text(html)  # 只快取有表格的頁；沒表格的（該輪不存在或異常頁）不存，重跑會再確認
    if "<table" not in html:
        return pd.DataFrame()  # 該輪不存在
    tables = pd.read_html(io.StringIO(html))
    return postprocess(pd.concat(tables))


class Blocked(RuntimeError):
    pass


def fetch_round(year: int, rnd: int) -> pd.DataFrame:
    """失敗就等 30/60/120/240/480 秒重試；Cloudflare 擋頁直接中止整個程式。"""
    for attempt in range(6):
        try:
            return amateur_draft(year, rnd)
        except RuntimeError as e:
            raise Blocked(str(e)) from e
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "No tables found" in msg or "list index" in msg or isinstance(e, ImportError):
                return pd.DataFrame()  # 該輪不存在
            if attempt == 5:
                raise
            wait = 30 * 2**attempt
            print(f"  {year} 第 {rnd} 輪失敗：{type(e).__name__}: {msg[:80]}，{wait} 秒後重試", flush=True)
            time.sleep(wait)


def fetch_year(year: int) -> pd.DataFrame:
    frames = []
    for rnd in range(1, MAX_ROUNDS + 1):
        df = fetch_round(year, rnd)
        if df.empty:
            break
        df = df.copy()
        df["year"] = year
        df["round"] = rnd
        frames.append(df)
        if rnd % 10 == 0:
            print(f"  {year} 第 {rnd} 輪，累計 {sum(len(f) for f in frames):,} picks", flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"{year}: {len(frames)} 輪, {len(out):,} picks", flush=True)
    return out


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "OvPck": "pick_overall",
        "Tm": "team",
        "Signed": "signed",
        "Bonus": "bonus",
        "Name": "name",
        "Pos": "position",
        "WAR": "bwar_career",
        "G": "g_bat",
        "AB": "ab",
        "HR": "hr",
        "BA": "ba",
        "OPS": "ops",
        "G.1": "g_pit",
        "W": "w",
        "L": "l",
        "ERA": "era",
        "WHIP": "whip",
        "SV": "sv",
        "Type": "level",
        "Drafted Out of": "school",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "bonus" in df.columns:
        df["bonus"] = pd.to_numeric(df["bonus"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    for c in ("pick_overall", "bwar_career", "g_bat", "g_pit"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    g_bat = df["g_bat"] if "g_bat" in df.columns else pd.Series(0, index=df.index)
    g_pit = df["g_pit"] if "g_pit" in df.columns else pd.Series(0, index=df.index)
    df["reached_mlb"] = (g_bat.fillna(0) + g_pit.fillna(0)) > 0
    return df


def main(start: int, end: int, out: str):
    frames = []
    for y in range(start, end + 1):
        try:
            frames.append(fetch_year(y))
        except Blocked as e:
            print(f"\n中止：{e}。已抓到的年份先存檔，稍後重跑會從快取接續。", flush=True)
            break
    df = pd.concat(frames, ignore_index=True)
    df = tidy(df)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    df.to_csv(outp.with_suffix(".csv"), index=False)
    print(f"{len(df):,} picks, {df['reached_mlb'].sum():,} 登上一軍 -> {outp}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
