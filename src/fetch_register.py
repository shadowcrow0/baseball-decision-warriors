"""抓 Baseball-Reference 小聯盟球員頁（register），取得體系外軌跡：獨立聯盟、日韓台、墨西哥、冬季聯盟。

    python src/fetch_register.py data/ reports/ [--limit N]

對象：路徑表裡 next_status == gone_return 的人（離開體系後又回來）。
每人一次請求，B-Ref 限速約 6 秒一次；HTML 快取在 cache/register/，中斷重跑接續。
先用 --limit 5 測試，確認輸出正常再全跑。

輸出：
    data/register_paths.parquet   一列 = 一人 × 一年 × 一隊：year, age, team, league, lev, aff, role, g, pa/ip
    reports/register_summary.md   Lev 與 Lg 的分布
"""

import io
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pybaseball.datasources.bref import BRefSession  # noqa: E402

URL = "https://www.baseball-reference.com/register/player.fcgi?id={}"
CACHE = Path(__file__).resolve().parent.parent / "cache" / "register"
session = BRefSession()


def get_html(bid: str) -> str:
    fn = CACHE / f"{bid}.html"
    if fn.exists():
        return fn.read_text()
    for attempt in range(5):
        r = session.get(URL.format(bid), timeout=None)
        if r.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  429 限流，等 {wait} 秒", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 403 or "Just a moment" in r.text[:2000]:
            raise RuntimeError("Cloudflare 擋頁，中止")
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        fn.parent.mkdir(parents=True, exist_ok=True)
        fn.write_text(r.text)
        return r.text
    raise ValueError("連續 429")


def parse(html: str) -> pd.DataFrame:
    if not html:
        return pd.DataFrame()
    html = html.replace("<!--", "").replace("-->", "")  # B-Ref 把表格藏在註解裡
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame()
    rows = []
    for t in tables:
        cols = set(map(str, t.columns))
        if not {"Year", "Tm", "Lg", "Lev"} <= cols:
            continue
        role = "bat" if "PA" in cols else ("pit" if "IP" in cols else "other")
        t = t[t["Year"].astype(str).str.fullmatch(r"\d{4}")].copy()
        t["year"] = t["Year"].astype(int)
        t["role"] = role
        t["g"] = pd.to_numeric(t.get("G"), errors="coerce")
        t["expo"] = pd.to_numeric(t.get("PA") if role == "bat" else t.get("IP"), errors="coerce")
        t = t.rename(columns={"Age": "age", "Tm": "team", "Lg": "league", "Lev": "lev", "Aff": "aff"})
        t = t.loc[:, ~t.columns.duplicated()]
        for c in ("age", "team", "league", "lev", "aff"):
            if c not in t.columns:
                t[c] = None
        t["lev"] = t["lev"].astype(str).where(t["lev"].notna(), None)
        t["is_total"] = t["team"].astype(str).str.contains("Teams", na=False)  # 「2 Teams」合計列
        rows.append(t[["year", "age", "team", "league", "lev", "aff", "role", "g", "expo", "is_total"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(data_dir: str, out_dir: str, limit: int | None):
    data, out = Path(data_dir), Path(out_dir)
    ps = pd.read_parquet(data / "player_season.parquet")
    ids = pd.read_parquet(data / "ids.parquet")[["key_mlbam", "key_bbref_minors"]].dropna()
    targets = ps.loc[ps["next_status"] == "gone_return", "key_mlbam"].drop_duplicates().to_frame()
    targets = targets.merge(ids, on="key_mlbam", how="left")
    print(f"gone_return {len(targets):,} 人，有 B-Ref id {targets['key_bbref_minors'].notna().sum():,}", flush=True)
    targets = targets.dropna(subset=["key_bbref_minors"])
    if limit:
        targets = targets.head(limit)

    frames, failed = [], []
    for i, (mlbam, bid) in enumerate(zip(targets["key_mlbam"], targets["key_bbref_minors"]), 1):
        try:
            df = parse(get_html(bid))
        except RuntimeError as e:
            print(f"\n{e}。已抓 {i - 1} 人，先存檔。", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            failed.append((bid, f"{type(e).__name__}: {str(e)[:60]}"))
            continue
        if df.empty:
            failed.append((bid, "no table"))
            continue
        df.insert(0, "key_bbref_minors", bid)
        df.insert(0, "key_mlbam", mlbam)
        frames.append(df)
        if i % 25 == 0 or limit:
            levs = sorted(set(map(str, df.loc[~df["is_total"], "lev"].dropna())))
            print(f"  {i}/{len(targets)}  最新：{bid} {len(df)} 列，Lev={levs}", flush=True)

    if not frames:
        print("沒有任何結果；失敗：", failed[:5])
        return
    reg = pd.concat(frames, ignore_index=True)
    # 統一型態：B-Ref 表格裡 age 偶爾是字串、team/league 可能是數字
    reg["age"] = pd.to_numeric(reg["age"], errors="coerce")
    for c in ("team", "league", "lev", "aff", "key_bbref_minors"):
        reg[c] = reg[c].astype("string")
    for c in ("g", "expo"):
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    reg["year"] = reg["year"].astype(int)
    reg["key_mlbam"] = pd.to_numeric(reg["key_mlbam"], errors="coerce").astype("Int64")
    reg.to_parquet(data / "register_paths.parquet", index=False)
    one = reg[~reg["is_total"]]
    md = [
        f"# B-Ref 球員頁軌跡\n",
        f"{reg['key_mlbam'].nunique():,} 人，{len(reg):,} 列（合計列 {int(reg['is_total'].sum()):,}）；失敗 {len(failed)}\n",
        "### Lev 分布（排除合計列）\n",
        one["lev"].astype(str).value_counts().to_frame("列數").to_markdown(),
        "\n### 非所屬體系的 Lg 分布（Lev 不在 AAA/AA/A+/A/A-/Rk）\n",
        one[~one["lev"].astype(str).isin(["AAA", "AA", "A+", "A", "A-", "Rk"])]["league"]
        .astype(str)
        .value_counts()
        .head(40)
        .to_frame("列數")
        .to_markdown(),
    ]
    if failed:
        md.append("\n### 失敗\n" + "\n".join(f"- {b}: {m}" for b, m in failed[:50]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "register_summary.md").write_text("\n".join(md))
    print("\n".join(md[:4]))
    print(f"-> {data / 'register_paths.parquet'}, {out / 'register_summary.md'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lim = None
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
        args = [a for a in args if a != str(lim)]
    main(args[0] if args else "data", args[1] if len(args) > 1 else "reports", lim)
