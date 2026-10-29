"""把 B-Ref 選秀表接上 MLBAM id，並做「選秀出身 vs 非選秀」的路徑對照。

    python src/link_draft.py data/ reports/

B-Ref 選秀頁沒有球員 id，用「標準化姓名 + 職業生涯首年落在選秀年到選秀年+1」對 Chadwick。
輸出：
    data/draft_linked.parquet    選秀表 + key_mlbam（配不到的為空）
    reports/draft_link.md        配對率、選秀出身 vs 非選秀的升級率與離開率
"""

import io
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

CACHE = Path(__import__("os").environ.get("BREF_CACHE", Path(__file__).resolve().parent.parent / "cache" / "bref"))

RANK_NAME = {0: "DSL", 1: "R/CPX", 2: "A-", 3: "A", 4: "A+", 5: "AA", 6: "AAA"}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.split())


def ids_from_html() -> pd.DataFrame:
    """從快取的選秀頁 HTML 取每個 pick 的 B-Ref 球員連結 → key_bbref（大聯盟頁）或 key_bbref_minors（register 頁）。"""
    rows = []
    for fn in sorted(CACHE.glob("*.html")):
        year, rnd = fn.stem.split("_")
        html = fn.read_text()
        if "<table" not in html:
            continue
        try:
            tables = pd.read_html(io.StringIO(html), extract_links="body")
        except ValueError:
            continue
        for t in tables:
            cols = [c[0] if isinstance(c, tuple) else c for c in t.columns]
            t.columns = cols
            if "Name" not in cols or "OvPck" not in cols:
                continue
            for _, r in t.iterrows():
                name, href = r["Name"] if isinstance(r["Name"], tuple) else (r["Name"], None)
                pick = r["OvPck"][0] if isinstance(r["OvPck"], tuple) else r["OvPck"]
                bbref = minors = None
                if href:
                    m = re.search(r"/players/\w/(\w+)\.shtml", href)
                    if m:
                        bbref = m.group(1)
                    m = re.search(r"player\.fcgi\?id=([\w.-]+)", href)
                    if m:
                        minors = m.group(1)
                rows.append(
                    {
                        "year": int(year),
                        "pick_overall": pd.to_numeric(pick, errors="coerce"),
                        "href_bbref": bbref,
                        "href_minors": minors,
                    }
                )
    return pd.DataFrame(rows).dropna(subset=["pick_overall"]).drop_duplicates(["year", "pick_overall"])


def main(data_dir: str, out_dir: str):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    draft = pd.read_parquet(data / "draft.parquet")
    ids_all = pd.read_parquet(data / "ids.parquet")
    ids = ids_all[ids_all["key_mlbam"].notna()].copy()

    # 第一優先：HTML 連結裡的 B-Ref id 直接對 Chadwick
    via_href = pd.DataFrame()
    if CACHE.exists():
        links = ids_from_html()
        if not links.empty:
            by_bbref = (
                ids[["key_bbref", "key_mlbam"]]
                .dropna()
                .drop_duplicates("key_bbref")
                .rename(columns={"key_bbref": "href_bbref", "key_mlbam": "mlbam_a"})
            )
            by_minors = (
                ids[["key_bbref_minors", "key_mlbam"]]
                .dropna()
                .drop_duplicates("key_bbref_minors")
                .rename(columns={"key_bbref_minors": "href_minors", "key_mlbam": "mlbam_b"})
            )
            links = links.merge(by_bbref, on="href_bbref", how="left").merge(by_minors, on="href_minors", how="left")
            links["key_mlbam_href"] = links["mlbam_a"].fillna(links["mlbam_b"])
            via_href = links[["year", "pick_overall", "key_mlbam_href"]]
            print(
                f"HTML 連結：{len(links):,} picks，有連結 {links[['href_bbref','href_minors']].notna().any(axis=1).sum():,}，對到 MLBAM {links['key_mlbam_href'].notna().sum():,}",
                flush=True,
            )
    ids["key"] = (ids["name_first"].fillna("") + " " + ids["name_last"].fillna("")).map(norm)
    draft["key"] = draft["name"].map(norm)
    draft["signed"] = draft["signed"].astype(str).str.upper().str.startswith("Y") if "signed" in draft else True

    cand = draft.reset_index().merge(
        ids[["key", "key_mlbam", "pro_played_first", "mlb_played_first"]], on="key", how="left"
    )
    cand["lag"] = cand["pro_played_first"] - cand["year"]
    ok = cand["lag"].between(0, 1)
    # 第二輪：允許 0~3 年（未簽約隔年再選、傷停）
    ok2 = cand["lag"].between(0, 3)
    best = pd.concat([cand[ok], cand[~ok & ok2]]).sort_values(["index", "lag"]).drop_duplicates("index", keep="first")
    n_amb = cand[ok].groupby("index")["key_mlbam"].nunique().gt(1).sum()
    linked = draft.copy()
    linked["key_mlbam_name"] = best.set_index("index")["key_mlbam"].reindex(linked.index).astype("Int64")
    if not via_href.empty:
        linked = linked.merge(via_href, on=["year", "pick_overall"], how="left")
        linked["key_mlbam"] = linked["key_mlbam_href"].fillna(linked["key_mlbam_name"]).astype("Int64")
        linked["link_source"] = (
            linked["key_mlbam_href"].notna().map({True: "href", False: "name"}).where(linked["key_mlbam"].notna(), None)
        )
        linked = linked.drop(columns=["key_mlbam_href"])
    else:
        linked["key_mlbam"] = linked["key_mlbam_name"]
        linked["link_source"] = "name"
    linked = linked.drop(columns=["key_mlbam_name"])
    first = ids_all[["key_mlbam", "mlb_played_first"]].dropna(subset=["key_mlbam"]).drop_duplicates("key_mlbam")
    linked = linked.drop(columns=[c for c in ("mlb_played_first",) if c in linked.columns]).merge(
        first, on="key_mlbam", how="left"
    )
    linked.drop(columns="key").to_parquet(data / "draft_linked.parquet", index=False)

    md = ["# 選秀表連接\n"]
    sgn = linked[linked["signed"]]
    md.append(
        f"選秀 {len(linked):,} 人，簽約 {len(sgn):,}；簽約者配到 MLBAM id {sgn['key_mlbam'].notna().sum():,}（{100 * sgn['key_mlbam'].notna().mean():.1f}%）；"
        f"來源：連結 {int((sgn['link_source']=='href').sum()):,}、姓名 {int((sgn['link_source']=='name').sum()):,}；同名多候選 {n_amb} 筆\n"
    )
    md.append("### 配對率 × 年份（簽約者）\n")
    md.append(
        (sgn.groupby("year")["key_mlbam"].apply(lambda s: round(100 * s.notna().mean(), 1)))
        .to_frame("配到 %")
        .T.to_markdown()
    )
    md.append("\n### 配對率 × 登上一軍（簽約者）\n")
    md.append(
        (sgn.groupby("reached_mlb")["key_mlbam"].apply(lambda s: round(100 * s.notna().mean(), 1)))
        .to_frame("配到 %")
        .to_markdown()
    )

    # 路徑表對照
    ps_path = data / "player_season.parquet"
    if ps_path.exists():
        ps = pd.read_parquet(ps_path)
        dr = (
            linked[linked["key_mlbam"].notna()]
            .sort_values("year")
            .drop_duplicates("key_mlbam", keep="last")[["key_mlbam", "year", "pick_overall", "level", "round"]]
        )
        dr = dr.rename(columns={"year": "draft_year", "level": "draft_level"})
        ps = ps.merge(dr, on="key_mlbam", how="left")
        ps["origin"] = ps["draft_year"].notna().map({True: "drafted", False: "undrafted"})
        sub = ps[(ps["role"] == "bat") & ~ps["post_debut"] & (ps["next_status"] != "censored")]
        sub = sub[sub["season"] < sub["season"].max()]
        for name, col, fn in (("升級率 promoted+mlb %", "promoted", "mean"), ("離開率 gone_exit %", None, None)):
            if col:
                t = sub.groupby(["origin", "max_rank"])[col].mean().unstack(0).round(3) * 100
            else:
                t = (
                    sub.assign(x=sub["next_status"] == "gone_exit")
                    .groupby(["origin", "max_rank"])["x"]
                    .mean()
                    .unstack(0)
                    .round(3)
                    * 100
                )
            t.index = t.index.map(RANK_NAME)
            md.append(f"\n### {name} × 層級 × 出身（打者）\n")
            md.append(t.to_markdown())
        md.append("\n### 離開率 gone_exit % × 層級 × 選秀來源類型（打者，僅選秀出身）\n")
        d2 = sub[sub["origin"] == "drafted"]
        t = (
            d2.assign(x=d2["next_status"] == "gone_exit")
            .groupby(["draft_level", "max_rank"])["x"]
            .mean()
            .unstack(0)
            .round(3)
            * 100
        )
        t.index = t.index.map(RANK_NAME)
        md.append(t.to_markdown())
        md.append(
            f"\n路徑表人季：選秀出身 {int((ps['origin']=='drafted').sum()):,}，非選秀 {int((ps['origin']=='undrafted').sum()):,}\n"
        )
    (out / "draft_link.md").write_text("\n".join(md))
    print("\n".join(md[:2]))
    print(f"-> {out / 'draft_link.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data", sys.argv[2] if len(sys.argv) > 2 else "reports")
