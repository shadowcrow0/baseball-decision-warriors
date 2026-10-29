"""「該再看一眼」的名單：某一季每個層級，真相的尺分數高、但球隊沒升的人；以及球隊升了、真相的尺排很後面的人。給投影片的「實際應用」頁。

    python src/watchlist.py outputs/ data/ 2024 [--levels AA AAA] [--top 8] [--min-pa 150]
    python src/watchlist.py --selftest

真相的尺 = WAR 段十三季（2007–2019）學到的權重，反變異數合併：Σ_k w̄_k·η_ik + w̄_scout·ζ_i（係數在 outputs/<季>/war_<tag>/struct_M1.csv 的 w_beta_*）。
球隊的尺 = 該季升級模型的線性預測（eta_M1.parquet 的 score_promoted）。
四把尺換成球探的 20–80 分：層級內標準化後 50 + 10·z。
姓名、守位從 data/minors_bat_clean.parquet 接（Arc 有；本機沒有就印 key_mlbam）。
輸出 outputs/watchlist_<季>.md 與 .csv。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ABILITIES = ["contact", "power", "speed", "discipline"]
LABEL = {"contact": "打得到球", "power": "打得遠", "speed": "跑得快", "discipline": "選得到球"}
NOT_PROMOTED = {"same", "demoted", "gone_exit", "gone_return"}


def truth_weights(out: Path, war_tag: str) -> dict:
    """WAR 段每季的 w_beta_*，反變異數合併成一組權重。"""
    rows = []
    for p in sorted(out.glob(f"20*/{war_tag}/struct_M1.csv")):
        s = pd.read_csv(p, index_col=0)
        r = {}
        for k in ABILITIES + ["scout"]:
            key = f"w_beta_{k}[0]" if k != "scout" else "w_beta_scout"
            if key in s.index:
                r[k] = (float(s.loc[key, "mean"]), float(s.loc[key, "sd"]))
        if len(r) == 5:
            rows.append(r)
    if not rows:
        raise SystemExit(f"找不到 {war_tag} 的 struct_M1.csv")
    w = {}
    for k in ABILITIES + ["scout"]:
        b = np.array([r[k][0] for r in rows])
        se = np.array([r[k][1] for r in rows])
        wt = 1 / se**2
        w[k] = float((b * wt).sum() / wt.sum())
    w["n_seasons"] = len(rows)
    return w


def load_names(data: Path, season: int) -> pd.DataFrame | None:
    p = data / "minors_bat_clean.parquet"
    if not p.exists():
        return None
    cols = [c for c in ("key_mlbam", "season", "name", "positionDB", "org", "team") if c in pd.read_parquet(p).columns]
    d = pd.read_parquet(p, columns=cols)
    d = d[d["season"] == season].drop_duplicates("key_mlbam")
    return d.drop(columns=["season"])


def scale_20_80(x: pd.Series) -> pd.Series:
    return (50 + 10 * (x - x.mean()) / (x.std() + 1e-9)).round(0)


def build(e: pd.DataFrame, w: dict, levels: list, min_pa: int, top: int, names: pd.DataFrame | None) -> tuple:
    d = e[e["level_main"].isin(levels) & (e["PA"] >= min_pa) & e["score_promoted"].notna()].copy()
    d["zeta_scout"] = d["zeta_scout"].fillna(0.0)
    d["truth"] = sum(w[k] * d[f"eta_{k}"] for k in ABILITIES) + w["scout"] * d["zeta_scout"]
    for k in ABILITIES:
        d[f"g_{k}"] = d.groupby("level_main")[f"eta_{k}"].transform(scale_20_80)
    d["truth_pct"] = d.groupby("level_main")["truth"].rank(pct=True).mul(100).round(0)
    d["club_pct"] = d.groupby("level_main")["score_promoted"].rank(pct=True).mul(100).round(0)
    if names is not None:
        d = d.merge(names, on="key_mlbam", how="left", suffixes=("", "_n"))
    else:
        d["name"] = d["key_mlbam"].astype(str)
    d["name"] = d["name"].fillna(d["key_mlbam"].astype(str))
    missed = d[d["next_status"].isin(NOT_PROMOTED)].sort_values(["level_main", "truth"], ascending=[True, False]).groupby("level_main").head(top)
    doubtful = d[d["next_status"].isin({"promoted", "mlb"})].sort_values(["level_main", "truth"], ascending=[True, True]).groupby("level_main").head(top)
    return d, missed, doubtful


def table(t: pd.DataFrame) -> pd.DataFrame:
    cols = {"level_main": "層級", "name": "球員", "age": "年齡", "PA": "打席", "g_contact": "打得到球", "g_power": "打得遠", "g_speed": "跑得快", "g_discipline": "選得到球", "truth_pct": "真相的尺（層級內百分位）", "club_pct": "球隊的尺（百分位）", "next_status": "隔年去向"}
    if "positionDB" in t.columns:
        cols = {**{"level_main": "層級", "name": "球員", "positionDB": "守位"}, **{k: v for k, v in cols.items() if k not in ("level_main", "name")}}
    out = t[[c for c in cols if c in t.columns]].rename(columns=cols)
    for c in ("年齡", "打席", "打得到球", "打得遠", "跑得快", "選得到球"):
        if c in out.columns:
            out[c] = out[c].astype(int)
    return out


def main(out_dir: str, data_dir: str, season: int, promoted_tag: str, war_tag: str, levels: list, top: int, min_pa: int) -> str:
    out, data = Path(out_dir), Path(data_dir)
    e = pd.read_parquet(out / str(season) / promoted_tag / "eta_M1.parquet")
    w = truth_weights(out, war_tag)
    names = load_names(data, season)
    d, missed, doubtful = build(e, w, levels, min_pa, top, names)
    status_map = {"same": "留原層級", "demoted": "降級", "gone_exit": "離開體系", "gone_return": "離開後回來", "promoted": "升一級", "mlb": "登上大聯盟"}
    for t in (missed, doubtful):
        t["next_status"] = t["next_status"].map(status_map).fillna(t["next_status"])
    md = [
        f"# {season} 年「該再看一眼」的名單 · {promoted_tag}\n",
        f"真相的尺 = WAR 段 {w['n_seasons']} 季合併的權重：打得到球 {w['contact']:+.2f}、打得遠 {w['power']:+.2f}、跑得快 {w['speed']:+.2f}、選得到球 {w['discipline']:+.2f}、球探評價 {w['scout']:+.2f}。"
        f"四把尺是球探的 20–80 分（層級內，50 = 該層級平均）。樣本：{'、'.join(levels)}，打席 ≥ {min_pa}，共 {len(d):,} 人季。\n",
        f"## 一、真相的尺排前面、球隊沒升的人（每層級前 {top}）\n",
        table(missed).to_markdown(index=False),
        "",
        f"## 二、球隊升了、真相的尺排最後面的人（每層級 {top}）\n",
        table(doubtful).to_markdown(index=False),
        "",
        "讀法：第一張是「每年一份名單」的樣子——不是說球隊錯了，是說這幾個人值得球探再看一眼：成績單量得到的四把尺說他們該在前面。"
        "第二張是反過來：球隊升了但尺說還早，通常是守備或球探看到成績單沒有的東西（第 21 頁）。\n",
    ]
    text = "\n".join(md)
    (out / f"watchlist_{season}.md").write_text(text)
    pd.concat([missed.assign(名單="沒升"), doubtful.assign(名單="升了")]).to_csv(out / f"watchlist_{season}.csv", index=False)
    print(text)
    return text


def selftest() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for s in (2010, 2011):
            (out / str(s) / "W").mkdir(parents=True)
            pd.DataFrame({"mean": [0.4, 0.2, 0.0, -0.1, 0.2], "sd": [0.05] * 5}, index=["w_beta_contact[0]", "w_beta_power[0]", "w_beta_speed[0]", "w_beta_discipline[0]", "w_beta_scout"]).to_csv(out / str(s) / "W" / "struct_M1.csv")
        n = 300
        e = pd.DataFrame({"key_mlbam": np.arange(n), "season": 2024, "level_main": rng.choice(["AA", "AAA"], n), "PA": rng.integers(50, 500, n), "age": rng.integers(20, 28, n), "zeta_scout": rng.normal(size=n), "score_promoted": rng.normal(size=n), "next_status": rng.choice(["same", "promoted", "mlb", "gone_exit"], n)})
        for k in ABILITIES:
            e[f"eta_{k}"] = rng.normal(size=n)
        (out / "2024" / "P").mkdir(parents=True)
        e.to_parquet(out / "2024" / "P" / "eta_M1.parquet", index=False)
        text = main(tmp, tmp, 2024, "P", "W", ["AA", "AAA"], 5, 150)
        csv = pd.read_csv(out / "watchlist_2024.csv")
    assert (csv[csv["名單"] == "沒升"]["next_status"].isin(["留原層級", "離開體系"])).all()
    assert csv.groupby(["名單", "level_main"]).size().max() <= 5
    assert "真相的尺" in text
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="該再看一眼的名單：真相的尺高但沒升、升了但尺低")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("season", nargs="?", type=int)
    p.add_argument("--promoted", default="promoted_f4sed_linear_p2")
    p.add_argument("--war", default="war_f4sed_linear_p2")
    p.add_argument("--levels", nargs="*", default=["AA", "AAA"])
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--min-pa", dest="min_pa", type=int, default=150)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        if a.season is None:
            p.error("要給 season")
        main(a.out_dir, a.data_dir, a.season, a.promoted, a.war, a.levels, a.top, a.min_pa)
