"""逐球版的打擊樣貌：用 Savant 小聯盟 Statcast（3A 2023 起）算每個人季真的點了幾次、內野安打幾支、幾顆被接殺、打多強多高，
看這些在四把尺之外有沒有影響球隊留不留他、升不升他。

    python src/hit_profile_statcast.py outputs/ data/ --promoted promoted_f4sed_linear_p2 [--level AAA]
    python src/hit_profile_statcast.py --selftest

輸入：data/savant_minors_<season>.parquet（src/fetch_savant_minors.py 抓的；data/ 沒有就用 share/ 裡推上 git 的那份），有哪幾季就用哪幾季。
每個人季（同一層級）從逐球算：
  點：有觸擊動作的打席比例（des 含 bunt，或 events 是 sac_bunt / bunt_*）、觸擊安打 ÷ 安打。
  內野安打：滾地球形成的一壘安打、而且是內野手（守位 1–6）處理的 ÷ 安打。
  接殺：飛球與內野高飛被接出局 ÷ 進場球；內野高飛 ÷ 進場球；滾地 ÷ 進場球。
  打多強：平均初速、強勁擊球率（≥ 95 mph）、barrel 率（launch_speed_angle = 6）、平均仰角、理想仰角率（8–32 度）、
         接觸後的預期 wOBA（xwOBA on contact）。
  方向：順向（拉打）比例、反向比例，用落點 hc_x/hc_y 算噴射角再依左右打翻轉；拉打飛球率 = 拉打且仰角 ≥ 20 度 ÷ 進場球（文獻：拉打本身不好、拉打飛球才好）。
  文獻說最穩的過程量（docs/advanced_predictors.md）：EV90 = 初速第 90 百分位；區內接觸率 = 九宮格內揮棒沒揮空的比例（要有 zone 欄）。
  洞（要有進壘點欄 zone，舊抓的資料沒有就跳過）：九宮格裡揮空率最高那格的揮空率減他九格整體的揮空率（洞有多深，格內揮棒 ≥ MIN_SWING_ZONE）、
       投手投進那格的比例（洞被攻多少；九格平均是 1/9）、整體揮空率、追打壞球率（zone ≥ 11 的球揮棒比例）。
結果與基底跟 hit_profile 一樣：被留下來、留在同一隊、隔年升不升、直接上大聯盟；基底 = 四把尺 + 球探評價 + 年齡 + 打席 + 層級。
再加兩個「真相」結果：到資料最後一季為止有沒有登上大聯盟、登上者從那季起到最後一季的 fWAR（asinh）；
2023、2024 的人六季窗口還沒走完，所以窗口是「到目前為止」，逐季各自迴歸，窗口長短由季吸收。WAR 來自 FanGraphs 大聯盟排行榜，不是 StatsAPI。
只有兩三季，所以主要看逐季係數與合併；樣本外那欄要三季以前的資料才算得出來，多半是空的。
輸出 outputs/hit_profile_statcast_<promoted tag>.md。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hit_profile import BASE, OUTCOMES, add_retention, one_table
from decision_style import LEVELS

BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play", "bunt_groundout", "bunt_pop_out", "bunt_lineout"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
OUT_EVENTS = {
    "field_out",
    "sac_fly",
    "double_play",
    "grounded_into_double_play",
    "force_out",
    "fielders_choice_out",
    "sac_fly_double_play",
    "triple_play",
}
# 變數名 → 中文（給表用）
LABELS = {
    "bunt_pa": "觸擊打席比例（點）",
    "bunt_h": "觸擊安打 ÷ 安打",
    "ifh_h": "內野安打 ÷ 安打",
    "caught_bip": "飛球被接殺 ÷ 進場球",
    "popup_bip": "內野高飛 ÷ 進場球",
    "gb_bip": "滾地 ÷ 進場球",
    "ev_mean": "平均擊球初速",
    "hard_bip": "強勁擊球率（≥ 95 mph）",
    "barrel_bip": "barrel 率",
    "la_mean": "平均仰角",
    "sweet_bip": "理想仰角率（8–32 度）",
    "xwoba_con": "接觸後預期 wOBA",
    "pull_bip": "順向（拉打）比例",
    "oppo_bip": "反向比例",
    "ev90": "初速第 90 百分位（EV90，最強那一成球）",
    "pullair_bip": "拉打飛球率（拉打且仰角 ≥ 20 度 ÷ 進場球）",
    "zcon": "區內接觸率（Z-Contact%）",
    "whiff_sw": "整體揮空率（揮空 ÷ 揮棒）",
    "hole_gap": "洞有多深（最差一格揮空率 − 九格整體）",
    "hole_attack": "洞被攻多少（投進最差那格的比例，均勻是 0.11）",
    "chase": "追打壞球率",
}
MIN_BIP = 50
MIN_SWING_ZONE = 15  # 一格至少揮這麼多次才算得出那格的揮空率
SWING_DESC = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "foul_bunt",
    "missed_bunt",
    "bunt_foul_tip",
}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}


def spray_side(hc_x: pd.Series, hc_y: pd.Series, stand: pd.Series) -> pd.Series:
    """落點 → 順向 / 中間 / 反向。噴射角 = atan((hc_x − 125.42) / (198.27 − hc_y))，右打者負角度是拉打、左打者相反；±15 度以內算中間。"""
    ang = np.degrees(np.arctan2(hc_x - 125.42, 198.27 - hc_y))
    ang = np.where(stand == "L", -ang, ang)  # 翻成「負 = 拉打」對兩種打者都成立
    side = np.select([ang < -15, ang > 15], ["pull", "oppo"], "center")
    side = np.where(np.isfinite(ang), side, None)
    return pd.Series(side, index=hc_x.index)


def zone_profile(pitches: pd.DataFrame) -> pd.DataFrame:
    """每一球（不只打席結束那球）→ 每位打者每季每層級的洞：九宮格（zone 1–9）裡揮空率最高的那格有多差、投手投那格的比例、整體揮空率、追打壞球率。
    沒有 zone 欄（舊抓的資料）回傳空表，四個變數留空。"""
    keys = ["batter", "game_year", "level"]
    if "zone" not in pitches.columns:
        return pd.DataFrame(columns=keys + ["whiff_sw", "hole_gap", "hole_attack", "chase", "zcon"])
    p = pitches[pitches["zone"].notna()].copy()
    p["zone"] = pd.to_numeric(p["zone"], errors="coerce")
    p["swing"] = p["description"].isin(SWING_DESC)
    p["whiff"] = p["description"].isin(WHIFF_DESC)
    p["in_zone"] = p["zone"].between(1, 9)
    g = p.groupby(keys)
    z = p[p["in_zone"]]
    gz = z.groupby(keys)
    overall = pd.DataFrame(
        {
            "whiff_sw": g["whiff"].sum() / g["swing"].sum().clip(lower=1),
            "chase": p[~p["in_zone"]].groupby(keys)["swing"].mean(),
            "zcon": 1 - gz["whiff"].sum() / gz["swing"].sum().clip(lower=1),  # 區內揮棒碰到球的比例
        }
    )
    cell = z.groupby(keys + ["zone"]).agg(pitches=("swing", "size"), swings=("swing", "sum"), whiffs=("whiff", "sum"))
    cell["rate"] = cell["whiffs"] / cell["swings"].clip(lower=1)
    cell = cell[cell["swings"] >= MIN_SWING_ZONE]
    worst = cell.sort_values("rate").groupby(keys).tail(1).reset_index()  # 每個人季揮空率最高的那格
    worst = worst.set_index(keys)
    hole = pd.DataFrame(
        {
            "hole_gap": worst["rate"] - gz["whiff"].sum() / gz["swing"].sum().clip(lower=1),
            "hole_attack": worst["pitches"] / gz.size(),
        }
    )
    return overall.join(hole, how="left").reset_index()


def per_player_season(pitches: pd.DataFrame) -> pd.DataFrame:
    """逐球 → 每位打者每季每層級一列的打擊樣貌。打球的部分只用有結果的那一球（打席結束）；洞的部分用每一球。"""
    pa = pitches[pitches["events"].notna()].copy()
    des = pa["des"].fillna("").str.lower()
    pa["is_bunt"] = des.str.contains("bunt") | pa["events"].isin(BUNT_EVENTS)
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS)
    pa["is_bip"] = pa["bb_type"].notna()
    pa["is_out"] = pa["events"].isin(OUT_EVENTS)
    pa["bunt_hit"] = pa["is_bunt"] & pa["is_hit"]
    loc = pd.to_numeric(pa["hit_location"], errors="coerce")
    pa["ifh"] = (pa["events"] == "single") & (pa["bb_type"] == "ground_ball") & loc.between(1, 6)
    pa["caught"] = pa["is_bip"] & pa["is_out"] & pa["bb_type"].isin(["fly_ball", "popup", "line_drive"])
    pa["popup"] = pa["bb_type"] == "popup"
    pa["gb"] = pa["bb_type"] == "ground_ball"
    ev, la = pd.to_numeric(pa["launch_speed"], errors="coerce"), pd.to_numeric(pa["launch_angle"], errors="coerce")
    pa["hard"] = ev >= 95
    pa["barrel"] = pd.to_numeric(pa["launch_speed_angle"], errors="coerce") == 6
    pa["sweet"] = la.between(8, 32)
    pa["ev"], pa["la"] = ev.where(pa["is_bip"]), la.where(pa["is_bip"])
    pa["xwoba"] = pd.to_numeric(pa["estimated_woba_using_speedangle"], errors="coerce").where(pa["is_bip"])
    side = spray_side(
        pd.to_numeric(pa["hc_x"], errors="coerce"), pd.to_numeric(pa["hc_y"], errors="coerce"), pa["stand"]
    )
    pa["pull"], pa["oppo"] = side == "pull", side == "oppo"
    pa["pullair"] = (side == "pull") & (la >= 20)
    keys = ["batter", "game_year", "level"]
    g = pa.groupby(keys)
    out = pd.DataFrame(
        {
            "pa": g.size(),
            "h": g["is_hit"].sum(),
            "bip": g["is_bip"].sum(),
            "bunt_pa": g["is_bunt"].mean(),
            "bunt_h": g["bunt_hit"].sum() / g["is_hit"].sum().clip(lower=1),
            "ifh_h": g["ifh"].sum() / g["is_hit"].sum().clip(lower=1),
            "caught_bip": g["caught"].sum() / g["is_bip"].sum().clip(lower=1),
            "popup_bip": g["popup"].sum() / g["is_bip"].sum().clip(lower=1),
            "gb_bip": g["gb"].sum() / g["is_bip"].sum().clip(lower=1),
            "ev_mean": g["ev"].mean(),
            "hard_bip": g["hard"].sum() / g["is_bip"].sum().clip(lower=1),
            "barrel_bip": g["barrel"].sum() / g["is_bip"].sum().clip(lower=1),
            "la_mean": g["la"].mean(),
            "sweet_bip": g["sweet"].sum() / g["is_bip"].sum().clip(lower=1),
            "xwoba_con": g["xwoba"].mean(),
            "pull_bip": g["pull"].sum() / g["is_bip"].sum().clip(lower=1),
            "oppo_bip": g["oppo"].sum() / g["is_bip"].sum().clip(lower=1),
            "pullair_bip": g["pullair"].sum() / g["is_bip"].sum().clip(lower=1),
            "ev90": g["ev"].quantile(0.9),
        }
    ).reset_index()
    zp = zone_profile(pitches)
    if len(zp):
        out = out.merge(zp, on=keys, how="left")
    else:
        out = out.assign(whiff_sw=np.nan, hole_gap=np.nan, hole_attack=np.nan, chase=np.nan, zcon=np.nan)
    return out.rename(columns={"batter": "key_mlbam", "game_year": "season", "level": "level_main"})


def add_truth(m: pd.DataFrame, data: Path) -> pd.DataFrame:
    """到資料最後一季為止：有沒有登上大聯盟（y_reach）、登上者從那季起的 fWAR 總和（y_war，asinh）。
    FanGraphs 大聯盟排行榜在 data/fg_batting.parquet、fg_pitching.parquet（Arc 的建表資料），沒有就用 share/ 的那份。"""
    share = Path(__file__).resolve().parent.parent / "share"
    frames = []
    for kind in ("batting", "pitching"):
        for f in (data / f"fg_{kind}.parquet", *sorted(share.glob(f"fg_{kind}_*.parquet"))):
            if f.exists():
                frames.append(pd.read_parquet(f, columns=["key_mlbam", "Season", "WAR"]))
                break
    if not frames:
        m["y_reach"], m["y_war"] = np.nan, np.nan
        return m
    fg = pd.concat(frames).dropna(subset=["key_mlbam"]).astype({"key_mlbam": "int64", "Season": "int64"})
    last = int(fg["Season"].max())
    war = fg.groupby(["key_mlbam", "Season"])["WAR"].sum()
    tot = []
    for pid, s in zip(m["key_mlbam"], m["season"]):
        tot.append(sum(war.get((int(pid), y), 0.0) for y in range(int(s), last + 1)))
    reached = m["mlb_played_first"].notna() & (m["mlb_played_first"] <= last) & (m["mlb_played_first"] >= m["season"])
    m["y_reach"] = np.where(m["season"] < last, reached.astype(float), np.nan)  # 最後一季的人沒有「之後」，留空
    m["y_war"] = np.where(reached & (m["season"] < last), np.arcsinh(np.array(tot, dtype=float)), np.nan)
    m["fg_last"] = last
    return m


def load_pitches(data: Path, season: int) -> pd.DataFrame:
    """一季的 Savant 逐球：data/savant_minors_<season>.parquet，沒有就用 repo 裡 share/ 的；都沒有回傳空表。"""
    for f in (data / f"savant_minors_{season}.parquet", Path(__file__).resolve().parent.parent / "share" / f"savant_minors_{season}.parquet"):
        if f.exists():
            return pd.read_parquet(f)
    return pd.DataFrame()


def statcast_counts(pitches: pd.DataFrame) -> pd.DataFrame:
    """fit_year --statcast 要的計數版指標（分子、分母成對，給二項；連續量給常態）：每人季每層級一列。
    sc_bip 進場球；sc_barrel、sc_pullair（拉打且仰角 ≥ 20）；sc_ev90、sc_ev10 初速第 90、10 百分位；sc_xwobacon 接觸後 xwOBA 平均；
    sc_zswing 區內揮棒、sc_zcon 區內碰到球；sc_ozpitch 區外球數、sc_chase 區外揮棒（要有 zone 欄，沒有就空）。"""
    cols = ["key_mlbam", "season", "level_main", "sc_bip", "sc_barrel", "sc_pullair", "sc_ev90", "sc_ev10", "sc_xwobacon", "sc_zswing", "sc_zcon", "sc_ozpitch", "sc_chase"]
    if len(pitches) == 0:
        return pd.DataFrame(columns=cols)
    keys = ["batter", "game_year", "level"]
    pa = pitches[pitches["events"].notna() & pitches["bb_type"].notna()].copy()
    ev, la = pd.to_numeric(pa["launch_speed"], errors="coerce"), pd.to_numeric(pa["launch_angle"], errors="coerce")
    side = spray_side(pd.to_numeric(pa["hc_x"], errors="coerce"), pd.to_numeric(pa["hc_y"], errors="coerce"), pa["stand"])
    pa["barrel"] = (pd.to_numeric(pa["launch_speed_angle"], errors="coerce") == 6).astype(float)
    pa["pullair"] = ((side == "pull") & (la >= 20)).astype(float)
    pa["ev"], pa["xw"] = ev, pd.to_numeric(pa["estimated_woba_using_speedangle"], errors="coerce")
    g = pa.groupby(keys)
    out = pd.DataFrame(
        {
            "sc_bip": g.size(),
            "sc_barrel": g["barrel"].sum(),
            "sc_pullair": g["pullair"].sum(),
            "sc_ev90": g["ev"].quantile(0.9),
            "sc_ev10": g["ev"].quantile(0.1),
            "sc_xwobacon": g["xw"].mean(),
        }
    )
    few = out["sc_bip"] < 20  # 進場球太少，十分位與平均沒意義，留空（二項計數照留，分母小似然自己知道）
    out.loc[few, ["sc_ev90", "sc_ev10", "sc_xwobacon"]] = np.nan
    if "zone" in pitches.columns:
        p = pitches[pitches["zone"].notna()].copy()
        p["zone"] = pd.to_numeric(p["zone"], errors="coerce")
        p["swing"] = p["description"].isin(SWING_DESC).astype(float)
        p["contact"] = (p["swing"] > 0) & ~p["description"].isin(WHIFF_DESC)
        inz = p[p["zone"].between(1, 9)].groupby(keys)
        outz = p[~p["zone"].between(1, 9)].groupby(keys)
        out["sc_zswing"] = inz["swing"].sum()
        out["sc_zcon"] = inz["contact"].sum()
        out["sc_ozpitch"] = outz.size()
        out["sc_chase"] = outz["swing"].sum()
    else:
        for c in ("sc_zswing", "sc_zcon", "sc_ozpitch", "sc_chase"):
            out[c] = np.nan
    out = out.reset_index().rename(columns={"batter": "key_mlbam", "game_year": "season", "level": "level_main"})
    return out[cols]


def load(out: Path, data: Path, tag_p: str, level: str, profile_fn=None, labels: dict | None = None) -> tuple:
    """逐球彙整後接上建模表（結果、基底）與模型分數；只留那個層級、進場球 ≥ MIN_BIP 的人季；變數在季內標準化。
    profile_fn 是「逐球 → 每人季一列」的函數（預設這支的 per_player_season；dist_profile 會給它自己的分布版），labels 是那些變數的中文名。"""
    profile_fn = per_player_season if profile_fn is None else profile_fn
    labels = LABELS if labels is None else labels
    files = sorted(data.glob("savant_minors_*.parquet"))
    if not files:  # data/ 沒有就用 repo 裡 share/ 的那份（2023、2024 的 3A 已經推上 git）
        files = sorted((Path(__file__).resolve().parent.parent / "share").glob("savant_minors_*.parquet"))
    if not files:
        raise SystemExit("找不到 savant_minors_<season>.parquet（data/ 或 share/），先跑 src/fetch_savant_minors.py")
    pitches = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    prof = profile_fn(pitches)
    prof = prof[(prof["level_main"] == level) & (prof["bip"] >= MIN_BIP)]
    m = add_truth(add_retention(pd.read_parquet(data / "model_bat.parquet")), data)
    m["y_mlb_next"] = np.where(m["y_keep"].notna(), (m["next_status"].astype(str) == "mlb").astype(float), np.nan)
    m = m[(m["level_main"] == level) & m["structural"].astype(bool)]  # 跟主模型一樣：還沒登上大聯盟、沒被截尾的人
    frames = []
    for season in sorted(prof["season"].unique()):
        p = out / str(int(season)) / tag_p / "eta_M1.parquet"
        if not p.exists():
            print(f"  {int(season)} 沒有 {tag_p} 的模型分數，跳過", flush=True)
            continue
        e = pd.read_parquet(p)[["key_mlbam", "season"] + BASE[:5]]
        frames.append(m[m["season"] == season].merge(e, on=["key_mlbam", "season"], how="inner"))
    if not frames:
        raise SystemExit("沒有任何一季同時有 Statcast 與模型分數")
    d = pd.concat(frames, ignore_index=True).merge(prof, on=["key_mlbam", "season", "level_main"], how="inner")
    d["zeta_scout"] = d["zeta_scout"].fillna(0.0)
    d["log_pa"] = np.log(d["PA"])
    names = [v for v in labels if v in d.columns and d[v].notna().sum() > 0]
    for c in BASE + names:
        d[c] = d[c].astype(float)
        d[c] = d.groupby("season")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
    d["y_prom"] = d["promoted"].astype(float)
    return d, names


def describe(d: pd.DataFrame, names: list, outcome: str, label: str) -> pd.DataFrame:
    """先看原始平均：結果 = 1 與 = 0 的兩群人，每個逐球變數（標準化後）的平均差多少。"""
    ok = d[outcome].notna()
    a, b = d[ok & (d[outcome] == 1)], d[ok & (d[outcome] == 0)]
    rows = [
        {
            "逐球變數": LABELS[v],
            f"{label}=1 的平均": round(a[v].mean(), 2),
            f"{label}=0 的平均": round(b[v].mean(), 2),
            "差": round(a[v].mean() - b[v].mean(), 2),
        }
        for v in names
    ]
    return pd.DataFrame(rows)


def main(out_dir: str, data_dir: str, tag_p: str, level: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    d, names = load(out, data, tag_p, level)
    outcomes = [
        ("y_keep", "被留下來：隔季還在（或直接上大聯盟）"),
        ("y_same", "留在同一隊"),
        ("y_prom", "隔年升不升（3A 升上去就是大聯盟）"),
        ("y_mlb_next", "直接上大聯盟（隔季身分是 mlb）"),
        ("y_reach", "真相 · 到資料最後一季為止有沒有登上大聯盟"),
        ("y_war", "真相 · 登上者從那季起到最後一季的 fWAR（asinh，線性迴歸）"),
    ]
    seasons = ", ".join(str(int(s)) for s in sorted(d["season"].unique()))
    fg_last = int(d["fg_last"].iloc[0]) if "fg_last" in d.columns and d["fg_last"].notna().any() else None
    has_zone = "hole_gap" in names
    md = [
        f"# 逐球版打擊樣貌（Statcast，{level}）：點、內野安打、接殺、打多強，影響留不留嗎 · {tag_p}\n",
        f"樣本：{len(d):,} 人季（{seasons}），{level}，還沒登上大聯盟的人（structural 遮罩，跟主模型一樣），進場球 ≥ {MIN_BIP}。每個係數 = 四把尺、球探評價、年齡、打席一樣的兩個人，"
        "那個逐球變數差一個標準差（季內），log-odds 差多少；0.1 約是五成變五成二。只有兩三季，「幾季正 / 幾季負」看方向，"
        "樣本外那欄要三季以前的資料，多半空白。觸擊 = 描述含 bunt 的打席；內野安打 = 滾地形成、守位 1–6 處理的一壘安打；"
        "接殺 = 飛球、平飛、內野高飛被接出局。"
        + (f" 真相兩段的窗口是「那季起到 {fg_last}」（FanGraphs 最後一季），不是主模型的六季。" if fg_last else "")
        + (
            " 洞 = 九宮格裡揮空率最高那格；洞被攻 = 投手投進那格的比例。"
            if has_zone
            else " 這批逐球資料沒有進壘點欄，洞的四個變數算不出來（重抓 fetch_savant_minors.py 就會有）。"
        )
        + "\n",
    ]
    for outcome, label in outcomes:
        ok = d[outcome].notna()
        if ok.sum() < 50 or d.loc[ok, outcome].nunique() < 2:
            continue
        if outcome == "y_mlb_next" and (d.loc[ok, "y_mlb_next"] == d.loc[ok, "y_prom"]).all():
            continue  # 3A 升上去就是大聯盟，兩個結果一樣，不重印
        if outcome == "y_war":
            md += [f"## {label}（{int(ok.sum()):,} 人季，WAR 中位數 {np.sinh(d.loc[ok, outcome]).median():.1f}）\n"]
        else:
            md += [f"## {label}（{int(ok.sum()):,} 人季，正例 {d.loc[ok, outcome].mean():.1%}）\n"]
            md += ["原始平均（標準化單位）：\n", describe(d, names, outcome, label[:5]).to_markdown(index=False), ""]
        tab = one_table(d, names, outcome, LABELS)
        if not tab.empty:
            md += ["四把尺之外的係數：\n", tab.drop(columns="_se").to_markdown(index=False), ""]
    text = "\n".join(md)
    path = out / f"hit_profile_statcast_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成逐球：點很多的人比較不被留下、barrel 多的人比較會被留下，看彙整與方向。"""
    rng = np.random.default_rng(1)
    root = Path(tempfile.mkdtemp())
    data, out = root / "data", root / "outputs"
    data.mkdir()
    etas = ["eta_" + f for f in ("contact", "power", "speed", "discipline")]
    rows, table = [], []
    for season in (2023, 2024, 2025):
        for i in range(300):
            key = 500000 + i
            bunt_p, barrel_p = rng.uniform(0, 0.08), rng.uniform(0, 0.15)
            n = rng.integers(150, 400)
            ev = rng.normal(88, 8, n)
            bb = rng.choice(
                ["ground_ball", "fly_ball", "line_drive", "popup", None], n, p=[0.3, 0.25, 0.15, 0.05, 0.25]
            )
            is_bunt = rng.random(n) < bunt_p
            events = np.where(
                bb == None,
                rng.choice(["strikeout", "walk"], n),
                rng.choice(["field_out", "single", "double", "home_run"], n, p=[0.65, 0.25, 0.07, 0.03]),
            )  # noqa: E711
            events = np.where(is_bunt, "sac_bunt", events)
            zone = rng.choice(np.r_[1:10, 11:15], n)
            desc = np.where(
                bb == None, rng.choice(["swinging_strike", "called_strike", "ball", "foul"], n), "hit_into_play"
            )  # noqa: E711
            rows.append(
                pd.DataFrame(
                    {
                        "batter": key,
                        "game_year": season,
                        "level": "AAA",
                        "events": events,
                        "description": desc,
                        "zone": zone,
                        "des": np.where(is_bunt, "x bunt grounds out", "x singles"),
                        "bb_type": bb,
                        "hit_location": rng.integers(1, 10, n).astype(float),
                        "launch_speed": ev,
                        "launch_angle": rng.normal(12, 20, n),
                        "launch_speed_angle": np.where(rng.random(n) < barrel_p, 6, 3),
                        "estimated_woba_using_speedangle": rng.uniform(0, 1, n),
                        "hc_x": rng.uniform(0, 250, n),
                        "hc_y": rng.uniform(0, 200, n),
                        "stand": rng.choice(["L", "R"], n),
                    }
                )
            )
            keep = rng.random() < 1 / (1 + np.exp(-(0.5 - 15 * bunt_p + 6 * barrel_p)))
            table.append(
                {
                    "key_mlbam": key,
                    "season": season,
                    "level_main": "AAA",
                    "org": "NYY",
                    "PA": n,
                    "promoted": rng.random() < 0.3,
                    "next_status": "mlb" if rng.random() < 0.05 else "same",
                    "mlb_played_first": np.nan,
                    "structural": True,
                    "age_rel": rng.normal(),
                    "_keep": keep,
                    **{e: rng.normal() for e in etas},
                }
            )
    pitches = pd.concat(rows, ignore_index=True)
    m = pd.DataFrame(table)
    kept = m[m["_keep"] | (m["season"] == 2023)].copy()  # 被留下的人隔季才有列（第一季全留，當起點）
    nxt = kept[kept["season"] < 2025].assign(season=lambda x: x["season"] + 1)
    m = pd.concat(
        [
            m[m["season"] == 2023],
            nxt[nxt["season"] == 2024].drop(columns="_keep").assign(_keep=True),
            nxt[nxt["season"] == 2025].drop(columns="_keep").assign(_keep=True),
        ],
        ignore_index=True,
    ).drop_duplicates(["key_mlbam", "season"])
    for season in (2023, 2024, 2025):
        pitches[pitches["game_year"] == season].to_parquet(data / f"savant_minors_{season}.parquet", index=False)
        (out / str(season) / "promoted_test").mkdir(parents=True)
        m[m["season"] == season][["key_mlbam", "season"] + etas].assign(zeta_scout=np.nan).to_parquet(
            out / str(season) / "promoted_test" / "eta_M1.parquet", index=False
        )
    m.drop(columns=etas + ["_keep"]).to_parquet(data / "model_bat.parquet", index=False)
    prof = per_player_season(pitches)
    assert prof["bunt_pa"].between(0, 1).all() and prof["bip"].gt(0).all(), "彙整比例不對"
    assert prof["hole_gap"].notna().any() and prof["hole_attack"].dropna().between(0, 1).all(), "洞沒算出來"
    text = main(str(out), str(data), "promoted_test", "AAA")
    d, names = load(out, data, "promoted_test", "AAA")
    tab = one_table(d, names, "y_keep", LABELS).set_index("打擊樣貌")
    assert tab.loc[LABELS["bunt_pa"], "合併係數"] < 0 and tab.loc[LABELS["barrel_bip"], "合併係數"] > 0, "方向沒抓到"
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="逐球版打擊樣貌（Savant 小聯盟 Statcast）對留不留、升不升的作用")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("--promoted", default=None, help="升級模型的 tag，例如 promoted_f4sed_linear_p2")
    p.add_argument("--level", default="AAA", choices=list(LEVELS), help="哪個層級（Statcast 只有 AAA 與 A 有）")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.promoted is None:
        p.error("--promoted 是必要的（或用 --selftest）")
    else:
        main(a.out_dir, a.data_dir, a.promoted, a.level)
