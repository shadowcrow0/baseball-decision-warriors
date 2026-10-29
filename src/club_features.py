"""球團層與「球團 × 守位」層的變數，接到每個人季上（給 fit_year --club 與 club_need 用；不取樣）。

    need_pos      母隊該守位缺口 = −(母隊當季在該守位群的 WAR 加總)，標準化；沒資料 0（pos_known = 0）
    top_war_c     母隊該守位主力（打席最多的人）的 WAR，標準化
    queue_above   農場同球團、同守位群、層級比他高的人數，標準化
    queue_total   農場同球團、同守位群的其他人數，標準化
    club_war_c    母隊當季團隊 WAR（打者 + 投手），標準化；沒資料 0
守位群來自小聯盟的 positionDB（捕手 / 游擊二壘 / 外野 / 一壘指定 / 三壘），大聯盟主守位來自 FanGraphs 的 position（fg_positions.parquet）。
"""

from pathlib import Path
import numpy as np
import pandas as pd

POS_GROUP = {
    "C": "捕手",
    "SS": "游擊/二壘",
    "2B": "游擊/二壘",
    "OF": "外野",
    "LF": "外野",
    "CF": "外野",
    "RF": "外野",
    "1B": "一壘/指定",
    "DH": "一壘/指定",
    "3B": "三壘",
}
CLUB_COLS = ["need_pos", "top_war_c", "queue_above", "queue_total", "club_war_c"]


def add_position(m: pd.DataFrame, data: Path) -> pd.DataFrame:
    """守位（FanGraphs 的 positionDB，如 '2B/SS'）→ 五個 0/1：有列出該守位就算 1。守備是成績單沒有、球探看得到的東西。"""
    pos = pd.read_parquet(
        data / "minors_bat_clean.parquet", columns=["key_mlbam", "season", "positionDB"]
    ).drop_duplicates(["key_mlbam", "season"])
    m = m.merge(pos, on=["key_mlbam", "season"], how="left")
    p = m["positionDB"].fillna("").astype(str)
    m["C"] = p.str.contains("C").astype(float)
    m["SS2B"] = (p.str.contains("SS") | p.str.contains("2B")).astype(float)
    m["OF"] = p.str.contains("OF").astype(float)
    m["1BDH"] = (p.str.contains("1B") | p.str.contains("DH")).astype(float)
    m["3B"] = p.str.contains("3B").astype(float)
    m["n_positions"] = p.str.count("/").astype(float) + (p != "").astype(float)  # "2B/SS" = 2，"OF" = 1，沒資料 = 0
    m["DH_only"] = (p == "DH").astype(float)
    m["pos_group"] = np.select(
        [
            p.str.contains("C"),
            p.str.contains("SS") | p.str.contains("2B"),
            p.str.contains("OF"),
            p.str.contains("1B") | p.str.contains("DH"),
            p.str.contains("3B"),
        ],
        ["捕手", "游擊/二壘", "外野", "一壘/指定", "三壘"],
        default="無資料",
    )
    return m


def add_club_strength(m: pd.DataFrame, data: Path) -> pd.DataFrame:
    """母隊強度：大聯盟母隊當季與隔季的團隊 WAR（打者 + 投手加總，約等於勝場 − 48）。
    弱隊空位多、重建中會提早叫人上來；強隊卡位。這是球隊的狀況，不是球員的能力。多球團的人季（'ARI/CIN'）取第一個。"""
    b = pd.read_parquet(data / "fg_batting.parquet", columns=["Team", "Season", "WAR"])
    pit = pd.read_parquet(data / "fg_pitching.parquet", columns=["Team", "Season", "WAR"])
    club = pd.concat([b, pit]).groupby(["Team", "Season"])["WAR"].sum().rename("club_war").reset_index()
    org = m["org"].fillna("").astype(str).str.split("/").str[0].replace({"ATH": "OAK"})
    key_t = pd.DataFrame({"Team": org, "Season": m["season"]})
    key_t1 = pd.DataFrame({"Team": org, "Season": m["season"] + 1})
    m = m.copy()
    m["club_war_t"] = key_t.merge(club, on=["Team", "Season"], how="left")["club_war"].to_numpy()
    m["club_war_t1"] = key_t1.merge(club, on=["Team", "Season"], how="left")["club_war"].to_numpy()
    return m


def load_mlb_positions(data: Path) -> pd.DataFrame:
    """大聯盟每人每季：Team、守位群、Age、PA、WAR。優先 fg_positions.parquet，沒有就看 fg_batting 有沒有 position 欄。"""
    fp = data / "fg_positions.parquet"
    if fp.exists():
        fg = pd.read_parquet(fp)
    else:
        fg = pd.read_parquet(data / "fg_batting.parquet")
        if "position" not in fg.columns:
            raise SystemExit("沒有守位欄：先跑 python src/fetch_fg_positions.py 2006 2025 data/")
    fg = fg[fg["Team"].astype(str).str.replace(" ", "") != "---"].copy()  # 多隊球員 FanGraphs 寫 '- - -'
    fg["Team"] = fg["Team"].replace({"ATH": "OAK"})
    prim = fg["position"].fillna("").astype(str).str.split("/").str[0].str.upper()
    fg["posg"] = prim.map(POS_GROUP)
    return fg.dropna(subset=["posg", "key_mlbam"])


def club_position_table(fg: pd.DataFrame) -> pd.DataFrame:
    """每隊每季每守位群：WAR 加總、打席加總、主力（打席最多的人）的 WAR 與年齡。"""
    t = fg.groupby(["Team", "Season", "posg"]).agg(pos_war=("WAR", "sum"), pos_pa=("PA", "sum")).reset_index()
    top = fg.sort_values("PA", ascending=False).drop_duplicates(["Team", "Season", "posg"])[
        ["Team", "Season", "posg", "WAR", "Age"]
    ]
    return t.merge(top.rename(columns={"WAR": "top_war", "Age": "top_age"}), on=["Team", "Season", "posg"], how="left")


def zscore(x: pd.Series) -> pd.Series:
    """標準化，缺值補 0（= 平均）。"""
    z = (x - x.mean()) / (x.std() + 1e-9)
    return z.fillna(0.0).clip(-4, 4)


def add_club_features(m: pd.DataFrame, data: Path, season: int) -> pd.DataFrame:
    """把五個球團層變數接到每個人季；m 必須已經過濾到單一球季。"""
    m = add_position(m, data)
    m = add_club_strength(m, data)
    cp = club_position_table(load_mlb_positions(data))
    club = m["org"].fillna("").astype(str).str.split("/").str[0].replace({"ATH": "OAK"})
    key = pd.DataFrame({"Team": club, "Season": season, "posg": m["pos_group"]})
    a = key.merge(cp, on=["Team", "Season", "posg"], how="left")
    m["pos_war_t"] = a["pos_war"].to_numpy()
    m["top_war_t"] = a["top_war"].to_numpy()
    m["top_age_t"] = a["top_age"].to_numpy()
    m["pos_known"] = m["pos_war_t"].notna().astype(float)
    lvl = m["level_rank_main"].to_numpy()
    above = np.zeros(len(m))
    total = np.zeros(len(m))
    for _, idx in m.groupby([club, m["pos_group"]]).indices.items():
        l = lvl[idx]
        above[idx] = (l[None, :] > l[:, None]).sum(1)
        total[idx] = len(idx) - 1
    m["queue_above_raw"] = above
    m["queue_total_raw"] = total
    m["need_pos"] = zscore(-m["pos_war_t"])
    m["top_war_c"] = zscore(m["top_war_t"])
    m["queue_above"] = zscore(m["queue_above_raw"])
    m["queue_total"] = zscore(m["queue_total_raw"])
    m["club_war_c"] = zscore(m["club_war_t"])
    return m
