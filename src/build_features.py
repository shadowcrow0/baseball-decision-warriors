"""把選秀、FanGraphs 逐季、Statcast 逐球接成一張「一人一列」的建模表。

    python src/build_features.py data/

讀取：
    data/draft.parquet        fetch_draft.py（B-Ref）
    data/fg_batting.parquet / fg_pitching.parquet / ids.parquet   fetch_war.py
    data/statcast_*.parquet   fetch_statcast.py（可多季，沒有就跳過 Statcast 欄位）
輸出：
    data/model_table.parquet

欄位：
    順位與背景       year, round, pick_overall, level(HS/4Yr/JC), position, bonus, team
    結果變數         bwar_career（B-Ref）、fwar_first6（登上一軍前 6 季 fWAR 合計，FanGraphs）
    第一季 FanGraphs  fg1_wOBA / fg1_FIP / fg1_PA / fg1_IP / fg1_age
    第一季 Statcast   sc1_ev_mean（擊球初速）、sc1_la_mean（擊球角度）、sc1_hardhit（≥95 mph 比例）、
                     sc1_whiff（揮空率）、sc1_velo（投手球速）、sc1_spin（轉速）、sc1_n_pitches

B-Ref 選秀表沒有球員 id，用「姓名 + 首次出賽年」對 Chadwick 取 key_mlbam / key_fangraphs；
對不上的保留在表裡但 id 為空（通常是沒登上一軍的人）。
"""

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").replace("'", "").replace("-", " ").strip()


def attach_ids(draft: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    ids = ids.copy()
    ids["key"] = ids["name_first"].map(norm) + " " + ids["name_last"].map(norm)
    ids = ids.sort_values("mlb_played_first")
    draft = draft.copy()
    draft["key"] = draft["name"].map(norm)
    # 只配登上一軍的人；同名者取首次出賽年 ≥ 選秀年且最接近的
    cand = draft[draft["reached_mlb"]].merge(ids, on="key", how="left")
    cand = cand[cand["mlb_played_first"].isna() | (cand["mlb_played_first"] >= cand["year"])]
    cand = cand.sort_values("mlb_played_first").drop_duplicates(subset=["year", "pick_overall"], keep="first")
    keep = ["year", "pick_overall", "key_mlbam", "key_fangraphs", "key_bbref", "mlb_played_first"]
    return draft.merge(cand[keep], on=["year", "pick_overall"], how="left")


def fg_features(df: pd.DataFrame, fg_b: pd.DataFrame, fg_p: pd.DataFrame) -> pd.DataFrame:
    b = fg_b.rename(columns={"IDfg": "key_fangraphs"})
    p = fg_p.rename(columns={"IDfg": "key_fangraphs"})
    rows = []
    for fid in df["key_fangraphs"].dropna().unique():
        sb = b[b["key_fangraphs"] == fid].sort_values("Season")
        sp = p[p["key_fangraphs"] == fid].sort_values("Season")
        seasons = sorted(set(sb["Season"]) | set(sp["Season"]))
        if not seasons:
            continue
        first = seasons[0]
        win = [s for s in seasons if s < first + 6]
        war = sb[sb["Season"].isin(win)]["WAR"].sum() + sp[sp["Season"].isin(win)]["WAR"].sum()
        b1, p1 = sb[sb["Season"] == first], sp[sp["Season"] == first]
        rows.append(
            {
                "key_fangraphs": fid,
                "fg_first_season": first,
                "fwar_first6": war,
                "fg1_age": (b1["Age"].iloc[0] if len(b1) else p1["Age"].iloc[0]),
                "fg1_PA": b1["PA"].sum() if len(b1) else 0,
                "fg1_wOBA": b1["wOBA"].mean() if len(b1) else np.nan,
                "fg1_IP": p1["IP"].sum() if len(p1) else 0,
                "fg1_FIP": p1["FIP"].mean() if len(p1) else np.nan,
            }
        )
    return df.merge(pd.DataFrame(rows), on="key_fangraphs", how="left")


def statcast_features(df: pd.DataFrame, data: Path) -> pd.DataFrame:
    files = sorted(f for f in data.glob("statcast_*.parquet") if f.is_file())
    if not files:
        print("沒有 Statcast 檔，跳過 sc1_* 欄位")
        return df
    cols = [
        "game_date",
        "batter",
        "pitcher",
        "launch_speed",
        "launch_angle",
        "release_speed",
        "release_spin_rate",
        "description",
    ]
    sc = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    sc["season"] = pd.to_datetime(sc["game_date"]).dt.year
    sc["swing"] = sc["description"].isin(
        ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt"]
    )
    sc["whiff"] = sc["description"].isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"])
    sc["hardhit"] = sc["launch_speed"] >= 95

    bat = (
        sc.groupby(["batter", "season"])
        .agg(
            sc1_ev_mean=("launch_speed", "mean"),
            sc1_la_mean=("launch_angle", "mean"),
            sc1_hardhit=("hardhit", "mean"),
            sc1_n_pitches=("batter", "size"),
            _sw=("swing", "sum"),
            _wh=("whiff", "sum"),
        )
        .reset_index()
    )
    bat["sc1_whiff"] = bat["_wh"] / bat["_sw"].replace(0, np.nan)
    bat = bat.drop(columns=["_sw", "_wh"]).rename(columns={"batter": "key_mlbam"})

    pit = (
        sc.groupby(["pitcher", "season"])
        .agg(
            sc1_velo=("release_speed", "mean"),
            sc1_spin=("release_spin_rate", "mean"),
            sc1_n_pitches_p=("pitcher", "size"),
            _sw=("swing", "sum"),
            _wh=("whiff", "sum"),
        )
        .reset_index()
    )
    pit["sc1_whiff_p"] = pit["_wh"] / pit["_sw"].replace(0, np.nan)
    pit = pit.drop(columns=["_sw", "_wh"]).rename(columns={"pitcher": "key_mlbam"})

    # 取每人在 Statcast 裡的第一季
    bat = bat.sort_values("season").drop_duplicates("key_mlbam").drop(columns="season")
    pit = pit.sort_values("season").drop_duplicates("key_mlbam").drop(columns="season")
    df = df.merge(bat, on="key_mlbam", how="left")
    df = df.merge(pit, on="key_mlbam", how="left")
    return df


def main(data_dir: str):
    data = Path(data_dir)
    draft = pd.read_parquet(data / "draft.parquet")
    ids = pd.read_parquet(data / "ids.parquet")
    df = attach_ids(draft, ids)
    matched = df["key_mlbam"].notna().sum()
    print(f"選秀 {len(df):,} 人，登上一軍 {df['reached_mlb'].sum():,}，配到 id {matched:,}")
    df = fg_features(df, pd.read_parquet(data / "fg_batting.parquet"), pd.read_parquet(data / "fg_pitching.parquet"))
    df = statcast_features(df, data)
    out = data / "model_table.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"{len(df):,} 列 × {df.shape[1]} 欄 -> {out}")
    print(df.filter(regex="^(fwar|fg1|sc1)").describe().T[["count", "mean"]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
