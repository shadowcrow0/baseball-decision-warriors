"""打擊是長是短、球打到外野還是留在內野、會不會點，有沒有影響球隊留不留人？四把尺之外還剩多少。

    python src/hit_profile.py outputs/ data/ --promoted promoted_f4sed_linear_p2
    python src/hit_profile.py --selftest        （用合成資料跑一遍整條流程，不需要 data/）

問題（領隊的話）：「打擊分成長的跟短打，也就是外野跟內野之間的差異，有沒有影響這個人被留下來」。
打擊樣貌 = 長打比例、一壘安打率、滾地率、平飛率、外野飛球比例、內野高飛比例、全壘打／飛球、犧牲觸擊率（點）。
FanGraphs 小聯盟 API 一人季只有 65 欄，沒有逐球資料，也沒有內野安打、觸擊安打；小聯盟逐球只在 MLB StatsAPI（本案不用）
與 Statcast 的 3A（2023 起）。所以「點」只有犧牲觸擊，「接殺」只能用內野高飛、外野飛球逼近。
基底 = 四種能力 + 球探評價 + 相對年齡 + 打席（log）+ 層級虛擬變數，每季各自配一條邏輯迴歸（WAR 用線性）。
基底 + 一個打擊樣貌變數：逐季係數與 z、幾季正幾季負、反變異數加權的合併係數；
前三季估、當季測的樣本外 log-loss 少多少（WAR 段是均方誤差），跟 variable_hunt 同一套。
結果：留不留（隔季還在小聯盟資料裡，或直接上大聯盟）、留在同一隊；對照用：隔年升不升、六季內登上、登上者六季 WAR。
輸出 outputs/hit_profile_<promoted tag>.md。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decision_style import LEVELS, fit_logit, logit_predict, log_loss
from bf import bf_str, log10_bf10
from misplaced_profile import logistic_irls

BASE = ["eta_contact", "eta_power", "eta_speed", "eta_discipline", "zeta_scout", "age_rel", "log_pa"]
# 變數名、中文、分子加的欄、分子減的欄、分母
PROFILE = [
    ("xbh_h", "長打比例 XBH/H", ["XBH"], [], "H"),
    ("s1_ab", "一壘安打率 1B/AB", ["H"], ["XBH"], "AB"),
    ("gb_bip", "滾地率 GB/BIP", ["GB"], [], "BIP"),
    ("ld_bip", "平飛率 LD/BIP", ["LD"], [], "BIP"),
    ("ofb_bip", "外野飛球比例 (FB−IFFB)/BIP", ["FB"], ["IFFB"], "BIP"),
    ("iffb_fb", "內野高飛比例 IFFB/FB（幾乎都是接殺）", ["IFFB"], [], "FB"),
    ("hr_fb", "全壘打／飛球 HR/FB", ["HR"], [], "FB"),
    ("sh_pa", "犧牲觸擊率 SH/PA（點）", ["SH"], [], "PA"),
]
LABELS = {name: label for name, label, _, _, _ in PROFILE}
OUTCOMES = [
    ("y_keep", "被留下來：隔季還在（或直接上大聯盟）"),
    ("y_same", "留在同一隊：隔季還在而且母隊相同"),
    ("y_prom", "對照 · 球隊：隔年升不升"),
    ("y_reach", "對照 · 真相：六季內登上"),
    ("y_war", "對照 · 真相：登上者六季 WAR（asinh，線性迴歸、均方誤差）"),
]
READING = """讀法：每個係數都是「四種能力、球探評價、年齡、打席、層級都一樣的兩個人，打擊樣貌差一個標準差（層級內），結果差多少」。係數 > 0 的意思：
- 被留下來：那種打法的人，隔季比較會還在職業體系裡（球隊留他）。
- 留在同一隊：比較會留在原球團（沒被釋出、也沒換隊）。
- 隔年升不升：球隊比較願意把他升上去。
- 六季內登上：比較會真的打上大聯盟（真相，不是球隊的看法）。
- 登上者六季 WAR：登上之後打得比較好（線性係數，單位 asinh(WAR)）。
邏輯迴歸係數是 log-odds：0.1 大約是五成變五成二。證據強度看 **BF10**（Savage–Dickey，先驗 N(0, 0.5²)）：> 10 強、3–10 中等、1/3–3 分不出來、< 1/3 支持零；「幾季正 / 幾季負」看方向穩不穩；
「平均每季變準」= 前三季估、當季測，加了這個變數少掉的 log-loss（WAR 是均方誤差），> 0 才是四把尺之外真的多帶資訊。
滾地、平飛、外野飛球、內野高飛四個比例加起來是 1，所以它們彼此不獨立，一次只放一個。分母不到 30 的人季，該變數補 0（= 層級平均）。
犧牲觸擊率已經在主模型裡（升級段 17 季全負），這裡是在四把尺之外再看一次它對留不留的作用。"""


def add_retention(m: pd.DataFrame) -> pd.DataFrame:
    """留不留：下一個有資料的球季（2019 的下一季是 2021）這個人還在小聯盟資料裡，或 next_status 是 mlb（直接上去了，當然算留下）。
    留在同一隊：隔季還在、而且母隊有重疊（org 像 'NYY' 或 'NYY/WSN'）；隔季不在但 next_status 是 mlb 的人看不到球隊，留空。
    最後一個球季沒有隔季，兩個結果都留空。"""
    seasons = sorted(m["season"].unique())
    nxt = dict(zip(seasons[:-1], seasons[1:]))
    back = {v: k for k, v in nxt.items()}
    key = m[["key_mlbam", "season", "org"]].drop_duplicates(["key_mlbam", "season"]).copy()
    key["season"] = key["season"].map(back)  # 把 t+1 的列對回 t
    key = key.dropna(subset=["season"]).astype({"season": int}).rename(columns={"org": "org_next"})
    key["present_next"] = 1.0
    m = m.merge(key, on=["key_mlbam", "season"], how="left")
    has_next = m["season"].isin(nxt)
    present = m["present_next"].fillna(0.0) == 1.0
    mlb = m["next_status"].astype(str) == "mlb"
    orgs = [set(str(x).split("/")) - {"", "nan", "None"} for x in m["org"].fillna("")]
    orgs_next = [set(str(x).split("/")) - {"", "nan", "None"} for x in m["org_next"].fillna("")]
    same = np.array([bool(a & b) for a, b in zip(orgs, orgs_next)])
    m["y_keep"] = np.where(has_next, (present | mlb).astype(float), np.nan)
    m["y_same"] = np.where(has_next & ~(mlb & ~present), (present & same).astype(float), np.nan)
    return m.drop(columns=["org_next", "present_next"])


def add_profile(m: pd.DataFrame) -> tuple:
    """算打擊樣貌的比率；分母 < 30 留空（之後補 0 = 層級平均）。建模表沒有的計數欄就跳過那個變數。"""
    names = []
    for name, _, plus, minus, den in PROFILE:
        if any(c not in m.columns for c in plus + minus + [den]):
            continue
        num = m[plus].sum(axis=1) - m[minus].sum(axis=1)
        m[name] = num / m[den].where(m[den] >= 30)
        names.append(name)
    return m, names


def load(out: Path, data: Path, tag_p: str, min_pa: int = 100) -> tuple:
    """每人季：基底、打擊樣貌、五個結果。基底與打擊樣貌都在球季 × 層級內標準化（跟 variable_hunt 一樣）。"""
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_retention(m)
    m, names = add_profile(m)
    frames = []
    for p in sorted(out.glob(f"20*/{tag_p}/eta_M1.parquet")):
        season = int(p.parts[-3])
        e = pd.read_parquet(p)[["key_mlbam", "season"] + BASE[:5]]
        frames.append(m[m["season"] == season].merge(e, on=["key_mlbam", "season"], how="inner"))
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["PA"] >= min_pa) & d["level_main"].isin(LEVELS)].copy()
    d["zeta_scout"] = d["zeta_scout"].fillna(0.0)
    d["log_pa"] = np.log(d["PA"])
    for c in BASE + names:
        d[c] = d[c].astype(float)
        d[c] = d.groupby(["season", "level_main"])[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
    for lv in LEVELS[1:]:
        d["lvl_" + lv] = (d["level_main"] == lv).astype(float)
    d["y_prom"] = d["promoted"].astype(float)
    d["y_reach"] = np.where(d["war_observable"].astype(bool), d["reached_fwd"].astype(float), np.nan)
    d["y_war"] = np.where(
        d["war_observable"].astype(bool) & d["reached_fwd"].astype(bool), np.arcsinh(d["war_fwd"].astype(float)), np.nan
    )
    return d, names


def design(s: pd.DataFrame, features: list) -> np.ndarray:
    return np.column_stack([np.ones(len(s))] + [s[f].to_numpy(dtype=float) for f in features])


def coef_se(s: pd.DataFrame, outcome: str, features: list) -> tuple:
    """一季一群人：結果對 features 迴歸（0/1 用邏輯迴歸，WAR 用線性），回傳最後一個特徵的係數與標準誤。人太少回傳 nan。"""
    s = s[s[outcome].notna()]
    if len(s) < 60:
        return np.nan, np.nan
    y = s[outcome].to_numpy(dtype=float)
    X = design(s, features)
    if outcome == "y_war":
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ b
        cov = resid @ resid / max(len(y) - X.shape[1], 1) * np.linalg.inv(X.T @ X + 1e-6 * np.eye(X.shape[1]))
        return float(b[-1]), float(np.sqrt(cov[-1, -1]))
    if y.sum() < 10 or (1 - y).sum() < 10:
        return np.nan, np.nan
    b, se = logistic_irls(X, y)
    return float(b[-1]), float(se[-1])


def oos_loss(d: pd.DataFrame, season: int, train: list, features: list, outcome: str) -> float:
    """前三季估、第 season 季測：邏輯迴歸的 log-loss，或 WAR 線性迴歸的均方誤差。"""
    tr = d[d["season"].isin(train) & d[outcome].notna()]
    te = d[(d["season"] == season) & d[outcome].notna()]
    if len(tr) < 60 or len(te) < 10:
        return np.nan
    ya, yb = tr[outcome].to_numpy(dtype=float), te[outcome].to_numpy(dtype=float)
    if outcome == "y_war":
        Xa, Xb = design(tr, features), design(te, features)
        coef = np.linalg.solve(Xa.T @ Xa + 1e-2 * np.eye(Xa.shape[1]), Xa.T @ ya)
        return float(((yb - Xb @ coef) ** 2).mean())
    if ya.sum() < 10 or (1 - ya).sum() < 10:
        return np.nan
    coef = fit_logit(tr[features].to_numpy(dtype=float), ya)
    return log_loss(yb, logit_predict(coef, te[features].to_numpy(dtype=float)))


def one_table(d: pd.DataFrame, names: list, outcome: str, labels: dict | None = None) -> pd.DataFrame:
    """一個結果、一群人：每個打擊樣貌變數的逐季係數（反變異數加權合併）、符號、樣本外變準。_se 是合併係數的標準誤。
    labels 是變數名 → 中文；不給就用這支的 LABELS（hit_profile_statcast 會給自己的）。"""
    labels = LABELS if labels is None else labels
    seasons = sorted(d["season"].unique())
    base = BASE + [c for c in d.columns if c.startswith("lvl_")]
    trains = {s: [t for t in range(s - 3, s) if t in seasons] for s in seasons}
    base_loss = {s: oos_loss(d, s, trains[s], base, outcome) if trains[s] else np.nan for s in seasons}
    rows = []
    for v in names:
        bs, ses, gains = [], [], []
        for s in seasons:
            b, se = coef_se(d[d["season"] == s], outcome, base + [v])
            if np.isfinite(b) and np.isfinite(se) and se > 0:
                bs.append(b)
                ses.append(se)
            if np.isfinite(base_loss[s]):
                loss = oos_loss(d, s, trains[s], base + [v], outcome)
                if np.isfinite(loss):
                    gains.append(base_loss[s] - loss)
        if not bs:
            continue
        w = 1.0 / np.array(ses) ** 2
        pooled, se_p = float((np.array(bs) * w).sum() / w.sum()), float(np.sqrt(1.0 / w.sum()))
        rows.append(
            {
                "打擊樣貌": labels[v],
                "合併係數": round(pooled, 3),
                "z": round(pooled / se_p, 1),
                "BF10": bf_str(log10_bf10(pooled, se_p)),
                "幾季正 / 幾季負": f"{sum(b > 0 for b in bs)} / {sum(b < 0 for b in bs)}",
                "平均每季變準": round(float(np.mean(gains)), 4) if gains else np.nan,
                "變準的季數": f"{sum(g > 0 for g in gains)} / {len(gains)}",
                "季數": len(bs),
                "_se": se_p,
            }
        )
    return pd.DataFrame(rows)


def main(out_dir: str, data_dir: str, tag_p: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    d, names = load(out, data, tag_p)
    skipped = [LABELS[n] for n in LABELS if n not in names]
    md = [
        f"# 打擊是長是短、外野還是內野、會不會點，影響被留下來嗎 · {tag_p}\n",
        READING,
        "",
        f"樣本：{len(d):,} 人季，{int(d['season'].min())}–{int(d['season'].max())}，打席 ≥ 100，層級 {LEVELS[0]}–{LEVELS[-1]}。"
        f"算得出來的打擊樣貌：{'、'.join(LABELS[n] for n in names)}。"
        + (f"建模表沒有、跳過：{'、'.join(skipped)}。" if skipped else "")
        + "小聯盟沒有逐球資料，內野安打、觸擊安打、接殺都不在 FanGraphs 的 65 欄裡。",
        "",
    ]
    for outcome, label in OUTCOMES:
        ok = d[outcome].notna()
        if ok.sum() == 0:
            continue
        tab = one_table(d, names, outcome)
        rate = (
            f"正例 {d.loc[ok, outcome].mean():.1%}"
            if outcome != "y_war"
            else f"平均 asinh(WAR) {d.loc[ok, outcome].mean():.2f}"
        )
        md += [f"## {label}（{int(ok.sum()):,} 人季，{rate}）\n", tab.drop(columns="_se").to_markdown(index=False), ""]
    text = "\n".join(md)
    path = out / f"hit_profile_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成一份小資料跑整條流程：長打比例高的人比較會被留下、滾地多的人比較不會，看表有沒有抓到方向。"""
    rng = np.random.default_rng(0)
    root = Path(tempfile.mkdtemp())
    data, out = root / "data", root / "outputs"
    data.mkdir()
    seasons, n, orgs = [2011, 2012, 2013, 2014, 2015, 2016, 2017], 700, ["NYY", "BOS", "LAD", "SEA", "CHC"]
    etas = ["eta_" + f for f in ("contact", "power", "speed", "discipline")]
    frames, alive = [], pd.DataFrame({"key_mlbam": pd.Series(dtype=int), "org": pd.Series(dtype=str)})
    for s in seasons:
        new = pd.DataFrame(
            {"key_mlbam": range(s * 10000, s * 10000 + n - len(alive)), "org": rng.choice(orgs, n - len(alive))}
        )
        g = pd.concat([alive, new], ignore_index=True)
        swap = rng.random(n) < 0.15
        g.loc[swap, "org"] = rng.choice(orgs, int(swap.sum()))
        g["season"] = s
        g["PA"] = rng.integers(60, 600, n)
        g["AB"] = (g["PA"] * 0.88).astype(int)
        g["SO"], g["BB"], sf = rng.binomial(g["PA"], 0.2), rng.binomial(g["PA"], 0.08), rng.binomial(g["PA"], 0.01)
        g["H"] = rng.binomial(g["AB"], 0.26)
        g["XBH"] = rng.binomial(g["H"], rng.uniform(0.15, 0.45, n))
        g["HR"] = rng.binomial(g["XBH"], 0.3)
        g["BIP"] = np.maximum(g["AB"] - g["SO"] + sf, 0)
        g["GB"] = rng.binomial(g["BIP"], rng.uniform(0.3, 0.6, n))
        g["FB"] = rng.binomial(g["BIP"] - g["GB"], 0.65)
        g["LD"] = g["BIP"] - g["GB"] - g["FB"]
        g["IFFB"], g["SH"] = rng.binomial(g["FB"], 0.1), rng.binomial(g["PA"], 0.01)
        g["HIP"], g["BIPH"] = g["H"] - g["HR"], np.maximum(g["BIP"] - g["HR"], 0)
        g["level_main"] = rng.choice(LEVELS, n)
        g["age_rel"], g["promoted"] = rng.normal(size=n), rng.random(n) < 0.3
        g["next_status"] = np.where(rng.random(n) < 0.03, "mlb", "same")
        g["war_fwd"], g["reached_fwd"], g["war_observable"] = np.abs(rng.normal(size=n)), rng.random(n) < 0.1, s <= 2013
        for c in etas:
            g[c] = rng.normal(size=n)
        long_z = (g["XBH"] / g["H"].clip(lower=1) - 0.3) / 0.1 - (g["GB"] / g["BIP"].clip(lower=1) - 0.45) / 0.1
        keep = rng.random(n) < 1 / (1 + np.exp(-(0.6 + 0.5 * long_z + 0.3 * g["eta_power"])))
        frames.append(g)
        alive = g.loc[keep, ["key_mlbam", "org"]]
    m = pd.concat(frames, ignore_index=True)
    m.drop(columns=etas).to_parquet(data / "model_bat.parquet", index=False)
    for s, g in m.groupby("season"):
        (out / str(s) / "promoted_test").mkdir(parents=True)
        g[["key_mlbam", "season"] + etas].assign(zeta_scout=np.nan).to_parquet(
            out / str(s) / "promoted_test" / "eta_M1.parquet", index=False
        )
    text = main(str(out), str(data), "promoted_test")
    d, names = load(out, data, "promoted_test")
    tab = one_table(d, names, "y_keep").set_index("打擊樣貌")
    assert d["y_keep"].notna().sum() > 0 and d.loc[d["season"] == 2017, "y_keep"].isna().all(), "最後一季應留空"
    assert tab.loc[LABELS["xbh_h"], "合併係數"] > 0 and tab.loc[LABELS["gb_bip"], "合併係數"] < 0, "方向沒抓到"
    assert "sh_pa" in names and "hr_fb" in names, "新變數沒算出來"
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="打擊是長是短、外野還是內野、會不會點，有沒有影響被留下來：四把尺之外的係數與樣本外增量"
    )
    p.add_argument("out_dir", nargs="?", default="outputs", help="模型輸出資料夾（含 <season>/<tag>/eta_M1.parquet）")
    p.add_argument("data_dir", nargs="?", default="data", help="資料夾（含 model_bat.parquet）")
    p.add_argument("--promoted", default=None, help="升級模型的 tag，例如 promoted_f4sed_linear_p2")
    p.add_argument("--selftest", action="store_true", help="用合成資料跑一遍，不需要 data/")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.promoted is None:
        p.error("--promoted 是必要的（或用 --selftest）")
    else:
        main(a.out_dir, a.data_dir, a.promoted)
