"""單一球季、打者：四種能力的測量模型 + 結構模型（升級，或 WAR 兩段），對應 docs/model_spec.md。

    python src/fit_year.py data/ 2015 outputs/ [選項]      （python src/fit_year.py --help 看全部選項）

讀這支程式的順序（由上到下，每一節一個工作）：
    1. 規格：哪些指標、歸哪種能力、符號錨、交叉負荷。這是「模型長什麼樣」的設定，改模型先改這裡。
    2. 讀資料：建模表讀進來、算衍生欄位（到首登的年數、跨季進步幅度）。
    3. 設計矩陣：共變數怎麼變成數字矩陣（正交多項式、0/1）。
    4. 建模型：三個函式各蓋一塊（球探先驗、四種能力的測量、結構模型的線性預測式），build() 把它們組起來。
    5. 取樣：跑 NUTS、算每個觀測的 log-likelihood、存檔。
    6. 診斷：鏡像對齊、樹深、尺度表、r-hat。
    7. 彙整：負荷量表、結構係數表、LOO/WAIC 比較、每人季分數匯出、summary.md。
    8. main：把 5–7 串起來。
    9. 命令列。

每次同時擬合三個模型，放在同一把尺上比：
    M0  共變數（+ 球探先驗），結果不連到四種能力
    M1  M0 + 四種能力
    Mh  共變數 + wOBA 多項式（人類設計的綜合指標當基準）
輸出 outputs/<season>/<tag>/：idata_*.nc（後驗）、summary.md、eta_M1.parquet（每人季分數）、struct_M1.csv、lambda_M1.csv、org_M1.csv（球團表）
"""

import argparse
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# =====================================================================================
# 1. 規格
# =====================================================================================
# (名稱, 分子, 分母, 似然, 所屬 factor)
#   factor 0 = 接觸與選球；1 = 力量與擊球品質；2 = 速度
INDICATORS = [
    ("AVG", "H", "AB", "binom", 0),
    ("OBP", "OB", "PA", "binom", 0),
    ("K%", "SO", "PA", "binom", 0),
    ("whiff%", "whiff", "Pitches", "binom", 0),
    ("XBH/H", "XBH", "H", "binom", 1),
    ("HR/AB", "HR", "AB", "negbin", 1),
    ("Pull%", "PULL", "BIP", "binom", 1),
    ("GB%", "GB", "BIP", "binom", 1),
    ("SB%", "SB", "SBA", "binom", 2),
    ("SBA/OBT", "SBA", "OBT", "binom", 2),
    ("3B/H", "3B", "H", "binom", 2),
    ("GDP/AB", "GDP", "AB", "binom", 2),
    ("BB%", "BB", "PA", "binom", 3),
    ("Ball%", "BALLS", "Pitches", "binom", 3),
    ("P/PA", "Pitches", "PA", "negbin", 3),
    # 第五把尺「擊球品質」（只在 --factors 5 啟用）：平飛球率、內野高飛率（負）、BABIP。構念事先定義：打到球之後球有多強勁、多容易變安打
    ("LD%", "LD", "BIP", "binom", 4),
    ("IFFB%", "IFFB", "FB", "binom", 4),
    ("BABIP", "HIP", "BIPH", "binom", 4),
]
#   factor 0 = 接觸；1 = 力量與擊球品質；2 = 速度；3 = 選球
# 每個 factor 的符號錨：(指標, 符號)
ANCHORS = {0: ("K%", -1), 1: ("HR/AB", +1), 2: ("SBA/OBT", +1), 3: ("BB%", +1), 4: ("BABIP", +1)}
FACTOR_NAMES = {0: "contact", 1: "power", 2: "speed", 3: "discipline", 4: "quality"}
# --statcast：3A 逐球（Savant 2023 起）算出來的過程量當指標，載到同樣的構念上。選的是文獻說最穩、最會預測隔季的
# （docs/advanced_predictors.md）：區內接觸率（打得到球）、EV90 與 barrel 率與拉打飛球率（打得遠）、追打率（選得到球，負）、
# 最弱一成球的初速與接觸後 xwOBA（擊球品質）。速度沒有公開的 Statcast 量，仍用成績單的四個。
# kind "normal"：連續量，層級內標準化後進常態似然，精度隨進場球數放大（sigma / sqrt(進場球 / 100)）。
# 區內接觸率、追打率要有進壘點欄（2026-10-23 之後抓的資料才有），沒有就自動不進模型。
STATCAST_INDICATORS = [
    ("Z-Con%", "sc_zcon", "sc_zswing", "binom", 0),
    ("EV90", "sc_ev90", "sc_bip", "normal", 1),
    ("Barrel%", "sc_barrel", "sc_bip", "binom", 1),
    ("PullAir%", "sc_pullair", "sc_bip", "binom", 1),
    ("O-Swing%", "sc_chase", "sc_ozpitch", "binom", 3),
    ("EV10", "sc_ev10", "sc_bip", "normal", 4),
    ("xwOBAcon", "sc_xwobacon", "sc_bip", "normal", 4),
]
# --statcast only（只用 3A、只用進階量）時換一組錨：每把尺由一個進階量定方向
STATCAST_ANCHORS = {0: ("Z-Con%", +1), 1: ("EV90", +1), 2: ("SBA/OBT", +1), 3: ("O-Swing%", -1), 4: ("xwOBAcon", +1)}


def anchors_for(statcast: str) -> dict:
    return STATCAST_ANCHORS if statcast == "only" else ANCHORS


# --statcast 的五種模式。要問「進階量自己組成的尺」跟「成績單組成的尺」哪個比較會預測，兩邊樣本要一樣：
#   none  成績單 15 個指標、全部人季（主模型）
#   add   成績單 + Statcast 7 個、全部人季（Statcast 只有 3A 人季有）
#   same  成績單 15 個指標、**只留有 Statcast 的 3A 人季**（跟 only 同一批人的基準）
#   only  Statcast 7 個 + 成績單的速度 4 個、只留有 Statcast 的 3A 人季（進階量自己組尺）
#   both  成績單 + Statcast、只留有 Statcast 的 3A 人季
# same / only / both 三個 fit 的升級段 LOO 可以直接比（同一批人、同一個結果），compare_fits.py 會算。
STATCAST_MODES = ["none", "add", "same", "only", "both"]
STATCAST_LETTER = {"none": "", "add": "S", "same": "N", "only": "O", "both": "B"}
# 交叉負荷（--crossload）：指標 -> 額外載入的 factor。三振/揮空也屬力量（力量型三振多）
CROSSLOADS = {"K%": 1, "whiff%": 1}  # 試跑：K%、whiff% → 力量各 +0.22（力量型三振多）；P/PA → 接觸只有 −0.02，拿掉
# factor 數不足時，指標併到哪個 factor（None = 不用）：f2 速度不用、選球併入接觸；f3 選球併入接觸
FOLD = {2: {2: None, 3: 0, 4: None}, 3: {3: 0, 4: None}, 4: {4: None}}  # 四把尺以下不用擊球品質的三個指標
# 起始值的符號（棒球常識），讓四條鏈從同一個盆地出發；不改變後驗
INIT_SIGN = {
    "AVG": +1,
    "OBP": +1,
    "BB%": +1,
    "K%": -1,
    "whiff%": -1,
    "XBH/H": +1,
    "HR/AB": +1,
    "Pull%": +1,
    "GB%": -1,
    "SB%": +1,
    "SBA/OBT": +1,
    "3B/H": +1,
    "GDP/AB": -1,
    "Ball%": +1,
    "P/PA": +1,
    "LD%": +1,
    "IFFB%": -1,
    "BABIP": +1,
    "Z-Con%": +1,
    "EV90": +1,
    "Barrel%": +1,
    "PullAir%": +1,
    "O-Swing%": -1,
    "EV10": +1,
    "xwOBAcon": +1,
}
N_LEVELS = 7
LEVEL_LABELS = {0: "DSL", 1: "R/CPX", 2: "A-", 3: "A", 4: "A+", 5: "AA", 6: "AAA"}
SUMMARY_COLS = ["mean", "sd", "eti89_lb", "eti89_ub", "ess_bulk", "r_hat"]
COVARIATES = ["age_rel", "log_pick", "log_bonus_f", "log_pa"]  # 多項式共變數
LINEAR_COVS = ["sh_rate_f"]  # 只放一次項
SPLINE_KNOTS = (-1.0, 0.0, 1.0)
DEBUT_CENSOR = 7.5
# --direct：這些指標的原始比率（層級內標準化）直接進結構模型，跟四種能力並排；係數 = 該數字在能力之外還多說了什麼
DIRECT = ["SB%", "3B/H", "GDP/AB", "P/PA", "BB%"]  # 六季窗口：T ∈ {1..7} 為事件，7.5 = 右設限


def indicator_tag(name: str) -> str:
    """指標名稱變成合法的變數名：K% → Kpct，HR/AB → HR_AB。"""
    return name.replace("%", "pct").replace("/", "_")


# =====================================================================================
# 2. 讀資料
# =====================================================================================
def add_extra(full: pd.DataFrame) -> pd.DataFrame:
    """球探可能忽略的候選（--extra）：跟上一季比的進步幅度、年資、重複層級、跨層級、打席佔比。
    進步幅度 = 本季比率 − 上一季比率，在層級內標準化；沒有上一季的人為 0 並以 has_prev 標記。"""
    full = full.sort_values(["key_mlbam", "season"]).copy()
    pa = full["PA"].replace(0, np.nan)
    ab = full["AB"].replace(0, np.nan)
    full["r_k"] = full["SO"] / pa
    full["r_hr"] = full["HR"] / ab
    full["r_bb"] = full["BB"] / pa
    g = full.groupby("key_mlbam")
    prev_season = g["season"].shift(1)
    prev_lvl = g["level_rank_main"].shift(1)
    consecutive = (full["season"] - prev_season).isin([1, 2])  # 2020 停賽允許隔兩年
    full["has_prev"] = consecutive.fillna(False).astype(float)
    for c in ("r_k", "r_hr", "r_bb"):
        d = (full[c] - g[c].shift(1)).where(consecutive)
        z = d.groupby(full["level_rank_main"]).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        full["d_" + c[2:]] = z.fillna(0.0).clip(-4, 4)
    full["repeat_level"] = ((full["level_rank_main"] == prev_lvl) & consecutive).fillna(False).astype(float)
    full["multi_level"] = (full["n_levels"] > 1).astype(float)
    full["years_in"] = full["season_idx_in_career"].astype(float)
    return full


def load(data: Path, season: int, club: bool = False, statcast: str = "none") -> pd.DataFrame:
    m = pd.read_parquet(data / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    if club:  # --club：母隊該守位缺口、主力、農場塞車、母隊強弱
        from club_features import add_club_features

        m = add_club_features(m, data, season)
    if statcast != "none":  # --statcast：3A 逐球彙整成計數，接到該季的 3A 人季上；連續量在層級內標準化
        from hit_profile_statcast import load_pitches, statcast_counts

        sc = statcast_counts(load_pitches(data, season))
        m = m.merge(sc, on=["key_mlbam", "season", "level_main"], how="left")
        for c in ("sc_ev90", "sc_ev10", "sc_xwobacon"):
            if c in m.columns:
                m[c] = m.groupby("level_rank_main")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        have = int((m["sc_bip"].fillna(0) > 0).sum()) if "sc_bip" in m.columns else 0
        print(f"  Statcast：{have:,} 個人季有 3A 逐球", flush=True)
        if statcast in ("same", "only", "both"):
            if have == 0:
                raise SystemExit(f"--statcast {statcast} 需要該季的 3A 逐球（data/savant_minors_<season>.parquet）")
            m = m[m["sc_bip"].fillna(0) > 0].reset_index(drop=True)
            print(f"  --statcast {statcast}：樣本縮到有 3A 逐球的 {len(m):,} 個人季", flush=True)
    m["org_i"] = pd.factorize(m["org"].fillna("UNK"))[0]
    m["lvl"] = m["level_rank_main"].astype(int)
    # 共變數不能有缺值：缺值會讓 η 的平均值變 NaN，整個模型在起始點就算不出來（2008、2022 有幾個人沒有出生日期）。
    # 年齡缺的人當作該層級中位數（相對年齡 0）；其他連續共變數缺的也補 0（= 平均）。
    for col in ("age_rel", "log_pa", "sh_rate", "d_k", "d_hr", "d_bb", "years_in", "pa_share"):
        if col in m.columns and m[col].isna().any():
            n_missing = int(m[col].isna().sum())
            print(f"  共變數 {col} 缺 {n_missing} 個人季，補 0", flush=True)
            m[col] = m[col].fillna(0.0)
    m["log_bonus_f"] = m["log_bonus"].fillna(0.0)
    m["sh_rate_f"] = m["sh_rate"].fillna(0.0)
    # 到首登的年數 T = 首登年 − 本季 + 1（同季登上 = 1），六季窗口內沒登上 → 右設限，記為上界 7.5
    T = m["mlb_played_first"] - m["season"] + 1.0
    m["T_debut"] = np.where(m["reached_fwd"].to_numpy(dtype=bool), T, DEBUT_CENSOR)
    for name in DIRECT + ["AVG"]:  # 能力之外的細節：比率在層級內標準化，缺值為 0（AVG 給 --avg 用）
        num, den = next((a, b) for n, a, b, _, _ in INDICATORS if n == name)
        rate = m[num] / m[den].replace(0, np.nan)
        z = rate.groupby(m["level_rank_main"]).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
        m["direct_" + indicator_tag(name)] = z.fillna(0.0).clip(-4, 4)
    m["drafted"] = m["pick_overall"].notna()
    m["lvl_HS"] = (m["draft_level"] == "HS").astype(float)  # 出身 dummies（基準：大學），--scout 時進結構模型
    m["lvl_JC"] = (m["draft_level"] == "JC").astype(float)
    m["undrafted"] = (~m["drafted"]).astype(float)
    # wOBA 代理（FanGraphs 2015 權重的近似，固定不逐年），只用在基準模型 Mh
    pa = m["PA"].replace(0, np.nan)
    m["woba_proxy"] = (0.69 * m["BB"] + 0.89 * (m["H"] - m["XBH"]) + 1.27 * (m["XBH"] - m["HR"]) + 2.10 * m["HR"]) / pa
    m["woba_z"] = (
        (
            (m["woba_proxy"] - m.groupby("lvl")["woba_proxy"].transform("mean"))
            / m.groupby("lvl")["woba_proxy"].transform("std")
        )
        .fillna(0.0)
        .clip(-4, 4)
    )
    return m


# =====================================================================================
# 3. 設計矩陣（共變數 → 數字矩陣）
# =====================================================================================
def ortho_poly(x: np.ndarray, degree: int) -> np.ndarray:
    """標準化後的正交多項式基底（不含常數項），shape (n, degree)。"""
    z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    z = np.nan_to_num(z)
    X = np.column_stack([z**d for d in range(1, degree + 1)])
    Q, _ = np.linalg.qr(np.column_stack([np.ones_like(z), X]))
    Q = Q[:, 1:] * np.sqrt(len(z))  # 尺度回到約單位變異
    for d in range(degree):  # QR 的欄位正負號是任意的，強制與 z^d 同向，係數才好解讀
        if np.corrcoef(Q[:, d], X[:, d])[0, 1] < 0:
            Q[:, d] *= -1
    return Q


def design(m: pd.DataFrame, poly: int, covariates=None, dummies=(), linear=None) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for c in (COVARIATES if covariates is None else covariates):
        if c == "log_bonus_f":
            # 只在簽約金已知者內做正交多項式（在已知者內置中），未知者為 0；
            # 否則一次項與 bonus_known 的 0/1 幾乎共線（ESS 3–5、r-hat 1.4–2.2），是一道山脊
            known = m["bonus_known"].to_numpy().astype(bool)
            P = np.zeros((len(m), poly))
            P[known] = ortho_poly(m.loc[known, "log_bonus"].to_numpy(dtype=float), poly)
            cols.append(known.astype(float)[:, None])
            names.append("bonus_known")
        else:
            P = ortho_poly(m[c].to_numpy(dtype=float), poly)
        cols.append(P)
        names += [f"{c}^{d}" for d in range(1, poly + 1)]
    for c in (LINEAR_COVS if linear is None else linear):
        z = m[c].to_numpy(dtype=float)
        z = (z - z.mean()) / (z.std() + 1e-9)
        cols.append(z[:, None])
        names.append(c)
    for c in dummies:  # 0/1 不標準化
        cols.append(m[c].to_numpy(dtype=float)[:, None])
        names.append(c)
    return np.column_stack(cols), names


def f_eta_basis(eta, feta: str):
    """factor 進結構模型的基底（pytensor），回傳 list of tensors。"""
    if feta == "linear":
        return [eta]
    if feta == "quad":
        return [eta, eta**2]
    if feta == "cubic":
        return [eta, eta**2, eta**3]
    if feta == "spline":  # 截斷冪三次樣條
        return [eta, eta**2, eta**3] + [pt.maximum(eta - k, 0.0) ** 3 for k in SPLINE_KNOTS]
    raise ValueError(feta)


# ---------------------------------------------------------------- 模型


def assign_indicators(factors: int, statcast: str = "none"):
    """回傳 [(name, num, den, kind, factor_k)]，依 factor 數把指標併到對應 factor。
    statcast = none（成績單 15 個）/ add（成績單 + Statcast 7 個）/ only（Statcast 7 個 + 成績單的速度 4 個，樣本只有 3A）。"""
    out = []
    if factors <= 0:  # Mh（無測量模型）
        return out
    items = list(INDICATORS) if statcast != "only" else [t for t in INDICATORS if t[4] == 2]
    if statcast in ("add", "only", "both"):
        items += STATCAST_INDICATORS
    for n, a, b, kind, k in items:
        if factors == 1:
            kk = 0
        elif factors >= 5:
            kk = k
        else:
            kk = FOLD[factors].get(k, k)
            if kk is None:
                continue
        out.append((n, a, b, kind, kk))
    return out


# =====================================================================================
# 4. 建模型
# =====================================================================================
def level_intercepts(name: str, present: np.ndarray, sigma: float):
    """層級截距：只對資料裡出現的層級建自由參數（{name}_free），再放回 N_LEVELS 格，沒資料的層級固定 0，
    不留「只在先驗上亂走」的自由度。回傳長度 N_LEVELS 的向量（Deterministic，名字 {name}）。要在 pm.Model() 裡面呼叫。"""
    free = pm.Normal(f"{name}_free", 0, sigma, shape=len(present))
    full = pt.set_subtensor(pt.zeros(N_LEVELS)[present], free)
    return pm.Deterministic(name, full)


def identification_report(m: pd.DataFrame, factors: int, statcast: str = "none") -> list:
    """識別檢查：哪些層級、哪些指標在這個球季沒有資料。回傳要印的行。"""
    lines = []
    present = np.unique(m["lvl"].to_numpy())
    absent = [LEVEL_LABELS[i] for i in range(N_LEVELS) if i not in present]
    lines.append(f"層級：有資料 {[LEVEL_LABELS[i] for i in present]}；沒資料 {absent or '無'}（沒資料的層級不建截距）")
    for name, num, den, kind, k in assign_indicators(factors, statcast):
        if num not in m.columns:
            lines.append(f"指標 {name}：建模表沒有這一欄，整個不進模型")
            continue
        y = m[num].to_numpy(dtype=float)
        d = m[den].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(d) & (d > 0)
        n_ok = int(ok.sum())
        if n_ok == 0:
            lines.append(f"指標 {name}：本季沒有資料，整個不進模型")
        else:
            lv_missing = [LEVEL_LABELS[i] for i in present if not ok[m["lvl"].to_numpy() == i].any()]
            if lv_missing:
                lines.append(f"指標 {name}：{n_ok:,} 人季有資料；層級 {lv_missing} 沒有（該格截距固定 0）")
    return lines


def build_scout_block(m: pd.DataFrame):
    """球探先驗 ζ：每位選秀球員一個分數，順位與簽約金是它的兩個指標。回傳長度 n 的向量（非選秀者為 0）。
    識別：兩個指標撐一個因子只有三個可觀測動差，要四個參數，差一個約束；約束 = 兩個指標標準化後負荷量大小相等。
    ζ 高 = 評價高 = 順位靠前，所以 log 順位的負荷量是 −λ，λ ≥ 0，符號由此固定。要在 pm.Model() 裡面呼叫。"""
    n = len(m)
    d_idx = np.where(m["drafted"].to_numpy())[0]
    zeta = pm.Normal("zeta", 0, 1, shape=len(d_idx))
    zp = m["log_pick"].to_numpy(dtype=float)[d_idx]
    zp = (zp - zp.mean()) / zp.std()
    kb = np.where(m["bonus_known"].to_numpy()[d_idx])[0]  # 有簽約金的選秀球員（在 ζ 裡的位置）
    zb = m["log_bonus"].to_numpy(dtype=float)[d_idx][kb]
    zb = (zb - zb.mean()) / zb.std()
    lam = pm.HalfNormal("lambda_scout", 1.0)
    pm.Deterministic("lambda_logpick", -lam)
    pm.Deterministic("lambda_logbonus", lam)
    sigma_pick = pm.HalfNormal("sigma_logpick", 1.0)
    sigma_bonus = pm.HalfNormal("sigma_logbonus", 1.0)
    pm.Normal("y_logpick", mu=-lam * zeta, sigma=sigma_pick, observed=zp)
    pm.Normal("y_logbonus", mu=lam * zeta[kb], sigma=sigma_bonus, observed=zb)
    return pt.set_subtensor(pt.zeros(n)[d_idx], zeta)


def build_measurement(m: pd.DataFrame, factors: int, crossload: bool, param: str, verify: bool = False, statcast: str = "none"):
    """四種能力 η（每人季一個向量）和十五個指標的似然。回傳 η（n × factors 的張量）。要在 pm.Model() 裡面呼叫。
    - 球團效應 θ_org：每種能力一組，跨球團總和為零（能力的位置由此識別）。
    - 能力之間允許相關（LKJ 先驗），各自變異數固定 1（尺度識別）。
    - 指標：比例用二項、計數用負二項加 offset；層級截距 γ_jℓ；負荷量 λ 不限制符號（取樣後 align_signs 對齊）。"""
    n = len(m)
    n_org = int(m["org_i"].max()) + 1
    lvl = m["lvl"].to_numpy()
    org = m["org_i"].to_numpy()
    sigma_org = pm.HalfNormal("sigma_org", 1.0, shape=factors)
    z_org = pm.Normal("theta_org_z", 0, 1, shape=(n_org, factors))
    theta_org = pm.Deterministic("theta_org", (z_org - z_org.mean(axis=0, keepdims=True)) * sigma_org[None, :])
    delta = pm.Normal("delta", 0, 1, shape=factors)  # 年齡（相對層級）對能力均值的影響
    mu = theta_org[org] + delta[None, :] * m["age_rel"].to_numpy()[:, None]
    chol = None
    if factors > 1:
        chol = pm.LKJCorr("factor_chol", n=factors, eta=2.0)  # 相關矩陣的 Cholesky 因子
        pm.Deterministic("factor_corr", pt.dot(chol, chol.T))
    if param == "centered":
        # 中心化：η 直接是參數。資料對多數球員很強（幾百打席），η 被似然釘住，和全域參數脫鉤，取樣快。
        if factors > 1:
            eta = pm.MvNormal("eta", mu=mu, chol=chol, shape=(n, factors))
        else:
            eta = pm.Normal("eta", mu=mu[:, 0], sigma=1.0, shape=n)[:, None]
    else:
        z = pm.Normal("eta_z", 0, 1, shape=(n, factors))
        eta = pm.Deterministic("eta", mu + (pt.dot(z, chol.T) if factors > 1 else z))

    anchors = anchors_for(statcast)
    for name, num, den, kind, k in assign_indicators(factors, statcast):
        if num not in m.columns:  # 建模表沒有這一欄（如沒有進壘點的 Statcast 資料），不建參數
            continue
        y = m[num].to_numpy(dtype=float)
        d = m[den].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(d) & (d > 0)
        if kind == "binom":
            ok &= y <= d
        idx = np.where(ok)[0]
        if len(idx) == 0:  # 本季沒有這個指標（如 2006 無揮空率），不建參數
            continue
        tag = indicator_tag(name)
        gamma = level_intercepts(f"gamma_{tag}", np.unique(lvl[idx]), 2.0)  # 層級截距（不另設總截距，避免山脊）
        lam = pm.Normal(f"lambda_{tag}", 0, 1)
        lin = gamma[lvl[idx]] + lam * eta[idx, k]
        if crossload and name in CROSSLOADS and factors > max(k, CROSSLOADS[name]):
            k2 = CROSSLOADS[name]
            lam_x = pm.Normal(f"lambda_{tag}_x{FACTOR_NAMES[k2]}", 0, 0.5)
            lin = lin + lam_x * eta[idx, k2]
        is_anchor = anchors[k][0] == name
        if verify and not is_anchor:
            # --verify 構念驗證：錨指標以外的每個指標都可以載到定義之外的構念，先驗壓在零附近（sd 0.3）；跑出去的就是定義有問題的。
            # 錨指標的交叉負荷固定為 0：每把尺至少有一個只屬於它的指標，因子才不能任意旋轉（否則主負荷量會被拉到零、鏈不收斂）。
            for k2 in range(factors):
                if k2 == k or (crossload and name in CROSSLOADS and CROSSLOADS[name] == k2):
                    continue
                lam_v = pm.Normal(f"lambda_{tag}_x{FACTOR_NAMES[k2]}", 0, 0.3)
                lin = lin + lam_v * eta[idx, k2]
        if kind == "binom":
            pm.Binomial(f"y_{tag}", n=d[idx], logit_p=lin, observed=y[idx])
        elif kind == "normal":  # 連續量（已層級內標準化）：進場球越多量得越準
            sigma = pm.HalfNormal(f"sigma_{tag}", 1.0)
            pm.Normal(f"y_{tag}", mu=lin, sigma=sigma * np.sqrt(100.0 / d[idx]), observed=y[idx])
        else:
            phi = pm.Gamma(f"phi_{tag}", 2, 0.5)
            pm.NegativeBinomial(f"y_{tag}", mu=pt.exp(lin + np.log(d[idx])), alpha=phi, observed=y[idx])
    return eta


def structural_predictor(
    prefix: str,
    m: pd.DataFrame,
    X: np.ndarray,
    s_idx: np.ndarray,
    mode: str,
    eta,
    zeta_all,
    factors: int,
    feta: str,
    woba: bool = False,
    org_slopes: bool = False,
    club: bool = False,
    theta_org=None,
    club_type: bool = False,
):
    """結構模型的線性預測式（結果遮罩內的每個人季一個數）。升級用一條（prefix ""），WAR 用兩條（"t_" 幾年登上、"w_" WAR）。
    = 層級截距 + 共變數 + 球團效應 (+ 球探先驗) (+ M1：四種能力) (+ Mh：wOBA 多項式)。要在 pm.Model() 裡面呼叫。
    --club（階層的第二層）：
        球團截距 u_org[o] = a_club · 母隊團隊 WAR[o] + Σ_k a_theta[k] · θ_org[o, k] + τ · z[o]
            前兩項是「截距被母隊強弱、農場人才解釋的部分」（u_org_ctx），τ 是解釋完剩下的；a_theta < 0 = 農場人才越多越不升（塞車）
        個人層：母隊該守位缺口、守位主力、農場塞車進共變數 X（b_cov）；缺口 × 能力、缺口 × 球探評價是跨層交互作用
            （beta_need_x_*：缺人時這個能力的權重多多少；負 = 缺人時門檻降低）"""
    n_org = int(m["org_i"].max()) + 1
    lvl = m["lvl"].to_numpy()[s_idx]
    org = m["org_i"].to_numpy()[s_idx]
    gamma_s = level_intercepts(f"{prefix}gamma_s", np.unique(lvl), 2.0)
    b_cov = pm.Normal(f"{prefix}b_cov", 0, 1, shape=X.shape[1])
    tau = pm.HalfNormal(f"{prefix}tau_org", 0.5)
    u_resid = pm.Normal(f"{prefix}u_org_z", 0, 1, shape=n_org) * tau
    if club:
        club_org = m.groupby("org_i")["club_war_c"].first().reindex(range(n_org)).fillna(0.0).to_numpy()
        a_club = pm.Normal(f"{prefix}a_club_war", 0, 1)
        ctx = a_club * club_org
        if theta_org is not None:
            a_theta = pm.Normal(f"{prefix}a_theta", 0, 1, shape=factors)
            ctx = ctx + pt.dot(theta_org, a_theta)
        u_ctx = pm.Deterministic(f"{prefix}u_org_ctx", ctx)
        u_org = pm.Deterministic(f"{prefix}u_org", u_ctx + u_resid)
        need = m["need_pos"].to_numpy(dtype=float)[s_idx]
    else:
        u_org = pm.Deterministic(f"{prefix}u_org", u_resid)
    lin = gamma_s[lvl] + pt.dot(X[s_idx], b_cov) + u_org[org]
    if zeta_all is not None:
        beta_scout = pm.Normal(f"{prefix}beta_scout", 0, 1)
        lin = lin + beta_scout * zeta_all[s_idx]
        if club:
            bx = pm.Normal(f"{prefix}beta_need_x_scout", 0, 1)
            lin = lin + bx * need * zeta_all[s_idx]
        if org_slopes:  # 球團各自對球探評價的權重 = 共識 + 偏離
            sig = pm.HalfNormal(f"{prefix}sigma_org_scout", 0.5)
            dev = pm.Deterministic(
                f"{prefix}beta_scout_org", pm.Normal(f"{prefix}beta_scout_org_z", 0, 1, shape=n_org) * sig
            )
            lin = lin + dev[org] * zeta_all[s_idx]
    if mode == "M1":
        for k in range(factors):
            basis = f_eta_basis(eta[s_idx, k], feta)
            beta = pm.Normal(f"{prefix}beta_{FACTOR_NAMES[k]}", 0, 1 if feta == "linear" else 0.5, shape=len(basis))
            for j in range(len(basis)):
                lin = lin + beta[j] * basis[j]
            if org_slopes:  # --org-slopes：每個球團對這個能力自己的斜率 = β_0 + σ_k·z[o]
                sig = pm.HalfNormal(f"{prefix}sigma_org_{FACTOR_NAMES[k]}", 0.5)
                dev = pm.Deterministic(
                    f"{prefix}beta_{FACTOR_NAMES[k]}_org",
                    pm.Normal(f"{prefix}beta_{FACTOR_NAMES[k]}_org_z", 0, 1, shape=n_org) * sig,
                )
                lin = lin + dev[org] * basis[0]  # 只放一次項
            if club:  # --club：缺人時這個能力的權重多多少（只放一次項）
                bx = pm.Normal(f"{prefix}beta_need_x_{FACTOR_NAMES[k]}", 0, 1)
                lin = lin + bx * need * basis[0]
        if club_type and factors >= 2:
            # --club-type：球隊買的是平均還是分布。型態 = 打得遠 − 打得到球（高變異型為正）。
            # 母隊強弱 × 型態（beta_type_x_club：負 = 弱隊更偏好高變異型，勝率凸性的預測）
            # 加每個球團自己對型態的斜率（beta_type_org[o] = σ·z[o]：三十隊各自偏好哪一種分布）。
            type_ = eta[s_idx, 1] - eta[s_idx, 0]
            club_c = m["club_war_c"].to_numpy(dtype=float)[s_idx]
            b_type_club = pm.Normal(f"{prefix}beta_type_x_club", 0, 1)
            sig_t = pm.HalfNormal(f"{prefix}sigma_org_type", 0.5)
            dev_t = pm.Deterministic(f"{prefix}beta_type_org", pm.Normal(f"{prefix}beta_type_org_z", 0, 1, shape=n_org) * sig_t)
            lin = lin + b_type_club * club_c * type_ + dev_t[org] * type_
    if mode == "Mh" or woba:  # --woba：M0/M1 也加，看能力之外 wOBA 還多說什麼
        W = ortho_poly(m["woba_z"].to_numpy(dtype=float), 3)[s_idx]
        beta_woba = pm.Normal(f"{prefix}beta_woba", 0, 1, shape=3)
        lin = lin + pt.dot(W, beta_woba)
    pm.Deterministic(f"{prefix}lin", lin)  # 每人季的預測分數（決策評估用）
    return lin


def build(
    m: pd.DataFrame,
    outcome: str,
    factors: int,
    feta: str,
    poly: int,
    mode: str,
    crossload: bool = False,
    param: str = "centered",
    scout: bool = False,
    extra: bool = False,
    direct: bool = False,
    avg: bool = False,
    woba: bool = False,
    org_slopes: bool = False,
    club: bool = False,
    verify: bool = False,
    statcast: str = "none",
    club_type: bool = False,
):
    """組出一個完整的 PyMC 模型。回傳 (model, info)，info 是字典：
        s_idx     結果遮罩內的列索引
        xnames    設計矩陣每一欄的名字
        initvals  取樣起始值（負荷量按棒球常識給符號）
    mode：M0 / M1 / Mh。outcome：promoted（下一季升級）或 war（幾年登上 + WAR|登上）。"""
    # 設計矩陣：哪些共變數進結構模型
    covariates = ["age_rel", "log_pa"] if scout else None  # 球探版把順位、簽約金移去當 ζ 的指標
    dummies = ("undrafted", "lvl_HS", "lvl_JC") if scout else ()
    linear = list(LINEAR_COVS)
    if extra:  # 球探可能忽略的候選
        dummies = tuple(dummies) + ("has_prev", "repeat_level", "multi_level")
        linear += ["d_k", "d_hr", "d_bb", "years_in", "pa_share"]
    if direct:  # 能力之外的細節（只在 M1 才有意義，但三個模式都放，比較才公平）
        linear += ["direct_" + indicator_tag(n) for n in DIRECT]
    if avg:  # 原始打擊率跟能力並排：係數 = 球隊對數字本身的反應（預期升級段 > 0、WAR 段 ≈ 0）
        linear += ["direct_AVG"]
    if club:  # 母隊該守位缺口、守位主力 WAR、農場同守位層級較高 / 總人數；母隊整體強弱放在 u_org 的第二層
        linear += ["need_pos", "top_war_c", "queue_above", "queue_total"]
        dummies = tuple(dummies) + ("pos_known",)
    X, xnames = design(m, poly, covariates=covariates, dummies=dummies, linear=linear)
    mask = (m["structural"] if outcome == "promoted" else m["war_structural"]).to_numpy()
    s_idx = np.where(mask)[0]

    with pm.Model() as model:
        zeta_all = build_scout_block(m) if scout else None
        eta = build_measurement(m, factors, crossload, param, verify, statcast) if mode in ("M0", "M1") else None
        theta_org = model["theta_org"] if (club and eta is not None) else None
        if outcome == "promoted":
            lin = structural_predictor(
                "", m, X, s_idx, mode, eta, zeta_all, factors, feta, woba, org_slopes, club, theta_org, club_type
            )
            pm.Bernoulli("promoted", logit_p=lin, observed=m["promoted"].to_numpy()[s_idx].astype(int))
        else:
            # 第一段：幾年登上大聯盟（log-normal 加速失敗時間；六季內沒登上 = 右設限）
            lin_t = structural_predictor(
                "t_", m, X, s_idx, mode, eta, zeta_all, factors, feta, woba, org_slopes, club, theta_org, club_type
            )
            sigma_t = pm.HalfNormal("sigma_t", 1.0)
            pm.Censored(
                "debut",
                pm.LogNormal.dist(mu=lin_t, sigma=sigma_t),
                lower=None,
                upper=DEBUT_CENSOR,
                observed=m["T_debut"].to_numpy(dtype=float)[s_idx],
            )
            # 第二段：登上後六季的 WAR，只對登上的人。
            # WAR 的分布：一半 ≤ 0、中位數 0.03、九成分位 6.9、最高 30。直接用 t 分布會估出自由度 < 1（比柯西還厚），
            # 位置只盯著中位數那團 0，所有係數都變 0。所以先 asinh 轉換（像 log，但 0 與負數也能放），
            # 再用自由度 ≥ 3 的 t 分布（變異數存在，位置會隨共變數動）。係數的單位是 asinh(WAR)，大 WAR 時約等於 log。
            reached = m["reached_fwd"].to_numpy()[s_idx].astype(bool)
            r_idx = np.where(reached)[0]
            lin_w = structural_predictor(
                "w_", m, X, s_idx, mode, eta, zeta_all, factors, feta, woba, org_slopes, club, theta_org, club_type
            )
            nu = pm.Deterministic("nu", 3.0 + pm.Gamma("nu_extra", 2, 0.5))
            sigma_w = pm.HalfNormal("sigma_w", 2.0)
            pm.StudentT(
                "war",
                nu=nu,
                mu=lin_w[r_idx],
                sigma=sigma_w,
                observed=np.arcsinh(m["war_fwd"].to_numpy(dtype=float)[s_idx][r_idx]),
            )

    initvals = {}
    if scout:
        initvals["lambda_scout"] = 0.7
    if mode in ("M0", "M1"):
        for name, _, _, _, _ in assign_indicators(factors, statcast):
            if f"lambda_{indicator_tag(name)}" in model.named_vars:  # 本季有這個指標才有起始值
                initvals[f"lambda_{indicator_tag(name)}"] = 0.3 * INIT_SIGN[name]
    info = {"s_idx": s_idx, "xnames": xnames, "initvals": initvals}
    return model, info


# =====================================================================================
# 5. 取樣與 log-likelihood
# =====================================================================================
INIT_SETTINGS = {
    "advi": {"init": "advi+adapt_diag", "n_init": 30000},  # 先跑 ADVI 量每個參數的尺度，當起始質量矩陣（預設）
    "grad": {"init": "jitter+adapt_diag_grad"},  # 用梯度變異數估尺度
    "diag": {"init": "adapt_diag"},  # 全部從 1 開始學（參數尺度差兩百倍時暖身走不完）
}


def save_idata(idata, path: Path):
    """存後驗；存檔失敗不中斷（印訊息繼續）。"""
    try:
        idata.to_netcdf(path)
    except Exception as e:  # noqa: BLE001
        print(f"  （存檔失敗，繼續：{type(e).__name__}: {str(e)[:80]}）", flush=True)


def sample_or_load(
    model, info: dict, nc_path: Path, reuse: bool, chains: int, draws: int, tune: int, sampler: str, init: str, save: bool = True
):
    """有存檔且 --reuse 就讀回，否則取樣並存檔（save=False 就不落地：基準模型 M0／Mh 只用來算 LOO，省配額）。回傳 (idata, 是否為讀回的)。"""
    if reuse and nc_path.exists():
        print(f"  讀取 {nc_path.name}", flush=True)
        return az.from_netcdf(nc_path), True
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=0.9,
            progressbar=False,
            random_seed=20261006,
            nuts_sampler=sampler,
            initvals=info["initvals"],
            **INIT_SETTINGS[init],
        )
    if save:
        save_idata(idata, nc_path)  # 先存，後處理出錯不白費
    return idata, False


def ensure_loglik(idata, model, outcome: str, m: pd.DataFrame, s_idx: np.ndarray, nc_path: Path, save: bool = True):
    """確保 idata 裡有每個觀測的 log-likelihood（LOO 要用）。WAR 的兩段加總成 hurdle。"""
    ll_vars = ["promoted"] if outcome == "promoted" else ["debut", "war"]
    missing = "log_likelihood" not in idata.children or any(v not in idata.log_likelihood for v in ll_vars)
    if missing:
        compute_ll(idata, model, ll_vars)
    if outcome == "war" and (missing or "hurdle" not in idata.log_likelihood):
        attach_hurdle_ll(idata, combined_loglik(idata, outcome, m, s_idx), "hurdle")
        missing = True
    if missing and save:
        save_idata(idata, nc_path)


def waic_manual(ll: np.ndarray):
    """ll: (n_obs, S)。WAIC = Σ_i [log mean_s exp(ll_is) − Var_s(ll_is)]。"""
    S = ll.shape[1]
    lppd_i = np.logaddexp.reduce(ll, axis=1) - np.log(S)
    p_i = ll.var(axis=1, ddof=1)
    elpd_i = lppd_i - p_i
    return {
        "elpd_waic": float(elpd_i.sum()),
        "p_waic": float(p_i.sum()),
        "se": float(np.sqrt(len(elpd_i) * elpd_i.var(ddof=1))),
        "elpd_i": elpd_i,
    }


def combined_loglik(idata, outcome: str, m: pd.DataFrame, s_idx: np.ndarray):
    """回傳 (n_obs, S) 的 pointwise log-lik；WAR 的 hurdle 把「幾年登上（設限）」與 WAR|登上加總。"""
    if outcome == "promoted":
        da = idata.log_likelihood["promoted"]
        return da.stack(sample=("chain", "draw")).values
    r = idata.log_likelihood["debut"].stack(sample=("chain", "draw")).values
    w = idata.log_likelihood["war"].stack(sample=("chain", "draw")).values
    reached = m["reached_fwd"].to_numpy()[s_idx].astype(bool)
    total = r.copy()
    total[reached] += w
    return total


def attach_hurdle_ll(idata, ll: np.ndarray, name: str):
    import xarray as xr

    n_chain = idata.posterior.sizes["chain"]
    n_draw = idata.posterior.sizes["draw"]
    arr = ll.reshape(ll.shape[0], n_chain, n_draw).transpose(1, 2, 0)
    da = xr.DataArray(arr, dims=("chain", "draw", f"{name}_dim_0"))
    idata.log_likelihood[name] = da
    return idata


def compute_ll_manual(idata, model, var_names, compile_kwargs=None):
    """PyMC 6.3 的 compute_log_likelihood 遇到 LKJCorr 會壞：RV 名 factor_chol、值變數名 factor_chol_cholesky，
    內部 astype 的鍵對不上而丟 ValueError。這裡照它的步驟自己走一遍，把後驗的變數名改成值變數名。"""
    from pymc.model.transform.conditioning import remove_value_transforms
    from pymc.backends.arviz import apply_function_over_dataset, coords_and_dims_for_inferencedata

    umodel = remove_value_transforms(model)
    outs = [umodel.named_vars[n] for n in var_names]
    fn = umodel.compile_fn(
        inputs=umodel.value_vars,
        outs=umodel.logp(vars=outs, sum=False),
        on_unused_input="ignore",
        **(compile_kwargs or {}),
    )
    coords, dims = coords_and_dims_for_inferencedata(umodel)
    post = idata.posterior.to_dataset()
    keep, rename, dtypes = [], {}, {}
    for rv in umodel.free_RVs:
        v = umodel.rvs_to_values[rv]
        src = rv.name if rv.name in post.data_vars else v.name
        keep.append(src)
        if src != v.name:
            rename[src] = v.name
        dtypes[v.name] = v.type.dtype
    ds = post[keep].rename(rename).astype(dtypes, copy=False)
    ll = apply_function_over_dataset(
        fn,
        ds,
        output_var_names=list(var_names),
        sample_dims=("chain", "draw"),
        dims=dims,
        coords=coords,
        progressbar=False,
    )
    idata["log_likelihood"] = ll


def compute_ll(idata, model, var_names):
    """先用 PyMC 內建；LKJCorr 命名 bug（ValueError: exact match required）改走手動版；numba 偶發 SystemError 則換編譯模式。"""
    for kw in ({}, {"compile_kwargs": {"mode": "FAST_RUN"}}, {"compile_kwargs": {"mode": "FAST_COMPILE"}}):
        try:
            with model:
                pm.compute_log_likelihood(idata, var_names=var_names, progressbar=False, **kw)
            return
        except ValueError as e:
            if "exact match" in str(e):
                print("  PyMC compute_log_likelihood 的 LKJCorr 命名 bug，改用手動版", flush=True)
                break
            print(f"  log-likelihood 計算失敗（{kw or '預設'}）：ValueError: {str(e)[:300]}，換模式重試", flush=True)
        except Exception as e:  # noqa: BLE001
            print(
                f"  log-likelihood 計算失敗（{kw or '預設'}）：{type(e).__name__}: {str(e)[:300]}，換模式重試",
                flush=True,
            )
    for kw in ({}, {"mode": "FAST_COMPILE"}):
        try:
            compute_ll_manual(idata, model, var_names, compile_kwargs=kw)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  手動 log-likelihood 失敗（{kw or '預設'}）：{type(e).__name__}: {str(e)[:300]}", flush=True)
    raise RuntimeError("log-likelihood 所有方法都失敗")


def dejitter_constant_loglik(idata, var: str, eps: float = 1e-8):
    """預測機率飽和到 0/1 的觀測，其 log-lik 在所有抽樣下完全相同，PSIS 會報「All tail values are the same」。
    對這些列加極小擾動，ELPD 不受影響。回傳被處理的觀測數。"""
    da = idata.log_likelihood[var]
    arr = da.values
    obs_axis = [i for i, d in enumerate(da.dims) if d not in ("chain", "draw")][0]
    flat = np.moveaxis(arr, obs_axis, -1).reshape(-1, arr.shape[obs_axis])  # (S, n_obs)
    const = flat.std(axis=0) < 1e-12
    if const.any():
        rng = np.random.default_rng(0)
        noise = np.zeros_like(flat)
        noise[:, const] = rng.normal(0, eps, size=(flat.shape[0], int(const.sum())))
        new = np.moveaxis((flat + noise).reshape(np.moveaxis(arr, obs_axis, -1).shape), -1, obs_axis)
        idata.log_likelihood[var] = da.copy(data=new)
    return int(const.sum())


# =====================================================================================
# 6. 診斷
# =====================================================================================
def align_signs(idata, factors: int, feta: str, crossload: bool, prefixes=("", "t_", "w_"), statcast: str = "none"):
    """因子模型對每個 factor 都有 η_k → −η_k、λ_k → −λ_k 的鏡像對稱，似然完全不變，後驗有 2^K 座等高的峰。
    取樣時不限制符號；取樣後**逐 draw** 看錨指標的負荷量，符號與 ANCHORS 約定不同的 draw，把該 factor 的所有量翻過來。
    逐 draw 而不是逐鏈：鏈在途中換邊也能對。等於把參數空間沿對稱面折成一半。
    翻的量：該 factor 的 λ（含交叉負荷）、η[:, k]、delta[k]、theta_org[:, k]、eta_z[:, k]、β_k（奇次項）、
    factor_chol 第 k 列、factor_corr 的第 k 列與行。LOO/WAIC 不受影響。回傳每個 factor 被翻的 draw 數。"""
    post = idata.posterior
    tags = {name: indicator_tag(name) for name, *_ in INDICATORS + STATCAST_INDICATORS}
    inds = assign_indicators(factors, statcast)
    anchors = anchors_for(statcast)
    flipped = {}
    # 球探先驗 ζ 的符號由 lambda_scout ≥ 0 固定（log 順位負荷量 = −λ），不需翻
    for k in range(factors):
        anchor, want = anchors[k]  # f1 時 k=0，錨 K% 負
        if f"lambda_{tags[anchor]}" not in post.data_vars:
            print(f"  錨指標 {anchor} 本季不在模型裡，factor {FACTOR_NAMES[k]} 不做鏡像對齊", flush=True)
            flipped[FACTOR_NAMES[k]] = 0
            continue
        s = np.sign(post[f"lambda_{tags[anchor]}"].values) * want  # (chain, draw)；+1 不動，−1 翻
        s[s == 0] = 1
        flipped[FACTOR_NAMES[k]] = int((s < 0).sum())
        if (s > 0).all():
            continue

        def flip(var, axis=None):
            """axis=None：整個變數乘 s；否則只有 axis 維的第 k 個切片乘 s。s 廣播到 (chain, draw, 1, 1, ...)。"""
            if var not in post.data_vars:
                return
            a = post[var].values.copy()
            sb = s.reshape(s.shape + (1,) * (a.ndim - 2))
            if axis is None:
                a = a * sb
            else:
                idx = [slice(None)] * a.ndim
                idx[axis] = k
                sub = a[tuple(idx)]
                a[tuple(idx)] = sub * s.reshape(s.shape + (1,) * (sub.ndim - 2))
            post[var] = post[var].copy(data=a)

        for name, _, _, _, kk in inds:
            if kk == k:
                flip(f"lambda_{tags[name]}")
        for var in list(
            post.data_vars
        ):  # 交叉負荷 lambda_<指標>_x<factor k> 跟著 factor k 翻（--crossload 與 --verify 都適用）
            if var.startswith("lambda_") and var.endswith(f"_x{FACTOR_NAMES[k]}"):
                flip(var)
        flip("eta", axis=3)
        flip("eta_z", axis=3)
        flip("delta", axis=2)
        flip("theta_org", axis=3)
        flip("theta_org_z", axis=3)
        for pre in ("", "t_", "w_"):  # 球團斜率的偏離整個跟著 η_k 翻
            flip(f"{pre}beta_{FACTOR_NAMES[k]}_org")
            flip(f"{pre}beta_{FACTOR_NAMES[k]}_org_z")
            flip(f"{pre}beta_need_x_{FACTOR_NAMES[k]}")
            flip(f"{pre}a_theta", axis=2)  # --club：缺口 × η_k、θ_org[:, k] 的第二層係數
            if k in (0, 1):  # --club-type：型態 = η_1 − η_0，任一邊翻了型態就不是原來的定義；錨夠強時不會發生，發生就印出來
                for v in (f"{pre}beta_type_x_club", f"{pre}beta_type_org"):
                    if v in post.data_vars:
                        print(f"  注意：factor {FACTOR_NAMES[k]} 有翻轉的 draw，{v} 的定義在那些 draw 不成立", flush=True)
        if factors > 1:
            flip("factor_chol", axis=2)  # 第 k 列
            if "factor_corr" in post.data_vars:
                flip("factor_corr", axis=2)
                flip("factor_corr", axis=3)  # 第 k 列與第 k 行（對角被乘兩次 → 不變）
        for pre in prefixes:
            var = f"{pre}beta_{FACTOR_NAMES[k]}"
            if var not in post.data_vars:
                continue
            if feta in ("linear", "quad", "cubic"):
                a = post[var].values.copy()
                for j in range(a.shape[-1]):
                    if j % 2 == 0:  # basis[j] = η^(j+1)：奇次項翻號
                        a[..., j] *= s
                post[var] = post[var].copy(data=a)
            # spline 基底對 η → −η 不是符號變換，不翻；對齊後 β 的解讀以 η 方向為準
    return flipped


def tree_depth_stats(idata, max_depth: int = 10):
    """回傳 (到最大樹深比例, 平均步長, 平均樹深, 平均每次迭代的梯度次數)。"""
    ss = idata.sample_stats
    td = ss["tree_depth"].values if "tree_depth" in ss else None
    frac = float((td >= max_depth).mean()) if td is not None else float("nan")
    mean_td = float(td.mean()) if td is not None else float("nan")
    n_steps = float(ss["n_steps"].mean()) if "n_steps" in ss else (2 ** mean_td if td is not None else float("nan"))
    step = float(ss["step_size"].mean()) if "step_size" in ss else float("nan")
    return frac, step, mean_td, n_steps


def scale_table(idata, m: pd.DataFrame) -> pd.DataFrame:
    """每群參數的後驗 sd（中位數、最小、最大）與 ESS 中位數。η 依打席分四組。
    sd 的最大/最小比就是對角質量矩陣要跨的尺度差；ESS 最低的群是拖慢取樣的群。"""
    post = idata.posterior
    ess = az.ess(idata)
    ess = ess.to_dataset() if hasattr(ess, "to_dataset") else ess
    rows = {}

    def add(name, sd, e):
        sd = np.asarray(sd).ravel()
        e = np.asarray(e).ravel()
        rows[name] = {
            "n": sd.size,
            "sd 中位": np.median(sd),
            "sd 最小": sd.min(),
            "sd 最大": sd.max(),
            "ESS 中位": np.median(e),
            "ESS 最小": e.min(),
        }

    groups = {
        "λ 負荷量": lambda v: v.startswith("lambda_"),
        "γ 層級截距": lambda v: v.startswith("gamma_")
        and "gamma_s" not in v
        and (v.endswith("_free") or v + "_free" not in post.data_vars),
        "φ 過度離散": lambda v: v.startswith("phi_"),
        "θ_org 球團": lambda v: v == "theta_org",
        "σ_org/τ_org": lambda v: v in ("sigma_org", "tau_org"),
        "δ 年齡": lambda v: v == "delta",
        "factor_corr": lambda v: v == "factor_corr",
        "結構 γ_s/b_cov/β": lambda v: v == "gamma_s" or v == "b_cov" or v.startswith("beta_"),
        "u_org 結構球團": lambda v: v == "u_org",
        "ζ 球探先驗": lambda v: v == "zeta",
    }
    for name, pred in groups.items():
        vs = [v for v in post.data_vars if pred(v)]
        if vs:
            add(
                name,
                np.concatenate([post[v].std(("chain", "draw")).values.ravel() for v in vs]),
                np.concatenate([np.asarray(ess[v]).ravel() for v in vs if v in ess]),
            )
    if "eta" in post.data_vars:
        sd = post["eta"].std(("chain", "draw")).values
        e = np.asarray(ess["eta"])
        pa = m["PA"].to_numpy()
        for lo, hi, lab in (
            (0, 50, "η，PA<50"),
            (50, 200, "η，PA 50–200"),
            (200, 400, "η，PA 200–400"),
            (400, 10**9, "η，PA≥400"),
        ):
            sel = (pa >= lo) & (pa < hi)
            if sel.any():
                add(lab, sd[sel], e[sel])
    return pd.DataFrame(rows).T


def rhat_max(idata, include_eta: bool = False):
    """所有參數裡最大的 r-hat。回傳 (數值, 參數名)。include_eta=False 時不看每人的 η、球團效應（它們數量多、r-hat 吵）。"""
    skip_suffix = ("_z", "_abs")
    per_player = ("eta", "theta_org", "u_org", "r_u_org", "w_u_org", "t_u_org", "zeta")
    names = []
    for v in idata.posterior.data_vars:
        if v.endswith(skip_suffix) or v in ("factor_chol", "factor_chol_cholesky", "lin", "t_lin", "w_lin"):
            continue
        if v + "_free" in idata.posterior.data_vars:  # 層級截距：看自由參數，不看放回七格的版本
            continue
        if not include_eta and v in per_player:
            continue
        names.append(v)
    rh = az.rhat(idata, var_names=names)
    ds = rh.to_dataset() if hasattr(rh, "to_dataset") else rh
    worst = max(ds.data_vars, key=lambda v: float(ds[v].max()))
    return float(ds[worst].max()), worst


def diagnostics(idata, seconds: float, n_iter: int) -> dict:
    """一個模型的取樣診斷，給 summary 的診斷表用。"""
    div = int(idata.sample_stats["diverging"].sum())
    frac_td, step, mean_td, n_steps = tree_depth_stats(idata)
    rmax, worst = rhat_max(idata)
    rmax_eta, _ = rhat_max(idata, include_eta=True)
    return {
        "div": div,
        "rhat_max": rmax,
        "rhat_worst": worst,
        "rhat_eta": rmax_eta,
        "maxdepth": frac_td,
        "step": step,
        "mean_td": mean_td,
        "n_steps": n_steps,
        "ms_grad": seconds / max(n_iter * n_steps, 1) * 1000,
        "sec": seconds,
    }


def describe(mode: str, r: dict) -> str:
    return (
        f"{mode}：{r['sec']:.0f}s，divergences {r['div']}，max r-hat {r['rhat_max']:.3f}（{r['rhat_worst']}），含 η {r['rhat_eta']:.3f}，"
        f"到最大樹深 {r['maxdepth']:.0%}，平均樹深 {r['mean_td']:.1f}（每迭代 {r['n_steps']:.0f} 次梯度，每次約 {r['ms_grad']:.0f} ms 含 ADVI/編譯），步長 {r['step']:.3g}"
    )


# =====================================================================================
# 7. 彙整（三個模型都有了之後）
# =====================================================================================
def loading_table(idata_m1, factors: int, statcast: str = "none") -> pd.DataFrame:
    """M1 的負荷量表：每個指標屬於哪種能力、估計值、區間、ESS、r-hat。"""
    names = [v for v in idata_m1.posterior.data_vars if v.startswith("lambda_") and v != "lambda_scout"]
    tab = az.summary(idata_m1, var_names=names)[SUMMARY_COLS]
    tab.index = [i.replace("lambda_", "") for i in tab.index]
    group = {}
    for name, _, _, _, k in assign_indicators(factors, statcast):
        group[indicator_tag(name)] = FACTOR_NAMES[k] if factors > 1 else "ability"
    labels = []
    for i in tab.index:
        if "_x" in i:  # 交叉負荷：Kpct_xpower
            base, target = i.split("_x")
            labels.append(group[base] + " ×" + target)
        else:
            labels.append(group.get(i, "scout"))
    tab.insert(0, "factor", labels)
    tab["_abs"] = tab["mean"].abs()
    tab = tab.sort_values(["factor", "_abs"], ascending=[True, False]).drop(columns="_abs")
    return tab


def construct_table(idata_m1, factors: int, statcast: str = "none") -> pd.DataFrame:
    """構念驗證表（--verify）：每個指標一列。定義上屬於哪個構念、先驗（符號錨、Normal(0,1)）、後驗主負荷量，
    以及放開之後載到其他構念的交叉負荷量（先驗 Normal(0, 0.3)）。錨指標的交叉負荷固定 0（識別旋轉），表上標「錨」。
    判定：交叉負荷的 89% 區間不含 0，而且 |交叉| 超過 |主負荷| 的一半，才算「跑過去」（用相對大小，因為二項指標的負荷量本來就小）。"""
    post = idata_m1.posterior
    anchors = anchors_for(statcast)
    rows = []
    for name, _, _, _, k in assign_indicators(factors, statcast):
        tag = indicator_tag(name)
        if f"lambda_{tag}" not in post:
            continue
        main = post[f"lambda_{tag}"].values.ravel()
        row = {
            "指標": name,
            "定義上的構念": FACTOR_NAMES[k],
            "先驗符號": "+" if INIT_SIGN.get(name, 1) > 0 else "−",
            "後驗主負荷": f"{main.mean():+.2f} [{np.quantile(main, 0.055):+.2f}, {np.quantile(main, 0.945):+.2f}]",
        }
        drift = []
        is_anchor = anchors[k][0] == name
        for k2 in range(factors):
            if k2 == k:
                continue
            v = f"lambda_{tag}_x{FACTOR_NAMES[k2]}"
            if v not in post:
                row[f"交叉 → {FACTOR_NAMES[k2]}"] = "錨，固定 0" if is_anchor else ""
                continue
            x = post[v].values.ravel()
            lo, hi = np.quantile(x, 0.055), np.quantile(x, 0.945)
            ratio = x.mean() / main.mean() if abs(main.mean()) > 1e-6 else float("nan")
            row[f"交叉 → {FACTOR_NAMES[k2]}"] = f"{x.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]（主負荷的 {abs(ratio):.0%}）"
            if (lo > 0 or hi < 0) and abs(ratio) > 0.5:
                drift.append(FACTOR_NAMES[k2])
        if is_anchor:
            row["判定"] = "錨（交叉固定 0，定義這把尺的方向）"
        else:
            row["判定"] = "跑過去：" + "、".join(drift) if drift else "留在定義的構念"
        rows.append(row)
    return pd.DataFrame(rows)


def struct_table(idata, xnames: list) -> pd.DataFrame:
    """結構模型係數表；b_cov[i] 換成共變數的名字。"""
    keys = (
        "beta_",
        "b_cov",
        "gamma_s",
        "tau_org",
        "sigma_org",
        "nu",
        "sigma_w",
        "sigma_t",
        "a_theta",
        "a_club_war",
    )  # nu_extra 會被 "nu" 包含，下面排掉
    names = [
        v
        for v in idata.posterior.data_vars
        if any(k in v for k in keys) and not v.endswith("_z") and not v.endswith("_free") and v != "nu_extra"
    ]
    tab = az.summary(idata, var_names=names)[SUMMARY_COLS]
    new_index = []
    for i in tab.index:
        if "b_cov[" in i:
            j = int(i.split("[")[1].rstrip("]"))
            new_index.append(i.split("b_cov[")[0] + xnames[j])  # "t_b_cov[3]" → "t_log_pa^2"
        else:
            new_index.append(i)
    tab.index = new_index
    return tab


def unround_compare(cmp: pd.DataFrame, loos: dict) -> pd.DataFrame:
    """arviz 1.x 的 az.compare 把 elpd 與 elpd_diff 照 se 的位數四捨五入（se 20 → 只剩十位）；換回 az.loo 的原始值，
    elpd_diff 重算成「減第一名」（負 = 輸第一名）。排名、dse、weight 不動。"""
    col = "elpd" if "elpd" in cmp.columns else next(c for c in cmp.columns if c.startswith("elpd_") and c != "elpd_diff")
    raw = {k: float(getattr(v, "elpd", getattr(v, "elpd_loo", np.nan))) for k, v in loos.items()}
    cmp = cmp.copy()
    cmp[col] = [raw[k] for k in cmp.index]
    cmp["elpd_diff"] = cmp[col] - cmp[col].max()
    return cmp


def comparison_tables(idatas: dict, outcome: str, m: pd.DataFrame, s_idx: np.ndarray):
    """PSIS-LOO（ArviZ）與自算 WAIC 的比較表，以及 M1 的 Pareto-k。"""
    ll_var = "promoted" if outcome == "promoted" else "hurdle"
    loos = {t: az.loo(idatas[t], var_name=ll_var) for t in idatas}
    cmp_loo = az.compare(loos)
    cmp_loo = unround_compare(cmp_loo, loos)
    w = {t: waic_manual(combined_loglik(idatas[t], outcome, m, s_idx)) for t in idatas}
    best = max(w, key=lambda t: w[t]["elpd_waic"])
    rows = {}
    for t in w:
        dse = float(np.sqrt(len(w[t]["elpd_i"]) * (w[best]["elpd_i"] - w[t]["elpd_i"]).var(ddof=1)))
        rows[t] = {
            "elpd_waic": w[t]["elpd_waic"],
            "p_waic": w[t]["p_waic"],
            "se": w[t]["se"],
            "elpd_diff": w[best]["elpd_waic"] - w[t]["elpd_waic"],
            "dse": dse,
        }
    cmp_waic = pd.DataFrame(rows).T.sort_values("elpd_waic", ascending=False)
    loo1 = az.loo(idatas["M1"], var_name=ll_var, pointwise=True)
    pareto_k = np.asarray(loo1.pareto_k)
    return cmp_loo, cmp_waic, pareto_k


def export_scores(idatas: dict, m: pd.DataFrame, s_idx: np.ndarray, season: int, factors: int, scout: bool, out: Path):
    """每人季的分數：四種能力的後驗均值與 sd、球探先驗 ζ、結構模型的預測分數、登上機率與年數；加上結果欄位方便後處理。
    M1 的分數是主欄（score_*）；M0、Mh 的預測分數另存 score_*_M0、score_*_Mh（看兩個模型意見不同的人）。"""
    idata_m1 = idatas["M1"]
    post = idata_m1.posterior
    eta_mean = post["eta"].mean(("chain", "draw")).values
    eta_sd = post["eta"].std(("chain", "draw")).values
    df = pd.DataFrame({"key_mlbam": m["key_mlbam"], "season": season, "level_main": m["level_main"]})
    for k in range(factors):
        df[f"eta_{FACTOR_NAMES[k]}"] = eta_mean[:, k]
        df[f"eta_{FACTOR_NAMES[k]}_sd"] = eta_sd[:, k]
    if scout:
        d_idx = np.where(m["drafted"].to_numpy())[0]
        df["zeta_scout"] = np.nan
        df["zeta_scout_sd"] = np.nan
        df.loc[d_idx, "zeta_scout"] = post["zeta"].mean(("chain", "draw")).values
        df.loc[d_idx, "zeta_scout_sd"] = post["zeta"].std(("chain", "draw")).values
    for prefix, col in (("", "score_promoted"), ("t_", "score_debut_logtime"), ("w_", "score_war_asinh")):
        if f"{prefix}lin" in post.data_vars:
            df[col] = np.nan
            df.loc[s_idx, col] = post[f"{prefix}lin"].mean(("chain", "draw")).values
    if "w_lin" in post.data_vars:  # WAR|登上 的中位數（sinh 轉回 WAR 單位）
        df["score_war"] = np.nan
        df.loc[s_idx, "score_war"] = np.sinh(post["w_lin"].values).mean((0, 1))
    if "t_lin" in post.data_vars:  # 六季內登上的機率、中位登上年數
        from scipy.stats import norm

        lt = post["t_lin"].values
        st = post["sigma_t"].values[..., None]
        df["score_preach"] = np.nan
        df["score_debut_years"] = np.nan
        df.loc[s_idx, "score_preach"] = norm.cdf((np.log(DEBUT_CENSOR) - lt) / st).mean((0, 1))
        df.loc[s_idx, "score_debut_years"] = np.exp(lt).mean((0, 1)) - 1.0
    for other in ("M0", "Mh"):
        if other in idatas:
            po = idatas[other].posterior
            for prefix, col in (("", "score_promoted"), ("t_", "score_debut_logtime"), ("w_", "score_war_asinh")):
                if f"{prefix}lin" in po.data_vars:
                    df[f"{col}_{other}"] = np.nan
                    df.loc[s_idx, f"{col}_{other}"] = po[f"{prefix}lin"].mean(("chain", "draw")).values
    for c in (
        "promoted",
        "next_status",
        "war_fwd",
        "reached_fwd",
        "war_observable",
        "PA",
        "age",
        "org",
        "pick_overall",
    ):
        if c in m.columns:
            df[c] = m[c].to_numpy()
    df.to_parquet(out / "eta_M1.parquet", index=False)


def org_table(idata_m1, m: pd.DataFrame, factors: int) -> pd.DataFrame:
    """球團表（M1）。u_org：同樣的能力，這個球團多升 / 早上 / 登上後 WAR 高多少（結構段的球團隨機效應）。
    theta_org：這個球團手上的人四種能力平均偏高還是偏低（人才本身）。兩個分開看：前者是決策，後者是球員。"""
    orgs = pd.factorize(m["org"].fillna("UNK"))[1]
    post = idata_m1.posterior
    tab = pd.DataFrame(index=list(orgs))
    tab.index.name = "org"
    tab["人季"] = m["org"].fillna("UNK").value_counts().reindex(orgs).to_numpy()
    for v in ("u_org", "t_u_org", "w_u_org"):
        if v in post:
            x = post[v].values.reshape(-1, len(orgs))
            label = {"u_org": "升級 u_org", "t_u_org": "登上時間 t_u_org（負 = 快）", "w_u_org": "WAR|登上 w_u_org"}[v]
            tab[label] = x.mean(0)
            tab[label + " sd"] = x.std(0)
            if v + "_ctx" in post:  # --club：截距被母隊強弱與農場人才解釋的部分
                tab[label + " 被解釋的"] = post[v + "_ctx"].values.reshape(-1, len(orgs)).mean(0)
    for pre, lab in (("", "升級"), ("t_", "登上時間"), ("w_", "WAR")):
        for k in range(factors):
            v = f"{pre}beta_{FACTOR_NAMES[k]}_org"
            if v in post:
                base = (
                    post[f"{pre}beta_{FACTOR_NAMES[k]}"].values[..., 0]
                    if post[f"{pre}beta_{FACTOR_NAMES[k]}"].ndim == 3
                    else post[f"{pre}beta_{FACTOR_NAMES[k]}"].values
                )
                dev = post[v].values.reshape(-1, len(orgs))
                x = dev + base.reshape(-1, 1)
                tab[f"{lab} β {FACTOR_NAMES[k]}"] = x.mean(0)
                tab[f"{lab} β {FACTOR_NAMES[k]} 偏離"] = dev.mean(0)  # 該隊 − 共識（org_rulers.py 跨季合併用）
                tab[f"{lab} β {FACTOR_NAMES[k]} 偏離 sd"] = dev.std(0)
        if f"{pre}beta_scout_org" in post:
            dev = post[f"{pre}beta_scout_org"].values.reshape(-1, len(orgs))
            x = dev + post[f"{pre}beta_scout"].values.reshape(-1, 1)
            tab[f"{lab} β scout"] = x.mean(0)
            tab[f"{lab} β scout 偏離"] = dev.mean(0)
            tab[f"{lab} β scout 偏離 sd"] = dev.std(0)
    if "theta_org" in post:
        th = post["theta_org"].values.reshape(-1, len(orgs), factors)
        for k in range(factors):
            tab[f"能力平均 {FACTOR_NAMES.get(k, k)}"] = th[:, :, k].mean(0)
    first = [c for c in tab.columns if c.startswith("升級") or c.startswith("登上時間") or c.startswith("WAR")]
    return tab.sort_values(first[0], ascending=False) if first else tab


def write_summary(
    out: Path,
    season: int,
    tag: str,
    m: pd.DataFrame,
    s_idx: np.ndarray,
    settings: dict,
    results: dict,
    lam: pd.DataFrame,
    idatas: dict,
    xnames: list,
    cmp_loo,
    cmp_waic,
    pareto_k,
    n_const: dict,
    outcome: str,
    factors: int,
    ident: list = (),
):
    """寫 summary.md。"""
    diag_rows = {}
    for t, r in results.items():
        diag_rows[t] = {
            "divergences": r["div"],
            "max r-hat（參數）": f"{r['rhat_max']:.3f} ({r['rhat_worst']})",
            "max r-hat（含 η）": round(r["rhat_eta"], 3),
            "到最大樹深": f"{r['maxdepth']:.0%}",
            "平均樹深": round(r["mean_td"], 1),
            "每迭代梯度次數": round(r["n_steps"]),
            "步長": round(r["step"], 4),
            "秒": round(r["sec"]),
        }
    ll_var = "promoted" if outcome == "promoted" else "hurdle"
    badcols = ["key_mlbam", "level_main", "age", "PA", "pick_overall", "next_status"] + (
        ["reached_fwd", "war_fwd"] if outcome == "war" else []
    )
    bad = m.iloc[s_idx[pareto_k > 0.7]][badcols]
    md = [
        f"# {season} 打者 · {tag}\n",
        f"{len(m):,} 人季，結構遮罩內 {len(s_idx):,}；draws {settings['draws']} × tune {settings['tune']} × chains {settings['chains']}；取樣器 {settings['sampler']}\n",
        "## 識別檢查\n",
        "\n".join("- " + x for x in ident) or "- 無",
        "\n## 診斷\n",
        pd.DataFrame(diag_rows).T.to_markdown(),
        f"\n## 負荷量 λ（M1，{factors} factor）\n",
        lam.round(3).to_markdown(),
    ]
    if factors > 1:
        md += [
            "\n### factor 相關\n",
            az.summary(idatas["M1"], var_names=["factor_corr"])[["mean", "sd", "eti89_lb", "eti89_ub"]]
            .round(3)
            .to_markdown(),
        ]
    md += [
        "\n### 後驗尺度表（M1）：各群參數的後驗 sd 與 ESS，看取樣器要同時遷就的尺度差距\n",
        scale_table(idatas["M1"], m).round(4).to_markdown(),
    ]
    md += ["\n## 結構模型係數\n"]
    for mode in ("M0", "M1", "Mh"):
        md += [f"### {mode}\n", struct_table(idatas[mode], xnames).round(3).to_markdown(), ""]
    md += [
        "\n## 球團（M1）：同樣的能力，哪些球團多升、早上、登上後 WAR 高；以及各球團手上球員的能力平均\n",
        org_table(idatas["M1"], m, factors).round(3).to_markdown(),
    ]
    md += [
        "\n## Model comparison（" + ll_var + " 的 pointwise log-lik）\n",
        "### PSIS-LOO\n",
        cmp_loo[["rank", "elpd", "p", "elpd_diff", "dse", "weight", "se"]].round(2).to_markdown(),
        "\n### WAIC（自算）\n",
        cmp_waic.round(2).to_markdown(),
        f"\n## Pareto-k > 0.7 的人季（M1）：{len(bad)} 筆；預測飽和的常數列 {n_const}\n",
        bad.head(30).to_markdown(index=False),
    ]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:7]))


# =====================================================================================
# 8. 主流程
# =====================================================================================
def make_tag(a) -> str:
    return (
        f"{a.outcome}_f{a.factors}{'x' if a.crossload else ''}{'s' if a.scout else ''}{'e' if a.extra else ''}{'d' if a.direct else ''}{'a' if a.avg else ''}{'w' if a.woba else ''}{'o' if a.org_slopes else ''}{'c' if a.club else ''}{'k' if a.club_type else ''}{'v' if a.verify else ''}{STATCAST_LETTER[a.statcast]}"
        f"_{a.feta}_p{a.poly}{'_nc' if a.param == 'noncentered' else ''}"
    )


def main(a):
    """a 是命令列參數（argparse 的結果）。流程：讀資料 → 對每個模式：建模、取樣（或讀回）、log-lik、對齊、診斷 → 彙整。"""
    tag = make_tag(a)
    out = Path(a.out_dir) / str(a.season) / tag
    out.mkdir(parents=True, exist_ok=True)
    draws, tune = (200, 200) if a.quick else (a.draws, a.tune)
    sampler = "pymc" if a.sampler == "auto" else a.sampler
    if a.club_type and not a.club:
        raise SystemExit("--club-type 需要 --club（母隊強弱來自 club_features）")
    m = load(Path(a.data_dir), a.season, a.club, a.statcast)
    mask = (m["structural"] if a.outcome == "promoted" else m["war_structural"]).to_numpy()
    print(
        f"{a.season} {tag}: {len(m):,} 人季，結構遮罩內 {int(mask.sum()):,}；參數化 {a.param}，init {a.init}",
        flush=True,
    )
    ident = identification_report(m, a.factors, a.statcast)
    for line in ident:
        print("  識別檢查：" + line, flush=True)

    modes = ("M0", "M1", "Mh") if a.only == "all" else (a.only,)
    idatas, results, info = {}, {}, None
    for mode in modes:
        t0 = time.time()
        nc_path = out / f"idata_{mode}.nc"
        model, info = build(
            m,
            a.outcome,
            a.factors,
            a.feta,
            a.poly,
            mode,
            a.crossload,
            a.param,
            a.scout,
            a.extra,
            a.direct,
            a.avg,
            a.woba,
            a.org_slopes,
            a.club,
            a.verify,
            a.statcast,
            a.club_type,
        )
        save = mode == "M1" or a.keep_baselines  # M0／Mh 不落地（以前是存了再刪，十個 job 同時跑時尖峰 30G 把配額撐爆）
        idata, loaded = sample_or_load(model, info, nc_path, a.reuse, a.chains, draws, tune, sampler, a.init, save=save)
        ensure_loglik(idata, model, a.outcome, m, info["s_idx"], nc_path, save=save)
        if mode in ("M0", "M1"):
            flipped = align_signs(idata, a.factors, a.feta, a.crossload, statcast=a.statcast)
            if any(flipped.values()):
                print(f"  {mode} 鏡像對齊，翻轉的 draw 數：{flipped}", flush=True)
        n_iter = idata.posterior.sizes["chain"] * (idata.posterior.sizes["draw"] + (0 if loaded else tune))
        results[mode] = diagnostics(idata, time.time() - t0, n_iter)
        idatas[mode] = idata
        print("  " + describe(mode, results[mode]), flush=True)

    if a.only != "all":
        print(
            f"-> {out}/idata_{a.only}.nc 已存；三個都跑完後用 --reuse 彙整。（{describe(a.only, results[a.only])}）",
            flush=True,
        )
        return

    s_idx = info["s_idx"]
    ll_var = "promoted" if a.outcome == "promoted" else "hurdle"
    n_const = {t: dejitter_constant_loglik(idatas[t], ll_var) for t in idatas}
    if any(n_const.values()):
        print(f"  log-lik 常數列（預測飽和）已加擾動：{n_const}", flush=True)
    lam = loading_table(idatas["M1"], a.factors, a.statcast)
    cmp_loo, cmp_waic, pareto_k = comparison_tables(idatas, a.outcome, m, s_idx)
    export_scores(idatas, m, s_idx, a.season, a.factors, a.scout, out)
    struct_table(idatas["M1"], info["xnames"]).to_csv(out / "struct_M1.csv")
    if a.verify:
        ct = construct_table(idatas["M1"], a.factors, a.statcast)
        ct.to_csv(out / "construct_M1.csv", index=False)
        print(ct.to_markdown(index=False))
    lam.to_csv(out / "lambda_M1.csv")
    org_table(idatas["M1"], m, a.factors).to_csv(out / "org_M1.csv")
    settings = {"draws": draws, "tune": tune, "chains": a.chains, "sampler": sampler}
    write_summary(
        out,
        a.season,
        tag,
        m,
        s_idx,
        settings,
        results,
        lam,
        idatas,
        info["xnames"],
        cmp_loo,
        cmp_waic,
        pareto_k,
        n_const,
        a.outcome,
        a.factors,
        ident,
    )
    if not a.keep_baselines:  # 基準模型 M0／Mh 的後驗只用來算 LOO，LOO 已寫進 summary.md；每個 .nc 幾百 MB，Arc 配額撐不住
        for mode in ("M0", "Mh"):
            nc = out / f"idata_{mode}.nc"
            if nc.exists():
                nc.unlink()
                print(f"  刪掉 {nc.name}（要留用 --keep-baselines）", flush=True)
    print(f"-> {out}")


# =====================================================================================
# 9. 命令列
# =====================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="單一球季、打者：四種能力的測量模型 + 結構模型。", formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("data_dir", help="建模表所在資料夾（含 model_bat.parquet）")
    p.add_argument("season", type=int)
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument(
        "--outcome",
        choices=["promoted", "war"],
        default="promoted",
        help="下一季升級（預設）或 WAR 兩段：幾年登上（右設限）+ WAR|登上",
    )
    p.add_argument(
        "--factors",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="1 單一能力；2 接觸(含選球)+力量；3 加速度；4 選球獨立；5 加擊球品質（LD%%、IFFB%%、BABIP）",
    )
    p.add_argument("--crossload", action="store_true", help="三振率、揮空率也載入打得遠；tag 加 x")
    p.add_argument(
        "--scout",
        action="store_true",
        help="順位與簽約金改當球探先驗 ζ 的指標；非選秀者用 undrafted，出身 HS/JC 當控制；tag 加 s",
    )
    p.add_argument(
        "--extra",
        action="store_true",
        help="加入球探可能忽略的候選：跨季進步幅度、年資、重複層級、跨層級、打席佔比；tag 加 e",
    )
    p.add_argument(
        "--org-slopes",
        dest="org_slopes",
        action="store_true",
        help="每個球團自己一組能力係數（β_0 + σ_k·z[o]）：三十隊各自的偏好側寫、哪個能力最分歧；tag 加 o",
    )
    p.add_argument(
        "--woba",
        action="store_true",
        help="M0/M1 也加 wOBA 多項式（Mh 本來就有）：LOO 比 M1 和 Mh 都好 = 能力與綜合指標各有對方沒有的資訊；tag 加 w",
    )
    p.add_argument(
        "--avg",
        action="store_true",
        help="原始打擊率（層級內標準化）直接進結構模型，跟四種能力並排；驗證「打擊率是門票」：預期升級段係數 > 0、WAR 段 ≈ 0；tag 加 a",
    )
    p.add_argument(
        "--club",
        action="store_true",
        help="階層的第二層：球團截距 = a·母隊團隊 WAR + Σ a_k·θ_org + 剩餘；個人層加母隊該守位缺口、主力 WAR、農場塞車，與缺口 × 能力 / 球探評價的跨層交互作用；需要 fg_positions.parquet；tag 加 c",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="構念驗證：每個指標都可載到定義之外的構念（先驗 N(0, 0.3)），輸出 construct_M1.csv；tag 加 v",
    )
    p.add_argument(
        "--statcast",
        choices=STATCAST_MODES,
        default="none",
        help="3A 逐球的過程量當指標。add = 成績單 + Statcast、全樣本（S）；same = 成績單、只留有 Statcast 的 3A 人季（N，only 的基準）；only = Statcast 自己組尺（+ 成績單的速度）、同一批人（O）；both = 兩個都放、同一批人（B）。要有 data/savant_minors_<season>.parquet",
    )
    p.add_argument(
        "--club-type",
        dest="club_type",
        action="store_true",
        help="球隊買平均還是分布：母隊強弱 × 型態（打得遠 − 打得到球）加每隊自己對型態的斜率；需要 --club；tag 加 k",
    )
    p.add_argument(
        "--direct",
        action="store_true",
        help="盜壘成功率、三壘打率、雙殺率、每打席投球數、保送率的原始比率直接進結構模型（能力之外的細節）；tag 加 d",
    )
    p.add_argument(
        "--feta", choices=["linear", "quad", "cubic", "spline"], default="linear", help="能力進結構模型的函數形式"
    )
    p.add_argument("--poly", type=int, choices=[1, 2, 3], default=2, help="共變數的正交多項式次數")
    p.add_argument(
        "--param", choices=["centered", "noncentered"], default="centered", help="η 的參數化；非中心化 tag 加 _nc"
    )
    p.add_argument(
        "--init", choices=["advi", "grad", "diag"], default="advi", help="質量矩陣的起始：advi（預設）/ grad / diag"
    )
    p.add_argument(
        "--only", choices=["all", "M0", "M1", "Mh"], default="all", help="只跑一個模型並存 idata，之後用 --reuse 彙整"
    )
    p.add_argument("--quick", action="store_true", help="200 暖身 + 200 取樣")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument(
        "--sampler", choices=["auto", "pymc", "numpyro"], default="auto", help="pymc 純 Python；numpyro 走 JAX"
    )
    p.add_argument("--reuse", action="store_true", help="idata_*.nc 已存在就讀回，不重新取樣")
    p.add_argument("--keep-baselines", dest="keep_baselines", action="store_true", help="跑完不要刪 M0／Mh 的 idata（預設刪，省磁碟）")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
