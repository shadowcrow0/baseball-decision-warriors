"""球隊的升人決策像哪一種分類學習者？用同一組輸入配五種模型，用過去估未來，比誰最準（不取樣）。

    python src/decision_style.py outputs/ --promoted promoted_f4sed_linear_p2 [--data data/] [--seasons 2012 2015 2019]

輸入：每人季的四種能力分數、球探評價、年齡、打席（層級內標準化），結果 = 隔年升不升。
六種模型（都在層級內各自配，因為每個層級的門檻本來就不同）：
    單一維度規則    在一個維度上找一個門檻，門檻兩邊各一個升級機率（決策樹樁）
    單一維度線性    同一個維度、換成平滑的邏輯迴歸：跟樹樁比 = 門檻這個形狀值多少；跟資訊整合比 = 其他變數值多少
    兩個規則（合取）  深度二的決策樹：先一個門檻，門檻兩邊各再一個門檻
    資訊整合        邏輯迴歸（線性邊界；就是我們的結構模型的簡化版）
    Exemplar 城市街區  相似度 = exp(−c · Σ w_k |Δ_k|)，對過去三季的每一個人加權平均他有沒有被升；w_k 是注意力
    Exemplar 歐氏      相似度 = exp(−c · sqrt(Σ w_k Δ_k²))
訓練 = 前三季（t−3 到 t−1），測試 = 第 t 季；報每季每種模型的 log-loss 與 AUC，和 exemplar 的注意力權重、規則選的維度。
再把測試季的 log-loss 依球團分開：哪一隊的決策最像哪一種學習者。
有 --data 時多一張表：打擊率 .300 上下升級率有沒有跳（整數門檻 = 規則的痕跡）。
輸出 outputs/decision_style_<tag>.md。
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURES = ["eta_contact", "eta_power", "eta_speed", "eta_discipline", "zeta_scout", "age", "log_pa"]
PRE_FEATURES = [
    "log_pick",
    "log_bonus_f",
    "bonus_known",
    "lvl_HS",
    "lvl_JC",
    "age",
    "pos_C",
    "pos_SS2B",
    "pos_OF",
    "pos_1BDH",
    "pos_3B",
]
POST_FEATURES = [
    "eta_contact",
    "eta_power",
    "eta_speed",
    "eta_discipline",
    "age",
    "log_pa",
]  # 沒有球探評價（那是順位變來的）
FEATURE_SETS = {
    "post": POST_FEATURES,
    "pre": PRE_FEATURES,
    "both": PRE_FEATURES + ["eta_contact", "eta_power", "eta_speed", "eta_discipline", "log_pa"],
}
FEATURE_NAMES = {
    "eta_contact": "打得到球",
    "eta_power": "打得遠",
    "eta_speed": "跑得快",
    "eta_discipline": "選得到球",
    "zeta_scout": "球探評價",
    "age": "年齡",
    "log_pa": "打席（log）",
    "log_pick": "順位（log）",
    "log_bonus_f": "簽約金（log）",
    "bonus_known": "有簽約金資料",
    "lvl_HS": "高中出身",
    "lvl_JC": "專科出身",
    "pos_C": "捕手",
    "pos_SS2B": "游擊/二壘",
    "pos_OF": "外野",
    "pos_1BDH": "一壘/指定",
    "pos_3B": "三壘",
}
RAW_RATES = {
    "AVG": "打擊率",
    "OBP": "上壘率",
    "Kpct": "三振率",
    "whiffpct": "揮空率",
    "XBH_H": "長打比",
    "HR_AB": "全壘打率",
    "Pullpct": "拉打率",
    "GBpct": "滾地率",
    "SBpct": "盜壘成功率",
    "SBA_OBT": "盜壘嘗試率",
    "3B_H": "三壘打率",
    "GDP_AB": "雙殺率",
    "BBpct": "保送率",
    "Ballpct": "壞球率",
    "P_PA": "每打席看球數",
}
RAW_FEATURES = ["rate_" + k for k in RAW_RATES] + ["age", "log_pa"]
FEATURE_NAMES.update({"rate_" + k: v for k, v in RAW_RATES.items()})
LEVELS = ["R/CPX", "A-", "A", "A+", "AA", "AAA"]
MODELS = ["單一維度規則", "單一維度線性", "兩個規則", "資訊整合", "Exemplar 城市街區", "Exemplar 歐氏"]


# ---------- 讀資料 ----------
def load_scores(out: Path, tag: str, data: Path | None = None, min_pa: int = 100) -> pd.DataFrame:
    frames = []
    for p in sorted(out.glob(f"20*/{tag}/eta_M1.parquet")):
        e = pd.read_parquet(p)
        e = e[(e["PA"] >= min_pa) & e["promoted"].notna() & e["level_main"].isin(LEVELS)].copy()
        e["zeta_scout"] = e["zeta_scout"].fillna(0.0)
        e["log_pa"] = np.log(e["PA"])
        e["club"] = e["org"].fillna("UNK").astype(str).str.split("/").str[0]
        frames.append(e)
    d = pd.concat(frames, ignore_index=True)
    if data is not None:  # 進職業前的資訊：選秀、出身、守位
        m = pd.read_parquet(
            data / "model_bat.parquet", columns=["key_mlbam", "season", "draft_level", "log_bonus", "bonus_known"]
        )
        pos = pd.read_parquet(
            data / "minors_bat_clean.parquet", columns=["key_mlbam", "season", "positionDB"]
        ).drop_duplicates(["key_mlbam", "season"])
        d = d.merge(m, on=["key_mlbam", "season"], how="left").merge(pos, on=["key_mlbam", "season"], how="left")
        d["log_pick"] = np.log(d["pick_overall"].astype(float))
        d["log_bonus_f"] = d["log_bonus"].fillna(0.0)
        d["bonus_known"] = d["bonus_known"].fillna(False).astype(float)
        d["lvl_HS"] = (d["draft_level"] == "HS").astype(float)
        d["lvl_JC"] = (d["draft_level"] == "JC").astype(float)
        p = d["positionDB"].fillna("").astype(str)
        d["pos_C"] = p.str.contains("C").astype(float)
        d["pos_SS2B"] = (p.str.contains("SS") | p.str.contains("2B")).astype(float)
        d["pos_OF"] = p.str.contains("OF").astype(float)
        d["pos_1BDH"] = (p.str.contains("1B") | p.str.contains("DH")).astype(float)
        d["pos_3B"] = p.str.contains("3B").astype(float)
        counts = pd.read_parquet(data / "model_bat.parquet")
        from fit_year import INDICATORS, indicator_tag

        need = sorted({c for _, a, b, _, _ in INDICATORS for c in (a, b)} - set(d.columns))  # PA 已經有了，不重複接
        d = d.merge(counts[["key_mlbam", "season"] + need], on=["key_mlbam", "season"], how="left")
        for name, num, den, _, _ in INDICATORS:  # 原始比率：分母太小的人季當缺值（之後補 0 = 層級平均）
            d["rate_" + indicator_tag(name)] = d[num] / d[den].where(d[den] >= 30)
    # 層級內、球季內標準化：每個模型看到的尺度一樣
    cols = [c for c in set(FEATURES + PRE_FEATURES + RAW_FEATURES) if c in d.columns]
    for c in cols:
        d[c] = d.groupby(["season", "level_main"])[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
    d["y"] = d["promoted"].astype(float)
    return d


# ---------- 共同工具 ----------
def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def leaf_prob(y: np.ndarray, prior: float, a: float = 2.0) -> float:
    """葉子的升級機率，加一點先驗避免 0 和 1。"""
    return (y.sum() + a * prior) / (len(y) + a)


# ---------- 規則模型 ----------
def best_stump(X: np.ndarray, y: np.ndarray, prior: float):
    """在所有維度、所有門檻裡找訓練 log-lik 最大的一個切法。回傳 (維度, 門檻, 左機率, 右機率)。"""
    best = (None, None, prior, prior, -np.inf)
    for k in range(X.shape[1]):
        xs = X[:, k]
        for q in np.linspace(0.05, 0.95, 37):
            t = np.quantile(xs, q)
            left = xs <= t
            if left.sum() < 20 or (~left).sum() < 20:
                continue
            pl, pr = leaf_prob(y[left], prior), leaf_prob(y[~left], prior)
            p = np.where(left, pl, pr)
            ll = -log_loss(y, p) * len(y)
            if ll > best[4]:
                best = (k, t, pl, pr, ll)
    return best[:4]


def stump_predict(rule, X: np.ndarray, prior: float) -> np.ndarray:
    k, t, pl, pr = rule
    if k is None:
        return np.full(len(X), prior)
    return np.where(X[:, k] <= t, pl, pr)


def fit_tree2(X: np.ndarray, y: np.ndarray, prior: float):
    """深度二：先一個樁，左右各再一個樁。"""
    root = best_stump(X, y, prior)
    k, t, _, _ = root
    if k is None:
        return root, None, None
    left = X[:, k] <= t
    return root, best_stump(X[left], y[left], prior), best_stump(X[~left], y[~left], prior)


def tree2_predict(tree, X: np.ndarray, prior: float) -> np.ndarray:
    root, lt, rt = tree
    k, t, _, _ = root
    if k is None:
        return np.full(len(X), prior)
    left = X[:, k] <= t
    p = np.empty(len(X))
    p[left] = stump_predict(lt, X[left], prior)
    p[~left] = stump_predict(rt, X[~left], prior)
    return p


# ---------- 線性整合 ----------
def fit_logit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-2, iters: int = 50) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Xc.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xc @ b))
        w = p * (1 - p)
        H = Xc.T @ (Xc * w[:, None]) + ridge * np.eye(Xc.shape[1])
        step = np.linalg.solve(H, Xc.T @ (y - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def logit_predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ b)))


# ---------- Exemplar（GCM）----------
def gcm_predict(
    params: np.ndarray,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    metric: str,
    prior: float,
    exclude_self: bool = False,
) -> np.ndarray:
    """params = [log c, 注意力的 logits...]。相似度加權的升級比例，加一點先驗。"""
    c = np.exp(params[0])
    w = np.exp(params[1:])
    w = w / w.sum()
    diff = Xte[:, None, :] - Xtr[None, :, :]
    if metric == "city":
        dist = (np.abs(diff) * w).sum(axis=2)
    else:
        dist = np.sqrt((diff**2 * w).sum(axis=2))
    s = np.exp(-c * dist)
    if exclude_self:
        np.fill_diagonal(s, 0.0)
    return (s @ ytr + 2.0 * prior) / (s.sum(axis=1) + 2.0)


def fit_gcm(
    Xtr: np.ndarray, ytr: np.ndarray, metric: str, prior: float, max_fit: int = 2500, seed: int = 0
) -> np.ndarray:
    """用訓練集自己（留一法）配 c 與注意力；人太多時抽一部分來配。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(max_fit, len(Xtr)), replace=False)
    Xs, ys = Xtr[idx], ytr[idx]

    def loss(params):
        p = gcm_predict(params, Xs, ys, Xs, metric, prior, exclude_self=True)
        return log_loss(ys, p)

    # 從全零出發時 Nelder-Mead 的初始步長只有 0.00025，權重根本不會動；自己給一個夠大的初始單純形
    k = Xtr.shape[1] + 1
    x0 = np.zeros(k)
    simplex = np.vstack([x0] + [x0 + 0.8 * np.eye(k)[i] for i in range(k)])
    res = minimize(
        loss,
        x0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "maxiter": 1500, "xatol": 1e-2, "fatol": 1e-5, "adaptive": True},
    )
    return res.x


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_and_disagree(res: pd.DataFrame) -> dict:
    """兩個檢驗：
    疊加：y ~ a·logit(線性) + b·logit(exemplar)，兩折交叉配適（一半估係數、另一半算 log-loss，再對調）。
        b 明顯不為零 = exemplar 在線性之外還有資訊；疊加後的 log-loss 比線性低多少 = 多出來的資訊值多少。
    意見不同：線性與 exemplar 預測差最大的一成的人，各自的 log-loss 誰低 = 球隊最後聽誰的。"""
    y = res["y"].to_numpy()
    pl = res["資訊整合"].to_numpy()
    pe = res["Exemplar 歐氏"].to_numpy()
    out = {}
    X = np.column_stack([logit(pl), logit(pe)])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 2, len(y))
    pred = np.empty(len(y))
    bs = []
    for f in (0, 1):
        b = fit_logit(X[fold != f], y[fold != f])
        pred[fold == f] = logit_predict(b, X[fold == f])
        bs.append(b)
    b = np.mean(bs, axis=0)
    out["疊加：線性係數"] = round(float(b[1]), 2)
    out["疊加：exemplar 係數"] = round(float(b[2]), 2)
    out["疊加比線性少的 log-loss"] = round(log_loss(y, pl) - log_loss(y, pred), 4)
    gap = np.abs(pl - pe)
    top = gap >= np.quantile(gap, 0.9)
    out["意見不同的一成：線性 log-loss"] = round(log_loss(y[top], pl[top]), 3)
    out["意見不同的一成：exemplar log-loss"] = round(log_loss(y[top], pe[top]), 3)
    out["意見不同的一成：實際升級率"] = round(float(y[top].mean()), 2)
    out["意見不同的一成：線性平均"] = round(float(pl[top].mean()), 2)
    out["意見不同的一成：exemplar 平均"] = round(float(pe[top].mean()), 2)
    return out


# ---------- 一季：訓練前三季、測試這一季 ----------
def run_season(
    d: pd.DataFrame, season: int, train_seasons: list, features: list = FEATURES
) -> tuple[pd.DataFrame, dict]:
    tr = d[d["season"].isin(train_seasons)]
    te = d[d["season"] == season]
    pred = {m: np.full(len(te), np.nan) for m in MODELS}
    notes = {"規則維度": [], "注意力 城市街區": [], "注意力 歐氏": [], "線性係數": []}
    te_index = te.index.to_numpy()
    for lvl in LEVELS:
        mtr = tr["level_main"].to_numpy() == lvl
        mte = te["level_main"].to_numpy() == lvl
        if mtr.sum() < 60 or mte.sum() < 10:
            continue
        Xtr = tr.loc[mtr, features].to_numpy(dtype=float)
        ytr = tr.loc[mtr, "y"].to_numpy(dtype=float)
        Xte = te.loc[mte, features].to_numpy(dtype=float)
        prior = float(ytr.mean())
        rule = best_stump(Xtr, ytr, prior)
        pred["單一維度規則"][mte] = stump_predict(rule, Xte, prior)
        notes["規則維度"].append(f"{lvl}:{FEATURE_NAMES[features[rule[0]]] if rule[0] is not None else '無'}")
        if rule[0] is not None:  # 同一個維度、換成平滑的線性：把「變數少」和「門檻形狀」分開
            k = rule[0]
            b1 = fit_logit(Xtr[:, [k]], ytr)
            pred["單一維度線性"][mte] = logit_predict(b1, Xte[:, [k]])
        else:
            pred["單一維度線性"][mte] = prior
        tree = fit_tree2(Xtr, ytr, prior)
        pred["兩個規則"][mte] = tree2_predict(tree, Xte, prior)
        b = fit_logit(Xtr, ytr)
        pred["資訊整合"][mte] = logit_predict(b, Xte)
        notes["線性係數"].append(b[1:])
        for metric, name, key in (
            ("city", "Exemplar 城市街區", "注意力 城市街區"),
            ("euclid", "Exemplar 歐氏", "注意力 歐氏"),
        ):
            params = fit_gcm(Xtr, ytr, metric, prior)
            pred[name][mte] = gcm_predict(params, Xtr, ytr, Xte, metric, prior)
            w = np.exp(params[1:])
            w = w / w.sum()
            notes[key].append(w)
    out = te[["season", "level_main", "club", "y"]].copy()
    for m in MODELS:
        out[m] = pred[m]
    out = out.dropna(subset=MODELS)
    summary = {"規則維度": "、".join(notes["規則維度"])}
    top = 5 if len(features) > 8 else len(features)  # 原始數字很多時只印前幾名
    for key in ("注意力 城市街區", "注意力 歐氏"):
        if notes[key]:
            w = np.mean(notes[key], axis=0)
            order = np.argsort(-w)[:top]
            summary[key] = "、".join(f"{FEATURE_NAMES[features[i]]} {w[i]:.2f}" for i in order)
    if notes["線性係數"]:
        b = np.mean(notes["線性係數"], axis=0)
        order = np.argsort(-np.abs(b))[:top]
        summary["線性係數（|大|前幾名）"] = "、".join(f"{FEATURE_NAMES[features[i]]} {b[i]:+.2f}" for i in order)
    return out, summary


# ---------- 打擊率 .300 的門檻痕跡 ----------
def avg_bunching(data: Path, out: Path, tag: str) -> pd.DataFrame:
    m = pd.read_parquet(data / "model_bat.parquet", columns=["key_mlbam", "season", "H", "AB", "promoted"])
    frames = []
    for p in sorted(out.glob(f"20*/{tag}/eta_M1.parquet")):
        e = pd.read_parquet(p)[["key_mlbam", "season", "score_promoted"]]
        frames.append(e)
    e = pd.concat(frames).merge(m, on=["key_mlbam", "season"])
    e = e[(e["AB"] >= 250) & e["score_promoted"].notna()]
    e["avg"] = e["H"] / e["AB"]
    e["resid"] = e["promoted"].astype(float) - 1 / (1 + np.exp(-e["score_promoted"]))
    bins = np.arange(0.270, 0.3301, 0.005)
    e["bin"] = pd.cut(e["avg"], bins, right=False)
    g = (
        e.groupby("bin", observed=True)
        .agg(人數=("resid", "size"), 升級率=("promoted", "mean"), 模型之外的殘差=("resid", "mean"))
        .round(3)
    )
    g.index = [f"{b.left:.3f}–{b.right:.3f}" for b in g.index]
    return g.reset_index().rename(columns={"index": "打擊率區間"})


SHARED = {}  # 行程共用的資料（fork 之後子行程直接讀，不用複製）


def one_test_season(args):
    """一個測試季的全部工作（給行程池用）：五種模型，加上有 --data 時的三組資訊比較。"""
    s, train, with_sets = args
    d = SHARED["d"]
    res, summary = run_season(d, s, train, SHARED.get("features", FEATURES))
    sets = {}
    if with_sets:
        dd = d[d["pick_overall"].notna()]
        for name, feats in FEATURE_SETS.items():
            r, _ = run_season(dd, s, train, feats)
            if not r.empty:
                y = r["y"].to_numpy()
                sets[name] = ({m: log_loss(y, r[m].to_numpy()) for m in MODELS}, len(r))
    return s, train, res, summary, sets


def linear_forward(d: pd.DataFrame, jobs: list, features: list, max_steps: int = 8) -> pd.DataFrame:
    """線性整合從零開始，一次加一個「所有測試季合併最有用」的數字，加到不再變準。回答：整合者要幾個數字、哪幾個。"""

    def pooled_loss(feats):
        losses = []
        for s, train, _ in jobs:
            tr = d[d["season"].isin(train)]
            te = d[d["season"] == s]
            for lvl in LEVELS:
                a = tr[tr["level_main"] == lvl]
                b = te[te["level_main"] == lvl]
                if len(a) < 60 or len(b) < 10:
                    continue
                if feats:
                    coef = fit_logit(a[feats].to_numpy(dtype=float), a["y"].to_numpy(dtype=float))
                    p = logit_predict(coef, b[feats].to_numpy(dtype=float))
                else:
                    p = np.full(len(b), a["y"].mean())
                losses.append((log_loss(b["y"].to_numpy(dtype=float), p), len(b)))
        return float(np.average([l for l, _ in losses], weights=[n for _, n in losses]))

    chosen, rows = [], []
    current = pooled_loss([])
    for _ in range(max_steps):
        best, best_loss = None, current
        for f in features:
            if f in chosen:
                continue
            loss = pooled_loss(chosen + [f])
            if loss < best_loss - 1e-4:
                best, best_loss = f, loss
        if best is None:
            break
        chosen.append(best)
        rows.append(
            {
                "步驟": len(chosen),
                "加進去的數字": FEATURE_NAMES[best],
                "合併 log-loss": round(best_loss, 4),
                "比上一步少": round(current - best_loss, 4),
            }
        )
        current = best_loss
    return pd.DataFrame(rows)


def main(out_dir: str, tag: str, data_dir: str | None, seasons: list | None, workers: int = 1, raw: bool = False):
    out = Path(out_dir)
    d = load_scores(out, tag, Path(data_dir) if data_dir else None)
    SHARED["d"] = d
    SHARED["features"] = RAW_FEATURES if raw else FEATURES
    suffix = "_raw" if raw else ""
    all_seasons = sorted(d["season"].unique())
    tests = seasons or [s for s in all_seasons if any(t in all_seasons for t in range(s - 3, s))]
    rows, per_club, notes_rows, set_rows = [], [], [], []
    jobs = []
    for s in tests:
        train = [t for t in range(s - 3, s) if t in all_seasons]
        if train:
            jobs.append((s, train, data_dir is not None))
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            results = pool.map(one_test_season, jobs)
    else:
        results = [one_test_season(j) for j in jobs]
    for s, train, res, summary, sets in results:
        if res.empty:
            continue
        if sets:
            row = {"測試季": s}
            for name, (ll, n) in sets.items():
                label = {"pre": "進職業前", "post": "職業後成績單", "both": "合併"}[name]
                best = min(ll, key=ll.get)
                row[f"{label} 整合"] = round(ll["資訊整合"], 4)
                row[f"{label} 最準"] = f"{best} {ll[best]:.4f}"
                row["n"] = n
            set_rows.append(row)
        y = res["y"].to_numpy()
        row = {"測試季": s, "訓練季": "、".join(map(str, train)), "n": len(res)}
        row.update(stack_and_disagree(res))
        for m in MODELS:
            row[f"{m} log-loss"] = round(log_loss(y, res[m].to_numpy()), 4)
        for m in MODELS:
            row[f"{m} AUC"] = round(auc(y, res[m].to_numpy()), 3)
        rows.append(row)
        notes_rows.append({"測試季": s, **summary})
        per_club.append(res)
        print(f"{s}: " + "  ".join(f"{m} {row[f'{m} log-loss']:.4f}" for m in MODELS), flush=True)
    tab = pd.DataFrame(rows)
    wins = (
        {m: int((tab[[f"{k} log-loss" for k in MODELS]].idxmin(axis=1) == f"{m} log-loss").sum()) for m in MODELS}
        if len(tab)
        else {}
    )
    md = [
        f"# 球隊的升人決策像哪一種分類學習者 · {tag}\n",
        "同一組輸入（四種能力、球探評價、年齡、打席；層級內標準化），五種模型在層級內各自配，訓練前三季、測試當季。log-loss 越低越準。\n",
        "## 一、每季的預測準度\n",
        tab[["測試季", "訓練季", "n"] + [f"{m} log-loss" for m in MODELS]].to_markdown(index=False),
        "",
        "AUC：",
        "",
        tab[["測試季"] + [f"{m} AUC" for m in MODELS]].to_markdown(index=False),
        "",
        "哪種模型最準的季數：" + "、".join(f"{m} {v}" for m, v in wins.items()),
        "",
        "## 一之二、exemplar 在線性之外有沒有額外資訊\n",
        "疊加 = 把線性與 exemplar（歐氏）的預測一起放進一條邏輯迴歸（兩折交叉配適）。exemplar 係數明顯不為零、而且疊加後 log-loss 比線性低，才算有額外資訊。意見不同的一成 = 兩個模型預測差最大的人：看球隊實際升了幾成、誰猜得準。\n",
        tab[["測試季"] + [c for c in tab.columns if c.startswith("疊加") or c.startswith("意見不同")]].to_markdown(
            index=False
        ),
        "",
        "## 二、規則選的維度與 exemplar 的注意力權重（各層級平均）\n",
        pd.DataFrame(notes_rows).to_markdown(index=False),
        "",
    ]
    if raw:
        md += [
            "## 二之二、整合者從零開始要幾個數字（線性整合的前向選擇，所有測試季合併）\n",
            linear_forward(d, jobs, RAW_FEATURES).to_markdown(index=False),
            "",
        ]
    if per_club:
        allc = pd.concat(per_club, ignore_index=True)
        allc = allc[allc["club"] != "UNK"]
        rows = []
        for club, g in allc.groupby("club"):
            if len(g) < 300:
                continue
            y = g["y"].to_numpy()
            ll = {m: log_loss(y, g[m].to_numpy()) for m in MODELS}
            best = min(ll, key=ll.get)
            rows.append(
                {
                    "球團": club,
                    "決策數": len(g),
                    **{m: round(v, 4) for m, v in ll.items()},
                    "最像": best,
                    "整合 − 最佳": round(ll["資訊整合"] - ll[best], 4),
                }
            )
        ct = pd.DataFrame(rows).sort_values("整合 − 最佳", ascending=False) if rows else pd.DataFrame()
        if len(ct):
            md += [
                "## 三、每一隊的決策最像哪一種學習者（測試季合併，log-loss）\n",
                "「整合 − 最佳」= 線性整合比該隊最準的模型差多少；大於 0.005 的隊，線性模型描述不好它的決策。\n",
                ct.to_markdown(index=False),
                "",
                "最像的分布：" + "、".join(f"{k} {v}" for k, v in ct["最像"].value_counts().items()),
                "",
            ]
    if data_dir:
        md += [
            "## 四、打擊率 .300 的門檻痕跡（AB ≥ 250，所有球季合併）\n",
            "若球隊用「過三成」這種規則，.300 上下的升級率會跳，而且模型之外的殘差在 .300 正上方為正、正下方為負。\n",
            avg_bunching(Path(data_dir), out, tag).to_markdown(index=False),
            "",
        ]
    text = "\n".join(md)
    (out / f"decision_style_{tag}{suffix}.md").write_text(text)
    print(text)
    print(f"-> {out / f'decision_style_{tag}{suffix}.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="球隊的升人決策像哪一種分類學習者")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("--promoted", required=True, help="升級模型的 tag")
    p.add_argument("--data", default=None, help="資料夾（含 model_bat.parquet）；給了才做打擊率 .300 的門檻表")
    p.add_argument("--seasons", nargs="*", type=int, default=None, help="只測這些季（預設全部有前三季的）")
    p.add_argument("--workers", type=int, default=1, help="同時算幾個球季（Arc 上給 8）")
    p.add_argument("--raw", action="store_true", help="用十五個原始比率當輸入，不用四把尺（需要 --data）")
    a = p.parse_args()
    if a.raw and not a.data:
        raise SystemExit("--raw 需要 --data")
    main(a.out_dir, a.promoted, a.data, a.seasons, a.workers, a.raw)
