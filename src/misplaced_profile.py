"""錯放名單比對：模型猜錯的人身上有什麼共同點（找新的候選變數，不取樣）。

    python src/misplaced_profile.py data/ <season> outputs/ --promoted <tag> --war <tag> [--score exp_war|score_promoted]

四群人（同層級內，「模型看好」= 依期望 WAR 排前 k 名，k = 該層級球團實際升的人數）：
    錯放成功    模型看好、球團沒升、後來登上且六季 WAR > 0 ：球團漏看了什麼
    升錯        模型看好、球團升了、後來沒登上或 WAR ≤ 0   ：模型和球團都漏看了什麼
    球團獨具眼光  模型不看好、球團升了、後來成功            ：球團看到模型沒看到的
    對照        模型看好、球團升了、後來成功
對每個候選欄位（原始比率、進步幅度、年資、出身、順位、使用方式、守位）算層級內標準化後的群平均，與對照組比。

第三張表：交互作用篩選，候選兩兩相乘後與兩種殘差的相關（三振率配上什麼才有用）。
第二張表：殘差相關。登上者的 WAR 殘差、登上與否（模型機率分位內校準）、升級與否（同），各與每個候選欄位的 Spearman 相關。
跟真相（WAR 或登上）有關但跟升級無關的 = 被放掉；只跟升級有關的 = 球團自己的偏好。
輸出 outputs/<season>/misplaced_<promoted tag>_vs_<war tag>.md。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cli import read_args
from club_features import add_position, add_club_strength  # noqa: F401  共用：守位與母隊強度
from fit_year import INDICATORS, add_extra, indicator_tag  # 只借規格與衍生欄位，不建模型

FACTORS = ["contact", "power", "speed", "discipline"]
# 候選欄位：(顯示名, 欄位) ；比率欄位在 build_candidates 裡算出來
EXTRA_COLS = [
    ("三振率進步", "d_k"),
    ("全壘打率進步", "d_hr"),
    ("保送率進步", "d_bb"),
    ("有上一季", "has_prev"),
    ("重複層級", "repeat_level"),
    ("跨層級", "multi_level"),
    ("年資", "years_in"),
    ("打席佔比", "pa_share"),
    ("犧牲觸擊率", "sh_rate"),
    ("年齡", "age"),
    ("相對年齡", "age_rel"),
    ("打席數", "PA"),
    ("順位（log）", "log_pick"),
    ("簽約金（log）", "log_bonus"),
    ("高中出身", "lvl_HS"),
    ("專科出身", "lvl_JC"),
    ("非選秀", "undrafted"),
]


POSITIONS = [
    ("守位：捕手", "C"),
    ("守位：游擊/二壘", "SS2B"),
    ("守位：外野", "OF"),
    ("守位：一壘/指定", "1BDH"),
    ("守位：三壘", "3B"),
    ("守幾個位置", "n_positions"),
    ("純指定打擊", "DH_only"),
    ("母隊強度（當季團隊 WAR）", "club_war_t"),
    ("母隊強度（隔季）", "club_war_t1"),
]


def position_by_bat(df: pd.DataFrame) -> pd.DataFrame:
    """守位 × 打擊能力（打得到球 + 打得遠 的平均，層級內分三等）→ 人數、登上比例、六季 WAR 平均。
    回答「守備型守位的人，棒子還有沒有差」：門檻低不等於棒子不算。"""
    d = df[df["war_observable"].astype(bool)].copy()
    bat = (d["eta_contact"] + d["eta_power"]) / 2
    d["bat_tier"] = bat.groupby(d["level_main"]).transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["低", "中", "高"])
    )
    rows = []
    for (pos, tier), g in d.groupby(["pos_group", "bat_tier"], observed=True):
        rows.append(
            {
                "守位": pos,
                "打擊能力": tier,
                "人數": len(g),
                "登上比例": g["reached_fwd"].mean(),
                "六季 WAR ≥ 2 的比例": (g["war_fwd"] >= 2).mean(),
                "六季 WAR ≥ 5 的比例": (g["war_fwd"] >= 5).mean(),
                "每百人 WAR 總和": g["war_fwd"].sum() / len(g) * 100,
            }
        )
    out = pd.DataFrame(rows)
    out["_o"] = out["打擊能力"].map({"低": 0, "中": 1, "高": 2})
    return out.sort_values(["守位", "_o"]).drop(columns="_o").round(2)


def prob_superiority(a: pd.Series, b: pd.Series) -> float:
    """隨機抽一個 a 群的人，比隨機抽一個 b 群的人高的機率（兩個分布的重疊程度；0.5 = 沒差）。
    不用平均：兩群各自有高有矮，這個數字只看排序。"""
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    ra = ranks[: len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2
    return float(u / (len(a) * len(b)))


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """排序相關與樣本數；缺值略過。"""
    ok = pd.notna(x) & pd.notna(y)
    if ok.sum() < 10 or pd.Series(x[ok]).nunique() < 2:
        return np.nan, int(ok.sum())
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())), int(ok.sum())


def build_candidates(m: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """建模表 → 候選欄位（原始值），回傳 (表, [(顯示名, 欄位), ...])。"""
    m = m.copy()
    names = []
    for name, num, den, _, _ in INDICATORS:
        col = "rate_" + indicator_tag(name)
        m[col] = m[num] / m[den].replace(0, np.nan)
        names.append((name, col))
    m["lvl_HS"] = (m["draft_level"] == "HS").astype(float)
    m["lvl_JC"] = (m["draft_level"] == "JC").astype(float)
    m["undrafted"] = m["pick_overall"].isna().astype(float)
    for disp, col in EXTRA_COLS + POSITIONS:
        if col in m.columns:
            names.append((disp, col))
    return m, names


def standardize_within_level(m: pd.DataFrame, cols: list) -> pd.DataFrame:
    """每個欄位在層級內減平均除標準差；缺值保留為 NaN（群平均時自動略過）。"""
    z = pd.DataFrame(index=m.index)
    for col in cols:
        x = pd.to_numeric(m[col], errors="coerce")
        mu = x.groupby(m["level_main"]).transform("mean")
        sd = x.groupby(m["level_main"]).transform("std")
        z[col] = ((x - mu) / (sd + 1e-9)).clip(-4, 4)
    return z


def model_pick(df: pd.DataFrame, score: str) -> pd.Series:
    """同層級內依 score 排前 k 名，k = 球團實際升的人數。"""
    pick = pd.Series(False, index=df.index)
    for lvl, g in df.groupby("level_main"):
        k = int(g["promoted"].sum())
        if k > 0:
            pick.loc[g.nlargest(k, score).index] = True
    return pick


def assign_groups(df: pd.DataFrame, score: str) -> pd.Series:
    """四群：錯放成功、升錯、球團獨具眼光、對照；其餘為空字串。"""
    picked = model_pick(df, score)
    promoted = df["promoted"].astype(bool)
    success = df["reached_fwd"].astype(bool) & (df["war_fwd"] > 0)
    group = pd.Series("", index=df.index)
    group[picked & ~promoted & success] = "錯放成功"
    group[picked & promoted & ~success] = "升錯"
    group[~picked & promoted & success] = "球團獨具眼光"
    group[picked & promoted & success] = "對照"
    return group


def group_profile(z: pd.DataFrame, group: pd.Series, names: list) -> pd.DataFrame:
    """每群在每個候選欄位上的中位數，加「隨機抽一人比對照組高的機率」（0.5 = 沒差）；依離 0.5 最遠排序。
    不用群平均：一群人裡有高有矮，平均會把他們壓成一個人。"""
    order = ["錯放成功", "升錯", "球團獨具眼光", "對照"]
    rows = []
    for disp, col in names:
        row = {"候選": disp}
        ref = z.loc[group == "對照", col]
        for gname in order:
            row[gname + " 中位"] = z.loc[group == gname, col].median()
        for gname in order[:3]:
            row[f"P({gname} > 對照)"] = prob_superiority(z.loc[group == gname, col], ref)
        row["最遠離 0.5"] = max(
            abs(row[f"P({g} > 對照)"] - 0.5) for g in order[:3] if np.isfinite(row[f"P({g} > 對照)"])
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("最遠離 0.5", ascending=False).round(2)


def binned_residual(y: pd.Series, p: pd.Series, bins: int = 20) -> pd.Series:
    """0/1 結果的殘差，不用 y − p：y 是 0 的人殘差 = −p，排序會被模型機率機械式地決定。
    改成「同一個模型機率十分位內，y 減該分位的平均」，分位內 p 幾乎一樣，剩下的排序才是候選欄位的訊息。"""
    ok = p.notna() & y.notna()
    out = pd.Series(np.nan, index=y.index)
    dec = pd.qcut(p[ok].rank(method="first"), bins, labels=False)
    yy = y[ok].astype(float)
    out[ok] = yy - yy.groupby(dec).transform("mean")
    return out


def residuals(df: pd.DataFrame) -> dict:
    """三種殘差，各自避開零堆積與 0/1 的機械排序：
    war    登上者的 asinh(六季 WAR) 減模型的條件期望（score_war_asinh）；沒登上的人為 NaN
    reach  登上與否，依模型登上機率十分位校準後的殘差
    prom   球團升不升，依模型升級機率十分位校準後的殘差"""
    reached = df["reached_fwd"].astype(bool) & df["war_observable"].astype(bool) & df["score_war_asinh"].notna()
    y = np.arcsinh(df.loc[reached, "war_fwd"].to_numpy(dtype=float))
    X = np.column_stack([np.ones(reached.sum()), df.loc[reached, "score_war_asinh"]])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r_war = pd.Series(np.nan, index=df.index)
    r_war[reached] = y - X @ b
    obs = df["war_observable"].astype(bool)
    r_reach = binned_residual(df["reached_fwd"].astype(float).where(obs), df["score_preach"])
    p_prom = 1.0 / (1.0 + np.exp(-df["score_promoted"]))  # score_promoted 是 logit
    r_prom = binned_residual(df["promoted"].astype(float), p_prom)
    return {"war": r_war, "reach": r_reach, "prom": r_prom}


def logistic_irls(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """邏輯迴歸（IRLS）。回傳 (係數, 標準誤)。"""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    w = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def beyond_model(df: pd.DataFrame, z: pd.DataFrame, col: str) -> dict:
    """「模型分數之外，這個候選還多說了什麼」：
        升級     logit(升不升) ~ 模型 logit + 候選        → 候選的係數與 z 值
        登上     logit(登上與否) ~ logit(模型登上機率) + 候選
        WAR|登上 asinh(六季 WAR) ~ 模型條件期望 + 候選（最小平方）
    直接把模型分數當控制變數，不用殘差，避免 0/1 殘差被模型機率機械式排序的問題。"""
    out = {}
    x = z[col]
    # 升級
    ok = x.notna() & df["score_promoted"].notna() & df["promoted"].notna()
    if ok.sum() > 30 and x[ok].std() > 0:
        X = np.column_stack([np.ones(ok.sum()), df.loc[ok, "score_promoted"], x[ok]])
        b, se = logistic_irls(X, df.loc[ok, "promoted"].astype(float).to_numpy())
        out["prom"] = (b[2], b[2] / se[2], int(ok.sum()))
    # 登上
    obs = df["war_observable"].astype(bool)
    ok = x.notna() & obs & df["score_preach"].notna()
    if ok.sum() > 30 and x[ok].std() > 0:
        pr = df.loc[ok, "score_preach"].clip(1e-4, 1 - 1e-4)
        X = np.column_stack([np.ones(ok.sum()), np.log(pr / (1 - pr)), x[ok]])
        b, se = logistic_irls(X, df.loc[ok, "reached_fwd"].astype(float).to_numpy())
        out["reach"] = (b[2], b[2] / se[2], int(ok.sum()))
    # WAR|登上
    ok = x.notna() & obs & df["reached_fwd"].astype(bool) & df["score_war_asinh"].notna()
    if ok.sum() > 30 and x[ok].std() > 0:
        X = np.column_stack([np.ones(ok.sum()), df.loc[ok, "score_war_asinh"], x[ok]])
        y = np.arcsinh(df.loc[ok, "war_fwd"].to_numpy(dtype=float))
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        s2 = resid @ resid / max(len(y) - 3, 1)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        out["war"] = (b[2], b[2] / se[2], int(ok.sum()))
    return out


def residual_table(df: pd.DataFrame, z: pd.DataFrame, names: list) -> pd.DataFrame:
    """每個候選欄位在模型分數之外的係數（候選已層級內標準化，係數 = 高一個標準差多多少）；|z| > 2 標 *。"""
    rows = []
    for disp, col in names:
        r = beyond_model(df, z, col)
        bw, zw, nw = r.get("war", (np.nan, np.nan, 0))
        br, zr, nr = r.get("reach", (np.nan, np.nan, 0))
        bp, zp, npn = r.get("prom", (np.nan, np.nan, 0))
        sig = lambda zz: "*" if np.isfinite(zz) and abs(zz) > 2 else ""
        sw, sr, sp = sig(zw), sig(zr), sig(zp)
        verdict = ""
        if (sw or sr) and not sp:
            verdict = "被放掉（跟真相有關、跟升級無關）"
        elif sp and not (sw or sr):
            verdict = "球團偏好（只跟升級有關）"
        elif sp and (sw or sr):
            verdict = "兩邊都有關"
        rows.append(
            {
                "候選": disp,
                "WAR|登上（asinh）": bw,
                "": sw,
                "登上（logit）": br,
                " ": sr,
                "升級（logit）": bp,
                "  ": sp,
                "n（登上者）": nw,
                "判讀": verdict,
            }
        )
    out = pd.DataFrame(rows)
    out["_k"] = out[["WAR|登上（asinh）", "登上（logit）"]].abs().max(axis=1).fillna(0)
    return out.sort_values("_k", ascending=False).drop(columns="_k").round(3)


def interaction_screen(z: pd.DataFrame, level: pd.Series, res: dict, names: list, top: int = 20) -> pd.DataFrame:
    """兩兩相乘的交互作用：z_a × z_b 在層級內重新標準化，各與兩種殘差算相關。
    門檻拉到 3/√n（兩兩配對有幾百個，2/√n 會有幾十個碰巧過線）；單年只當線索，跨年同號才算。"""
    r_war, r_prom = res["war"], res["prom"]
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            disp_a, a = names[i]
            disp_b, b = names[j]
            x = z[a] * z[b]
            mu = x.groupby(level).transform("mean")
            sd = x.groupby(level).transform("std")
            x = ((x - mu) / (sd + 1e-9)).clip(-4, 4)
            ok_w = x.notna() & r_war.notna()
            ok_p = x.notna() & r_prom.notna()
            if ok_w.sum() < 30 or x[ok_w].std() == 0:
                continue
            rw, _ = spearman(x, r_war)
            rp, _ = spearman(x, r_prom) if ok_p.sum() >= 30 else (np.nan, 0)
            rows.append(
                {
                    "交互作用": f"{disp_a} × {disp_b}",
                    "與 WAR 殘差的相關": rw,
                    "": "*" if abs(rw) > 3 / np.sqrt(ok_w.sum()) else "",
                    "與升級殘差的相關": rp,
                    " ": "*" if np.isfinite(rp) and abs(rp) > 3 / np.sqrt(ok_p.sum()) else "",
                    "n（WAR）": int(ok_w.sum()),
                }
            )
    out = pd.DataFrame(rows)
    out["_k"] = out["與 WAR 殘差的相關"].abs()
    return out.sort_values("_k", ascending=False).drop(columns="_k").head(top).round(3), len(rows)


def main(data_dir, season, out_dir, tag_p, tag_w, score):
    out = Path(out_dir)
    m = pd.read_parquet(Path(data_dir) / "model_bat.parquet")
    m = add_extra(m)
    m = m[m["season"] == season].reset_index(drop=True)
    m = add_position(m, Path(data_dir))
    m = add_club_strength(m, Path(data_dir))
    m, names = build_candidates(m)
    eta_p = pd.read_parquet(out / str(season) / tag_p / "eta_M1.parquet")
    eta_w = pd.read_parquet(out / str(season) / tag_w / "eta_M1.parquet")
    scores = eta_p[
        ["key_mlbam", "score_promoted"]
        + [f"eta_{f}" for f in FACTORS]
        + (["zeta_scout"] if "zeta_scout" in eta_p else [])
    ].merge(eta_w[["key_mlbam", "score_war", "score_war_asinh", "score_preach"]], on="key_mlbam", how="inner")
    df = m.merge(scores, on="key_mlbam", how="inner")
    df["exp_war"] = df["score_preach"] * df["score_war"]
    df = df[df["PA"] >= 100].reset_index(drop=True)  # 打席太少的人分數全是先驗
    names_all = (
        names
        + [(f"能力：{f}", f"eta_{f}") for f in FACTORS]
        + ([("球探評價 ζ", "zeta_scout")] if "zeta_scout" in df else [])
    )
    z = standardize_within_level(df, [c for _, c in names_all])
    group = assign_groups(df, score)
    sizes = (
        df[group != ""]
        .groupby(group[group != ""])
        .agg(人數=("war_fwd", "size"), 六季WAR平均=("war_fwd", "mean"), 登上比例=("reached_fwd", "mean"))
        .round(2)
    )
    prof = group_profile(z, group, names_all)
    res = residuals(df)
    rt = residual_table(df, z, names_all)
    inter, n_pairs = interaction_screen(z, df["level_main"], res, names)  # 只配原始候選，不配能力分數
    lead = ["key_mlbam", "level_main", "age", "PA", "pick_overall", "next_status", "exp_war", "reached_fwd", "war_fwd"]
    pb = position_by_bat(df)
    md = [
        f"# {season} 錯放名單比對 · 升級 {tag_p} · WAR {tag_w} · 模型看好 = {score} 前 k 名\n",
        "打席 ≥ 100；所有欄位在層級內標準化（0 = 該層級平均，1 = 高一個標準差）。\n",
        "## 四群人數\n",
        sizes.to_markdown(),
        "",
        "## 四群的分布比較（中位數與 P(群 > 對照)，依離 0.5 最遠排序）\n",
        "P(群 > 對照) = 隨機抽一個該群的人比隨機抽一個對照組的人高的機率，0.5 = 沒差。錯放成功偏離 0.5 的欄位 = 球團可能漏看的；升錯 = 模型和球團都漏看的；球團獨具眼光 = 球團看到而模型沒看到的。\n",
        prof.to_markdown(index=False),
        "",
        "## 模型分數之外，候選還多說了什麼\n",
        "模型分數當控制變數，再放一個候選：升級 = logit(升不升) ~ 模型 logit + 候選；登上 = logit(登上與否) ~ logit(模型登上機率) + 候選；WAR|登上 = asinh(六季 WAR) ~ 模型條件期望 + 候選。表中是候選的係數（候選已標準化，= 高一個標準差多多少），* = |z| > 2。不用殘差，避免 0/1 殘差被模型機率機械式排序。\n",
        rt.to_markdown(index=False),
        "",
        "## 交互作用篩選（兩兩相乘，與 WAR 殘差相關最強的 20 對）\n",
        f"共 {n_pairs} 對；* = |r| > 3/√n。就算全是雜訊也會有幾對過線，單年只當線索，四年同號才進候選名單。\n",
        inter.to_markdown(index=False),
        "",
        "## 守位 × 打擊能力：登上比例與六季 WAR（打席 ≥ 100）\n",
        "打擊能力 = 打得到球與打得遠的平均，層級內分三等。守備型守位（捕手、游擊二壘）每一等的登上比例都比一壘指定高，就是「門檻低」；同一守位內高等比低等高，就是「棒子還是算」。WAR 是零堆積加長右尾，中位數和平均都不能代表它：報登上比例（門檻）、達 2 與達 5 的比例（像樣的球員與明星）、每百人總和（整群值多少）。\n",
        pb.to_markdown(index=False),
        "",
        "## 錯放成功名單\n",
        df.loc[group == "錯放成功", lead].sort_values("war_fwd", ascending=False).round(2).to_markdown(index=False),
        "",
        "## 升錯名單（期望 WAR 最高 20 人）\n",
        df.loc[group == "升錯", lead].nlargest(20, "exp_war").round(2).to_markdown(index=False),
        "",
    ]
    name = f"misplaced_{tag_p}_vs_{tag_w}.md"
    (out / str(season) / name).write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {out / str(season) / name}")


if __name__ == "__main__":
    args = read_args("錯放名單比對：模型猜錯的人身上有什麼共同點（找新的候選變數，不取樣）。", score=True)
    main(args.data_dir, args.season, args.out_dir, args.promoted, args.war, args.score)
