"""後驗不只平均和標準差：把每個人四把尺的所有 MCMC 抽樣記成十分位，問球隊買的是平均、上限，還是「成為菁英的機率」。

    python src/eta_quantiles.py outputs/ data/ --promoted promoted_f4sed_linear_p2
    python src/eta_quantiles.py --selftest

每季讀 outputs/<season>/<tag>/idata_M1.nc 的 eta（chain, draw, 人, 尺），每人每尺存九個十分位、P(η > 1)（有多少機率是比平均高一個標準差以上的人）、P(η < −1）。
寫 outputs/<season>/<tag>/eta_quantiles.parquet（列順序跟 eta_M1.parquet 一樣）。
然後跟升級／登上／WAR 的基底（四把尺的後驗平均）比：在平均之外，再放
  上尾（第 90 百分位 − 平均）、下尾（平均 − 第 10 百分位）、P(η > 1)、P(η < −1)
各一個，看係數。後驗接近常態時上尾下尾只是標準差的倍數，差別在 P(菁英) 這種非線性的量——球隊會不會為「可能是明星」多付。
輸出 outputs/eta_quantiles_<promoted tag>.md。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hit_profile import BASE, coef_se, load
from bf import bf_str, log10_bf10

QS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
FACTORS = ["contact", "power", "speed", "discipline"]
FNAME = {"contact": "打得到球", "power": "打得遠", "speed": "跑得快", "discipline": "選得到球"}


def quantiles_from_draws(eta: np.ndarray, keys: pd.DataFrame) -> pd.DataFrame:
    """eta 抽樣 (draws, 人, 尺) → 每人每尺的十分位、P(η > 1)、P(η < −1)。"""
    out = keys.copy()
    for k, name in enumerate(FACTORS[: eta.shape[2]]):
        for q in QS:
            out[f"eta_{name}_q{q}"] = np.quantile(eta[:, :, k], q / 100, axis=0)
        out[f"eta_{name}_ptop"] = (eta[:, :, k] > 1).mean(axis=0)
        out[f"eta_{name}_pbot"] = (eta[:, :, k] < -1).mean(axis=0)
    return out


def quantile_file(season_dir: Path) -> pd.DataFrame:
    """一季：讀 idata_M1.nc 的 eta（chain, draw, 人, 尺），寫 eta_quantiles.parquet 並回傳。"""
    idata = az.from_netcdf(season_dir / "idata_M1.nc")
    eta = idata.posterior["eta"].values
    eta = eta.reshape(-1, eta.shape[2], eta.shape[3])
    keys = pd.read_parquet(season_dir / "eta_M1.parquet", columns=["key_mlbam", "season"])
    out = quantiles_from_draws(eta, keys)
    out.to_parquet(season_dir / "eta_quantiles.parquet", index=False)
    return out


def shape_vars(d: pd.DataFrame) -> list:
    """每把尺四個形狀量：上尾、下尾、P(菁英)、P(墊底)；季 × 層級內標準化。回傳變數名。"""
    names = []
    for name in FACTORS:
        if f"eta_{name}_q90" not in d.columns:
            continue
        d[f"{name}_up"] = d[f"eta_{name}_q90"] - d[f"eta_{name}_q50"]
        d[f"{name}_down"] = d[f"eta_{name}_q50"] - d[f"eta_{name}_q10"]
        d[f"{name}_ptop"] = d[f"eta_{name}_ptop"]
        d[f"{name}_pbot"] = d[f"eta_{name}_pbot"]
        for v in ("up", "down", "ptop", "pbot"):
            c = f"{name}_{v}"
            d[c] = d.groupby(["season", "level_main"])[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0.0)
            names.append(c)
    return names


def pooled(d: pd.DataFrame, outcome: str, feats: list) -> tuple:
    bs, ses = [], []
    for s in sorted(d["season"].unique()):
        b, se = coef_se(d[d["season"] == s], outcome, feats)
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            bs.append(b)
            ses.append(se)
    if not bs:
        return np.nan, np.nan, 0, 0
    w = 1.0 / np.array(ses) ** 2
    pooled_b = float((np.array(bs) * w).sum() / w.sum())
    return pooled_b, pooled_b / float(np.sqrt(1 / w.sum())), sum(b > 0 for b in bs), sum(b < 0 for b in bs)


VLABEL = {"up": "上尾（第 90 − 第 50）", "down": "下尾（第 50 − 第 10）", "ptop": "P(η > 1，菁英)", "pbot": "P(η < −1，墊底)"}


def main(out_dir: str, data_dir: str, tag_p: str) -> str:
    out, data = Path(out_dir), Path(data_dir)
    frames = []
    for season_dir in sorted(out.glob(f"20*/{tag_p}")):
        qf = season_dir / "eta_quantiles.parquet"
        if qf.exists():
            frames.append(pd.read_parquet(qf))
        elif (season_dir / "idata_M1.nc").exists():
            frames.append(quantile_file(season_dir))
            print(f"  {season_dir.parent.name} 十分位完成", flush=True)
    if not frames:
        raise SystemExit(f"找不到 {out}/20*/{tag_p}/idata_M1.nc")
    q = pd.concat(frames, ignore_index=True)
    d, _ = load(out, data, tag_p)
    d = d.merge(q, on=["key_mlbam", "season"], how="inner")
    names = shape_vars(d)
    base = BASE + [c for c in d.columns if c.startswith("lvl_")]
    outcomes = [("y_prom", "隔年升不升"), ("y_reach", "六季內登上"), ("y_war", "登上者六季 WAR")]
    outcomes = [(o, l) for o, l in outcomes if d[o].notna().sum() >= 100]
    rows = []
    for c in names:
        name, v = c.rsplit("_", 1)
        row = {"尺": FNAME[name], "形狀量": VLABEL[v]}
        for outcome, label in outcomes:
            b, z, pos, neg = pooled(d, outcome, base + [c])
            row[label] = f"{b:+.3f} [BF {bf_str(log10_bf10(b, abs(b / z) if z else np.nan))}]  {pos}/{neg}" if np.isfinite(b) else "—"
        rows.append(row)
    md = [
        f"# 後驗的形狀：球隊買平均、買上限，還是買「可能是明星」 · {tag_p}\n",
        f"樣本：{len(d):,} 人季，{int(d['season'].min())}–{int(d['season'].max())}。基底是四把尺的後驗平均 + 球探 + 年齡 + 打席 + 層級；"
        "每列多放一個形狀量（季 × 層級內標準化）。格式「係數 [BF10]  幾季正/幾季負」，BF10 是 Savage–Dickey（先驗 N(0, 0.5²)）。上尾、下尾在後驗接近常態時只是標準差的倍數（打席少的人大）；"
        "P(菁英) 是非線性的：同樣平均，不確定性大的人 P(η > 1) 也大。\n",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
    ]
    text = "\n".join(md)
    path = out / f"eta_quantiles_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合成後驗抽樣：平均一樣時，P(菁英) 高的人（不確定性大）被升得多；檢查 ptop 係數為正。（.nc 的讀取路徑只在 Arc 上走，本機沒有 netCDF 後端。）"""
    rng = np.random.default_rng(11)
    root = Path(tempfile.mkdtemp())
    data, out = root / "data", root / "outputs"
    data.mkdir()
    rows = []
    for season in (2016, 2017, 2018):
        n = 500
        mean = rng.normal(0, 1, (n, 4))
        sd = rng.uniform(0.2, 1.2, n)
        draws = mean[None, :, :] + sd[None, :, None] * rng.normal(0, 1, (400, n, 4))
        eta = draws.reshape(2, 200, n, 4)
        ptop = (draws[:, :, 0] > 1).mean(axis=0)
        logit = -1 + 0.5 * mean[:, 0] + 2.0 * (ptop - ptop.mean())
        prom = rng.random(n) < 1 / (1 + np.exp(-logit))
        sd_dir = out / str(season) / "promoted_test"
        sd_dir.mkdir(parents=True)
        keys = pd.DataFrame({"key_mlbam": 100000 * (season - 2015) + np.arange(n), "season": season})
        quantiles_from_draws(draws, keys).to_parquet(sd_dir / "eta_quantiles.parquet", index=False)  # 本機沒有 netCDF，直接寫十分位檔
        e = keys.copy()
        for k, name in enumerate(FACTORS):
            e[f"eta_{name}"] = mean[:, k]
        e.assign(zeta_scout=np.nan).to_parquet(sd_dir / "eta_M1.parquet", index=False)
        for i in range(n):
            rows.append(
                {
                    "key_mlbam": int(keys["key_mlbam"][i]),
                    "season": season,
                    "level_main": "AA",
                    "org": "NYY",
                    "PA": 400,
                    "promoted": bool(prom[i]),
                    "next_status": "same",
                    "age_rel": rng.normal(),
                    "war_observable": True,
                    "reached_fwd": rng.random() < 0.3,
                    "war_fwd": rng.normal(1, 2),
                }
            )
    pd.DataFrame(rows).to_parquet(data / "model_bat.parquet", index=False)
    main(str(out), str(data), "promoted_test")
    q = pd.concat([pd.read_parquet(p) for p in out.glob("20*/promoted_test/eta_quantiles.parquet")])
    d, _ = load(out, data, "promoted_test")
    d = d.merge(q, on=["key_mlbam", "season"])
    shape_vars(d)
    b, z, pos, _ = pooled(d, "y_prom", BASE + [c for c in d.columns if c.startswith("lvl_")] + ["contact_ptop"])
    assert b > 0 and z > 3 and pos == 3, f"P(菁英) 沒抓到：{b:.3f} z {z:.1f}"
    print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="後驗十分位：球隊買平均、上限還是菁英機率")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("--promoted", default=None, help="升級模型的 tag，例如 promoted_f4sed_linear_p2")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif a.promoted is None:
        p.error("--promoted 是必要的（或用 --selftest）")
    else:
        main(a.out_dir, a.data_dir, a.promoted)
