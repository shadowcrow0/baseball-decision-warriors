"""把單季的分析跑遍所有有模型分數的球季，再把每張表跨季合併：係數用反變異數加權、其他數字取平均，並數幾季正幾季負。

    python src/all_seasons.py discipline_split data/ outputs/ --promoted promoted_f4sed_linear_p2 --war war_f4sed_linear_p2
    python src/all_seasons.py club_need        data/ outputs/ --promoted promoted_f4sed_linear_p2 --war war_f4sed_linear_p2
    python src/all_seasons.py org_background   data/ outputs/ --promoted promoted_f4sed_linear_p2 --war war_f4sed_linear_p2
    python src/all_seasons.py validate_abilities data/ outputs/ --promoted promoted_f4sed_linear_p2
    python src/all_seasons.py --selftest

哪些季：outputs/<season>/<promoted tag>/eta_M1.parquet 存在的季；要 --war 的腳本再要求 war tag 也存在（13 季）。
每個腳本的 main() 回傳 {表名: (DataFrame, 括號裡是什麼)}；括號是 "se" 時格子像 "+0.49 (0.05)"，是 "z" 時像 "+0.49 (+3.2)"，None 就是純數字。
合併規則（每張表、每個格子，對齊第一欄當列名）：
  有 se → 反變異數加權平均，另給合併 z；有 z → 先換回 se 再同上；純數字 → 跨季平均。
  「幾季正 / 幾季負」= 該格逐季的符號數。文字格（如 ◆ 標記）不合併。
輸出 outputs/<腳本>_all_<promoted tag>.md，單季的 md 照舊各寫各的。
"""

import argparse
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
NEEDS_WAR = {"discipline_split": True, "club_need": True, "org_background": False, "validate_abilities": False}


def seasons_for(out: Path, tag_p: str, tag_w: str | None) -> list:
    have_p = {int(p.parent.parent.name) for p in out.glob(f"20*/{tag_p}/eta_M1.parquet")}
    if tag_w:
        have_w = {int(p.parent.parent.name) for p in out.glob(f"20*/{tag_w}/eta_M1.parquet")}
        have_p &= have_w
    return sorted(have_p)


def parse_cell(x, paren: str | None) -> tuple:
    """一個格子 → (值, se)。純數字 se 為 nan；"+0.49 (0.05)" → (0.49, 0.05)；"+0.49 (+3.2)" 且 paren == "z" → (0.49, 0.49/3.2)。拿不到數字回 (nan, nan)。"""
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return (float(x), np.nan) if np.isfinite(x) else (np.nan, np.nan)
    if not isinstance(x, str):
        return np.nan, np.nan
    nums = NUM.findall(x)
    if not nums:
        return np.nan, np.nan
    v = float(nums[0])
    if paren and len(nums) >= 2:
        second = float(nums[1])
        if paren == "se":
            return v, second if second > 0 else np.nan
        if paren == "z":
            return v, abs(v / second) if second != 0 else np.nan
    return v, np.nan


def pool_table(frames: list, paren: str | None) -> pd.DataFrame:
    """同名的表跨季合併。frames = [(season, DataFrame)]。第一欄當列名；其餘欄逐格合併。"""
    key = frames[0][1].columns[0]
    cols = [c for c in frames[0][1].columns if c != key]
    parsed = {}  # (row, col) → list of (season, v, se)
    for season, df in frames:
        if key not in df.columns:
            continue
        for _, r in df.iterrows():
            for c in cols:
                if c in df.columns:
                    v, se = parse_cell(r[c], paren)
                    if np.isfinite(v):
                        parsed.setdefault((str(r[key]), c), []).append((season, v, se))
    rows_order = []
    for _, df in frames:
        for k in df[key].astype(str):
            if k not in rows_order:
                rows_order.append(k)
    out = []
    for rk in rows_order:
        row = {key: rk}
        for c in cols:
            items = parsed.get((rk, c), [])
            if not items:
                row[c] = ""
                continue
            vs = np.array([v for _, v, _ in items])
            ses = np.array([s for _, _, s in items])
            pos, neg = int((vs > 0).sum()), int((vs < 0).sum())
            if np.isfinite(ses).all() and (ses > 0).all():
                w = 1 / ses**2
                m = float((vs * w).sum() / w.sum())
                z = m / float(np.sqrt(1 / w.sum()))
                row[c] = f"{m:+.3f} (z {z:+.1f})  {pos}/{neg}"
            elif (vs < 0).any() or (paren and np.isfinite(ses).any()):
                row[c] = f"{vs.mean():+.3f}  {pos}/{neg}"
            else:
                row[c] = f"{vs.mean():.3f}"
        row["季數"] = len({s for s, _, _ in parsed.get((rk, cols[0]), [])} | {s for c in cols for s, _, _ in parsed.get((rk, c), [])})
        out.append(row)
    return pd.DataFrame(out)


def run(script: str, data_dir: str, out_dir: str, tag_p: str, tag_w: str | None, extra: dict) -> str:
    mod = importlib.import_module(script)
    out = Path(out_dir)
    seasons = seasons_for(out, tag_p, tag_w if NEEDS_WAR.get(script, bool(tag_w)) else None)
    collected = {}
    done = []
    for season in seasons:
        print(f"== {script} {season}", flush=True)
        try:
            if script == "validate_abilities":
                tables = mod.main(data_dir, out_dir, season, tag_p, **extra)
            else:
                tables = mod.main(data_dir, season, out_dir, tag_p, tag_w)
        except Exception as e:  # noqa: BLE001  一季壞了不要拖垮全部
            print(f"  {season} 失敗：{type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if not tables:
            continue
        done.append(season)
        for name, (df, paren) in tables.items():
            collected.setdefault(name, {"paren": paren, "frames": []})["frames"].append((season, df))
    md = [
        f"# {script} · 全季合併 · {tag_p}" + (f" / {tag_w}" if tag_w else "") + "\n",
        f"跑了 {len(done)} 季：{', '.join(str(s) for s in done)}。格式「合併值 (z)  幾季正/幾季負」；有標準誤的格子用反變異數加權，其他取平均。單季的表在 outputs/<season>/ 裡。\n",
    ]
    for name, item in collected.items():
        md += [f"## {name}\n", pool_table(item["frames"], item["paren"]).to_markdown(index=False), ""]
    text = "\n".join(md)
    path = out / f"{script}_all_{tag_p}.md"
    path.write_text(text)
    print(text, flush=True)
    print(f"-> {path}", flush=True)
    return text


def selftest() -> None:
    """合併規則：se 格反變異數加權、z 格換回 se、純數字取平均、符號數對。"""
    a = pd.DataFrame({"能力": ["打得到球", "打得遠"], "選球差": ["+0.50 (0.10)", "-0.20 (0.10)"], "n": [100, 100]})
    b = pd.DataFrame({"能力": ["打得到球", "打得遠"], "選球差": ["+0.30 (0.10)", "+0.10 (0.10)"], "n": [200, 200]})
    t = pool_table([(2015, a), (2016, b)], "se").set_index("能力")
    assert t.loc["打得到球", "選球差"].startswith("+0.400") and "2/0" in t.loc["打得到球", "選球差"]
    assert "1/1" in t.loc["打得遠", "選球差"] and t.loc["打得到球", "n"] == "150.000"
    c = pd.DataFrame({"結果": ["升級"], "缺口": ["+0.40 (+4.0)"]})
    d = pd.DataFrame({"結果": ["升級"], "缺口": ["+0.20 (+1.0)"]})
    t2 = pool_table([(2015, c), (2016, d)], "z").set_index("結果")
    v = float(t2.loc["升級", "缺口"].split()[0])
    assert 0.35 < v < 0.40, t2  # se 0.1 與 0.2 → 權重 100 與 25 → 0.36
    print(t.to_string()); print(t2.to_string()); print("selftest OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="單季分析跑遍所有季並跨季合併")
    p.add_argument("script", nargs="?", choices=list(NEEDS_WAR))
    p.add_argument("data_dir", nargs="?", default="data")
    p.add_argument("out_dir", nargs="?", default="outputs")
    p.add_argument("--promoted", default=None)
    p.add_argument("--war", default=None)
    p.add_argument("--years", type=int, default=3, help="validate_abilities 用：登上後看幾年")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
    elif not (a.script and a.promoted):
        p.error("要 script 與 --promoted（或 --selftest）")
    else:
        extra = {"years": a.years} if a.script == "validate_abilities" else {}
        run(a.script, a.data_dir, a.out_dir, a.promoted, a.war, extra)
