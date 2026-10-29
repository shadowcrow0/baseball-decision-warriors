# Two GM moments: draft-day value and farm promotion vs. later WAR

**Decision WARriors · 2026 台灣棒球數據分析競賽 · 球隊經營戰略組**

這個公開版收錄本研究的**全部程式、模型輸出（算出來的結果）與投影片**，讓評審與讀者可以重現每一張圖的數字。
原始資料（FanGraphs、Baseball-Reference、Chadwick、Baseball Savant）依各站使用條款**不再散布**，請用 `src/fetch_*.py` 自行抓取（程式會限速、快取）。

This public mirror contains all code, derived outputs and the slide deck. Raw data from FanGraphs, Baseball-Reference, Chadwick and Baseball Savant is **not redistributed**; the `src/fetch_*.py` scripts download it (rate-limited, cached).

## 問題

1. 選秀日：第 N 順位值多少 WAR，簽約金可以付到幾倍？（PART I）
2. 農場的每一季：球隊升人時看成績單上的哪些能力？幾年後的 WAR 要的是哪些？兩邊差在哪、差多少？（PART II）

## 一句話的答案

球隊用成績單上的數字分配機會，機會變成 WAR；真相要的是數字背後的能力；兩邊挑的人三個裡有一個不同，照真相升人一支 3A 每年多約 34 WAR（2007–2019 十三季）。

## 目錄

| 路徑 | 內容 |
|---|---|
| `deck/project/` | 投影片（30 頁，HTML；每頁一個 `<section>`，`deck.json` 是順序） |
| `src/` | 全部程式（純 Python；PyMC 模型在 `fit_year.py`，每支腳本有 `--selftest`） |
| `scripts/` | Slurm 叢集的啟動與狀態腳本（UTSA Arc） |
| `outputs/` | 模型輸出：每季的 `summary.md`、`lambda_M1.csv`（負荷量）、`struct_M1.csv`（結構係數）、`eta_M1.parquet`（每人季的能力分數）、跨季彙整的 `.md`、圖 |
| `docs/` | `research_design.md`（設計）、`model_spec.md`（數學）、`findings.md`（發現，含所有數字的出處）、`data_inventory.md`、`technical.md` |
| `reports/` | 資料檢查與描述統計 |

## 重跑

```bash
python -m venv env && source env/bin/activate && pip install -r requirements.txt
# 1. 抓資料（FanGraphs 排行榜 API、Baseball-Reference 選秀頁、Chadwick、Savant；數小時，限速）
python src/fetch_minors.py data/ && python src/fetch_war.py 2005 2025 data/ && python src/fetch_draft.py 2005 2025 data/draft.parquet
python src/clean_minors.py data/ && python src/build_model_table.py data/
# 2. 一季一個模型（約 2–12 小時；叢集上用 scripts/arc_fit.sh）
python src/fit_year.py data/ 2015 outputs/ --factors 4 --scout --extra --direct            # 升級段
python src/fit_year.py data/ 2015 outputs/ --factors 4 --scout --extra --direct --outcome war   # 幾年登上 + WAR
# 3. 跨季彙整與檢查（讀 outputs/，一般電腦可跑）
python src/all_years.py outputs/ && python src/sim_policy.py outputs/ && python src/ppc.py outputs/ data/ --all promoted_f4sed_linear_p2
```

每支腳本開頭的 docstring 寫了用法與輸出；`docs/findings.md` 每一條發現都標了是哪支程式、哪個檔算出來的。

## 授權

- 程式：MIT（見 `LICENSE`）。
- 模型輸出與文件：CC BY 4.0，引用請註明 Decision WARriors（2026）。
- 資料：FanGraphs、Baseball-Reference、Chadwick Baseball Bureau、Baseball Savant 各自的條款；本庫不含原始資料。
