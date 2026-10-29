# 技術文件：資料管線、清洗、建模表、擬合

> 對應 `research_design.md`（總覽）與 `model_spec.md`（數學）。本文件講「怎麼做出來的」，可直接當重現手冊。

## 1. 執行環境

- UTSA Arc（Slurm）。login 節點不能載入 anaconda，所有 Python 工作在 compute 節點：
  `srun -p compute1 -n 1 -t 10:00:00 --cpus-per-task=8 --pty bash`，然後
  `module load anaconda3/2024.10-1 && source ~/pyb_env/bin/activate`。
- 虛擬環境 `~/pyb_env`，套件見 `requirements.txt`。隱性依賴要特別裝：`tabulate`（to_markdown）、`lxml`+`html5lib`+`bs4`（read_html 退路）、`h5netcdf`+`h5py`（ArviZ 存 netcdf）。
- 資料放 `/work/spt904/baseball/{data,cache}`，repo 內的 `data`、`cache` 是符號連結。git 不跟進符號連結，所以要進 git 的表複製到 `rawdata/`。
- 長時間工作用 `screen`，不用 sbatch（`--mem` 會被 Arc 拒絕；互動節點可開 10 小時）。
- 版本：Python 3.12、pandas 3、PyMC 6.3、ArviZ 1.3。ArviZ 1.x 是大改版：`InferenceData` 變 `DataTree`、`compare` 無 `ic` 參數、`waic` 移除、`summary` 欄位改名（`eti89_lb/ub`）。`fit_year.py` 已對應。

## 2. 資料來源與抓取

| 腳本 | 來源 | 輸出 | 要點 |
|---|---|---|---|
| `fetch_minors.py` | FanGraphs 小聯盟 JSON API `api/leaders/minor-league/data` | `minors_bat.parquet`、`minors_pit.parquet` | 逐年 × 逐層級（level 1–8）× 打者/投手，一列 = 一人一季一層級；每請求 4 秒間隔，429 退避；JSON 快取於 `cache/minors/` |
| `fetch_draft.py` | Baseball-Reference 選秀頁 | `draft.parquet` | 用 pybaseball 的 `BRefSession`（限速約 10 次/分），自己 `read_html(StringIO)` 繞過 pybaseball 在 pandas 3 的 bug；只快取含 `<table>` 的頁；Cloudflare 擋頁即中止 |
| `fetch_war.py` | FanGraphs 大聯盟 JSON API | `fg_batting.parquet`、`fg_pitching.parquet` | 舊端點 `leaders-legacy.aspx` 全球 403；新 API **不能加瀏覽器 UA**（會被 Cloudflare 挑戰），requests 預設 UA 可過 |
| `fetch_ids.py` | Chadwick register（GitHub raw CSV ×16） | `ids.parquet` | pybaseball 的版本會丟掉 `key_bbref_minors`，所以直接抓 |
| `fetch_statcast.py` | Baseball Savant（經 pybaseball `statcast`） | `statcast_<year>.parquet` | 按週分段、可續抓；12 季約 1 GB，第二階段用 |
| `fetch_register.py` | B-Ref 小聯盟球員頁（register） | `register_paths.parquet` | 只抓「離開後回來」的 2,738 人；表格藏在 HTML 註解內，先去掉 `<!-- -->` 再 `read_html`；Lev 代碼：`Ind` 獨立聯盟、`Fgn` 海外、`FgW` 海外冬季、`Fal` 秋季 |

連通性：`check_sources.py`。從雲端 IP 幾乎全被擋，從校園網路 B-Ref、Chadwick、Savant 可過，FanGraphs 要走新 API。

完整性：`check_data.py` → `reports/data_check.md`。對每個資料集逐年比對預期值（選秀各年輪數、小聯盟各年該有的層級、FanGraphs 各年人數、Statcast 例行賽週數與球數下限）。目前 0 WARN / 121 PASS。

## 3. 清洗（`clean_minors.py`）

原則：分母進模型、尾巴留著、只切母體。

1. **投手判定**：同季 `TBF > PA` 視為投手，其打者列移除；反之野手的投手列移除。不用「出現在投手表」判定，以免誤殺大比分上去投一局的野手。2022 年前 AA/AAA 有 35% 打者列是投手打擊。
2. **post_debut**：`season > mlb_played_first` 的列標記，保留。
3. 不設出賽數門檻，不 winsorize。

輸出：`minors_bat_clean.parquet`、`minors_pit_clean.parquet`、`player_season.parquet`（一人一季：最高/最低層級、跨幾層、總 PA 或 IP、年齡、球團、下一季去向）。

下一季去向 `next_status`：`mlb`（首登年 ≤ 本季+1，優先判定）、`promoted`、`same`、`demoted`、`gone_return`（之後有回來，記 `gap_years`）、`gone_exit`、`censored`。2020 停賽：2019 的下一季是 2021。

層級順序 `level_rank`：DSL 0、R/CPX 1、A- 2、A 3、A+ 4、AA 5、AAA 6。

## 4. 選秀連接（`link_draft.py`）

B-Ref 選秀表沒有球員 id。先從快取 HTML 的球員連結取 B-Ref id（`/players/x/xxx.shtml` → `key_bbref`；`register/player.fcgi?id=` → `key_bbref_minors`），對 Chadwick 取 MLBAM id；對不到的退回姓名 + 職業首年配對。簽約者 16,992 人，100% 配到（連結 16,987、姓名 5）。

## 5. 建模表（`build_model_table.py`）

一人一季一列，打者。跨層級的季：分子分母跨層加總，`level_rank_main` 取打席最多的層級。

| 群組 | 欄位 |
|---|---|
| 鍵 | `key_mlbam`、`season`、`org`、`player_idx`、`org_idx` |
| 指標分子/分母 | `H/AB`、`OB/PA`（OB = H+BB+HBP）、`BB/PA`、`SO/PA`、`XBH/H`、`SB/SBA`、`whiff/Pitches`（whiff = SwStr% × Pitches）、`GB/BIP`、`PULL/BIP`（BIP = AB−SO+SF）、`HR/AB` |
| 共變數 | `log_pick`（非選秀 = log 1500）、`log_bonus`、`bonus_known`、`draft_level`、`age_rel`（減該層級該年中位年齡）、`level_rank_main`、`n_levels` |
| 結果 A | `promoted`（promoted ∪ mlb）、`next_status`、`structural` 遮罩（排除 post_debut 與 censored） |
| 結果 B | `war_fwd`（本季+1 到 +6 的大聯盟 fWAR 合計，打擊+投球，沒登上 = 0）、`reached_fwd`（首登年在 [本季, 本季+6]）、`war_observable`（本季+6 ≤ 2025）、`war_structural` 遮罩 |

2006 年的 whiff/GB/PULL 設缺值（該年無擊球類型資料）。

規模：75,538 人季、20,995 人；結構遮罩 62,754；WAR 遮罩 46,733（2006–2019）。

## 6. 擬合（`fit_year.py`）

```
python src/fit_year.py data/ <season> outputs/ [選項]      （--help 列出全部選項）
```

- 一個球季一個 fit，同時跑 M0（無 factor）與 M1（有 factor），測量模型共用。
- 取樣：NUTS，4 鏈，`target_accept=0.9`，起始值依棒球常識的符號；取樣後 `align_signs` 逐鏈翻正鏡像峰。診斷表含到最大樹深的比例與平均步長。`--quick` = 200+200。純 Python 取樣器；`numpyro`（JAX）可選。
- `--reuse`：`outputs/<season>/idata_M*.nc` 已存在就讀回，不重新取樣。
- 後處理：負荷量表、結構係數、診斷（divergences、max r-hat 逐參數取最大）、PSIS-LOO 比較（`az.compare`）、WAIC（ArviZ 1.x 已移除，自算：`Σ_i [log mean_s exp(ll_is) − Var_s(ll_is)]`）、Pareto-k > 0.7 的人季、η 後驗均值表。
- 速度：PyMC 預設取樣器，3,849 人季、200+200 約 22 分鐘；正式 1,000+1,000 約 2 小時一個模型。多年份分節點平行。
- 多節點啟動：`scripts/arc_fit.sh <season> [參數]` 從 login 節點開一個 screen 視窗，裡面 `srun` 一個 compute 節點跑 `fit_year.py`，log 寫到 `outputs/logs/`；`scripts/arc_status.sh` 列視窗、Slurm 工作、各 log 最後一行；`scripts/arc_grid.sh` 送出正式網格（升級 2010/2015/2019/2023、WAR 2010/2015/2019）。`DRY_RUN=1` 只印指令。

## 7. 報告與產出

- `reports/`：變數字典、補充剖析、清洗摘要、選秀連接、資料檢查、建模表摘要、球員頁摘要。全部進 git。
- `outputs/<season>/summary.md`、`eta_M1.parquet` 進 git；`idata_*.nc` 不進。
- `rawdata/`：非 Statcast 的所有表（約 140 MB），私人 repo。

## 8. 已知問題與注意

- B-Ref 每分鐘超過 20 次請求封一天；抓取只能單一程式跑。
- FanGraphs 新 API 的長期穩定性與條款未明，資料只發布衍生結果。
- 同一人可能出現在多個選秀年（未簽約再被選），`draft_linked` 取最後一次簽約。
- 最後兩三季的 `gone_exit` 偏高（還沒機會回來），建模時當設限。
- DSL 只從 2021 年起在資料內，所以 WAR 遮罩內沒有 DSL。
- PyMC 6.3.2 的 `compute_log_likelihood` 遇到 `LKJCorr` 會丟 `ValueError: exact match required`（RV 名 `factor_chol`、值變數名 `factor_chol_cholesky` 對不上）。`fit_year.compute_ll` 偵測到這個錯就改走 `compute_ll_manual`，照 PyMC 內部步驟重算並改名。單 factor 模型沒有這個變數，不受影響。
