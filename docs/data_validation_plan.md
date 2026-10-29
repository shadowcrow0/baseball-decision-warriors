# 企劃書：資料交叉驗證（Data Validation Plan）

> 2026-10-07 版。對應 `research_design.md` §2、`technical.md` §2，驗證對象是 `rawdata/` 現有各表。本文件只規劃，不改動既有管線；執行後產出 `reports/validation.md`。

## 1. 目的

| 原因 | 說明 |
|---|---|
| 競賽評分 | 評分表中「數據正確」占 10 分。目前只有 `reports/data_check.md` 的「與預期範圍比對」（0 WARN / 121 PASS），那是**自我一致性**，不是**對外一致性**。需要第二來源的逐筆比對才能主張正確。 |
| FanGraphs 條款不明 | 小聯盟與大聯盟主資料都來自 FanGraphs 新 JSON API。FanGraphs 服務條款寫明「不得以 Fangraphs 提供之介面以外的方式存取服務」、「不得重製、複製…任何部分」；社群套件（fungo）轉述 FanGraphs 的立場是自動化存取「不受支援、端點可無預警更改」。若端點變動或資料被下架，我們需要證明結果可由他源重現。 |
| 單一來源風險 | 計數、層級代碼、球員 id、首登年全部依賴一條管線。任何一處 bug（例如 2021 改制後層級對應、投手打擊列剔除）在單一來源內看不出來。 |
| 選秀簽約金覆蓋不足 | B-Ref 簽約金 2005–2011 只有 20–26% 的 pick 有值，2018 以後 79–100%。需要第二來源確認「缺值 = 未公開」而非抓取遺漏，並評估能否補值。 |

驗證不是為了換掉來源，而是量化「我們的表跟別人的表差多少」，把差異分類（對應錯、聚合錯、來源本身不同），能修就修、不能修就寫明。

## 2. 現有資料與來源一覽

數字取自 `rawdata/*.parquet`（2026-10-07）與 `reports/`。

| 資料集 | 檔案 | 來源 | 套件／方法 | 列數 | 年份 | 已知缺口 |
|---|---|---|---|---|---|---|
| 小聯盟打者（原始） | `minors_bat.parquet` | FanGraphs `api/leaders/minor-league/data` | `requests`，每請求 4 秒，JSON 快取 | 116,118 列 × 64 欄 | 2006–2026（無 2020） | 2006 無 Pitches／擊球類型；DSL 僅 2021 起；A-、R 於 2021 併為 CPX；獨立聯盟與海外球季不在內 |
| 小聯盟打者（清洗） | `minors_bat_clean.parquet` | 同上 | `clean_minors.py` | 101,483 列，20,995 人 | 同上 | 剔除 14,635 投手打擊列（規則：TBF > PA） |
| 小聯盟投手（原始／清洗） | `minors_pit*.parquet` | 同上 | 同上 | 121,959 → 117,767 列，24,443 人 | 同上 | 剔除 4,192 野手投球列 |
| 大聯盟打者 | `fg_batting.parquet` | FanGraphs `api/leaders/major-league/data` | `requests`（不可加瀏覽器 UA） | 28,287 列 × 473 欄 | 2005–2025 | qual=0 全收；WAR 無缺值 |
| 大聯盟投手 | `fg_pitching.parquet` | 同上 | 同上 | 15,520 列 × 542 欄 | 2005–2025 | 同上 |
| 選秀 | `draft.parquet` | Baseball-Reference 選秀頁（逐年逐輪） | pybaseball `BRefSession` + 自修 `read_html` | 23,514 picks | 2005–2025 | 簽約金整體 43.5%（2005–2011 約 20–26%，2018+ 79–100%）；B-Ref 頁無球員 id |
| 選秀連接 | `draft_linked.parquet` | 快取 HTML 的球員連結 → Chadwick | `link_draft.py` | 23,514 列，`key_mlbam` 100%（連結 23,509、姓名 5） | 同上 | 姓名配對 5 筆、同名多候選 182 筆未人工覆核 |
| 球員 id 對照 | `ids.parquet` | Chadwick register（GitHub raw ×16） | `requests` | 526,894 人（key_mlbam 135,208；key_fangraphs 21,200；key_bbref 24,014；mlb_played_first 23,803） | — | 新人 key_fangraphs 落後約一季；非官方週更 |
| 一人一季路徑表 | `player_season.parquet` | 衍生 | `clean_minors.py` | 160,290 人季 | 2006–2026 | `next_status`：promoted 42,992／same 40,837／gone_exit 32,421／mlb 29,430／censored 6,540／demoted 5,218／gone_return 2,852 |
| B-Ref 球員頁軌跡 | `register_paths.parquet` | B-Ref register 球員頁 | pybaseball `BRefSession`，每人一請求 | 117,664 列，2,738 人（僅 gone_return） | 1990–2026 | 只抓「離開後回來」者；**已含 g、PA/IP、lev、team**，可直接當小聯盟計數的驗證樣本（與 FG 打者表重疊 3,148 人季） |
| 建模表 | `model_bat.parquet` | 衍生 | `build_model_table.py` | 75,538 人季 | 2006–2026 | 2006 whiff/GB/PULL 設缺 |
| Statcast 逐球 | `data/statcast_<year>/`（Arc，未進 git） | Baseball Savant | pybaseball `statcast` 按週 | 每季約 71–77 萬球 | 2015–2026 | 第二階段用；可反推大聯盟 PA |

共同點：小聯盟與大聯盟計數的**原始提供者都是 MLB**（FanGraphs 頁尾：「Major League and Minor League Baseball data provided by Major League Baseball」；B-Ref 小聯盟 register 亦同源）。所以第二來源能驗證的是**處理鏈**（層級對應、聚合、剔除、id 連接），不是記錄員的原始判定。這一點要在 `reports/validation.md` 寫明。

## 3. 候選替代／驗證來源一覽

套件版本與日期來自 PyPI JSON API 與各官網（2026-10-07 查）。「Arc 可達」依 `technical.md`：校園網路 B-Ref、Chadwick、Savant 可過，FanGraphs 要走新 API；其他未實測者標「未知」。

### 3.1 資料來源

| 來源 | Python 取用路徑 | 小聯盟／大聯盟 | 免費／付費 | Arc 可達 | 2026 維護狀態 | 可驗證什麼 | 建議 |
|---|---|---|---|---|---|---|---|
| **Baseball-Reference WAR daily files** `data/war_daily_bat.txt`、`war_daily_pitch.txt` | 兩個純 CSV，`requests` 直接下載；欄位含 `mlb_ID`、`player_ID`（bbref）、`year_ID`、`PA`、`G`、`WAR`、`WAR_rep`、`WAA`、`salary` | 大聯盟全史 | 免費（B-Ref 主動提供的下載檔） | 可（B-Ref 可達；檔案 Last-Modified 2026-09-28，另有 2018 起每日 zip 存檔） | 每日更新 | fWAR vs bWAR 相關、大聯盟 PA、`key_mlbam↔key_bbref`、首登年 | **採用（第一優先）** |
| **Lahman / Baseball Databank 2025 版** | `pylahman` 0.7.0（PyPI 2026-01-08，Python ≥3.12，內建 2026-01-08 的 SABR 資料；`pylahman.Batting()`、`People()`）；或 SABR 官網 Box 連結下載 CSV。**注意**：`pybaseball.lahman` 指向的 GitHub `chadwickbureau/baseballdatabank` 已 404/403，不能用 | 大聯盟 1871–2025；**無小聯盟** | 免費，CC BY-SA 3.0 | 可（pip 安裝即含資料，不需網路；Box 下載未知） | 2026-01-02 發布 2025 版，年更 | 大聯盟 G/AB/H/BB/SO/HBP/SH/SF（PA 可推）、`People.debut` 首登日、`bbrefID↔retroID` | **採用（第二優先）** |
| **Baseball-Reference 小聯盟 register**（球員頁 `register/player.fcgi`、球隊頁 `register/team.cgi`、組織頁 `register/affiliate.cgi?id=NYY&year=2019`） | 現有 `fetch_register.py` 模式：`BRefSession` 10 次/分、HTML 註解內表格、磁碟快取；已有 2,738 人頁快取 | 小聯盟全層級（含 Ind、Fgn、FgW、NCAA） | 免費；Sports-Reference 資料使用政策禁止大量抓取、每分鐘 >20 次封 IP、客製資料最低 5,000 美元 | 可 | 持續 | 小聯盟逐人逐季逐層級 G、PA/IP；層級代碼對應；gone_exit 是否其實去了獨立聯盟 | **採用（第三優先）**，只做分層抽樣，不全抓 |
| **MLB StatsAPI**（`statsapi.mlb.com/api/v1/draft/{year}`、`/stats?sportId=11..16`） | `python-mlb-statsapi` 1.1.2（PyPI 2026-09-29，Python ≥3.10）或 `MLB-StatsAPI` 1.9.0（2025-04-04）或裸 `requests` | 小聯盟 + 大聯盟 + 選秀 | 免費，無金鑰；條款未明 | 未知（雲端測試 OK） | 兩套件皆活躍 | 選秀：`pickNumber`、`pickValue`、`signingBonus`、`person.id`（MLBAM）、`school`（實測 2019 第 1 順位回傳完整）；小聯盟季計數 | **僅作參考對照**，不當主來源（使用者限制）；報告中明示 |
| Chadwick register（已在用） | 現有 `fetch_ids.py` | id 對照、首末年 | 免費，ODC-By | 可 | 約週更；公開版有延遲 | 本身是被驗對象之一；可用重新下載版比對漂移 | 已採用 |
| FanGraphs 球員頁 API `api/players/stats?playerid=` | `requests` | 同 FG | 同 FG | 同 FG | 同 FG | 只能做 FG 內部一致性（層級列加總 = 球員頁） | 低優先，可省 |
| MLB.com 選秀簽約金追蹤文章（2016–2025 各一篇） | HTML，人工或 `read_html` | 選秀 | 免費 | 未知 | 每年一篇 | 簽約金人工抽查 50 筆 | 備用（底層同 StatsAPI） |
| The Baseball Cube（選秀名冊含簽約金、小聯盟統計） | 網站 Cloudflare 擋 `curl`；資料在付費商店，價格未取得 | 小聯盟 + 選秀 1965– | 付費 | 未知 | 持續 | 簽約金歷史覆蓋最完整的商業來源 | 不採用（付費、無程式介面） |
| Baseball America 選秀資料庫 | 訂閱頁面，無 API | 選秀 | 付費訂閱 | 未知 | 持續 | 簽約金 | 不採用 |
| Baseball Prospectus（DRC+、WARP、小聯盟 DRC+ 自 2005） | 訂閱者可自訂報表、CSV；Premium 45.95 美元/年 | 小聯盟 + 大聯盟 | 付費 | 未知 | 持續 | 另一套 WAR（WARP）、小聯盟進階指標 | 不採用（付費、條款） |
| Smart Fantasy Baseball Player ID Map `/PLAYERIDMAPCSV` | 伺服器 Mod_Security 擋非瀏覽器請求（406）；需瀏覽器手動下載 | 僅「fantasy 相關」大聯盟球員，明示排除小聯盟 | 免費 | 未知 | 人工維護 | 約 2–3 千人的 MLBAM↔FG↔BBREF 抽查 | 低優先，可省 |
| Retrosheet（事件檔） | `pychadwick` 0.6.1（2023-07）、`pyretrosheet` 0.0.10（2024-02） | 大聯盟逐球；**無小聯盟** | 免費，需署名 | 未知 | 半年更（2025 秋季版含 2025 季） | 大聯盟 PA 計數（Statcast、Lahman 已覆蓋） | 不採用（重複） |
| Kaggle | — | 搜尋未見維護中的小聯盟逐季資料集；Lahman 鏡像只到 2015 | 免費 | — | — | — | 不採用 |
| MiLB.com 統計頁 | JS 前端，底層為 MLB StatsAPI | 小聯盟 | 免費 | 未知 | 持續 | 等同 StatsAPI | 併入 StatsAPI 參考項 |

### 3.2 Python 套件現況（供「Funko」比對與工具選擇）

| 套件 | 最新版／上傳日 | Python | 來源覆蓋 | 2026 可用性 | 備註 |
|---|---|---|---|---|---|
| `pybaseball` | 2.2.7 / 2023-09-08（GitHub releases 標 2024-09） | — | FG、B-Ref、Savant、Lahman、Chadwick | FG 舊端點 `leaders-legacy.aspx` 全球 403（issue #479，2025-05 開啟至今無回應）；`lahman()` 的 GitHub 來源 404；Savant、B-Ref session、Chadwick 仍可用 | 我們已自行繞過，只用其 `BRefSession` 與 `statcast` |
| **`fungo`** | 2.0.0 / 2026-07-11 | ≥3.12 | Statcast、MLB StatsAPI、FanGraphs（用 FG 手機 app 的 okhttp UA 過 Cloudflare）、B-Ref（WAR daily、選秀頁 `get_draft`、register `get_register_player`、內建 9 次/分限速與快取）、Retrosheet、Lahman（解析 SABR Box 連結）、Chadwick 查表 | 活躍；MIT；僅 2 個依賴（bs4、curl_cffi）；GitHub 3 顆星，單人維護 | **「Funko」最可能的本尊**（見 §8）。覆蓋面與本計畫完全重合，但 FG 部分靠假 UA、StatsAPI 部分違反使用者限制，建議只借用其 B-Ref WAR／Lahman 取法 |
| `fangraphs` | 1.1.0 / 2021-03-30 | ≥3.6 | FG 網頁（playwright 瀏覽器自動化） | 停更 5 年，依賴 playwright | 不採用 |
| `baseball_scraper` | 0.4.10 / 2020-03-15 | — | B-Ref、Savant | 停更 6 年 | 不採用 |
| `mlbgame` | 2.5.0 / 2018-04-16 | — | MLB GameDay XML | 底層 XML API 已被 MLB 關閉，功能失效 | 不採用 |
| `plyball` | 2.6.15 / 2025-06-24 | 3.10–3.13 | FG、Statcast 等（多為爬蟲） | 活躍，單人維護 | 不採用（與 FG 條款問題相同） |
| `pybbda` | 0.4.3 / 2023-07-15 | — | Lahman、B-Ref WAR、FG、Retrosheet、Statcast | 停更 3 年 | 不採用，但其 B-Ref WAR 讀法可參考 |
| `baseball-stats-python` | 0.3.0 / 2026-06-25 | ≥3.8 | Statcast、StatsAPI | 活躍 | 不需要 |
| `pylahman` | 0.7.0 / 2026-01-08 | ≥3.12 | Lahman 2025 版（內建） | 活躍 | **採用** |
| `python-mlb-statsapi` | 1.1.2 / 2026-09-29 | ≥3.10 | StatsAPI | 活躍 | 僅參考對照時可用；裸 `requests` 即可，不必裝 |
| `MLB-StatsAPI`（toddrob） | 1.9.0 / 2025-04-04 | — | StatsAPI | 活躍 | 同上 |
| `pychadwick` / `pyretrosheet` | 0.6.1 (2023-07) / 0.0.10 (2024-02) | — / ≥3.11 | Retrosheet 事件檔 | 可用 | 不需要 |
| `funko` | 0.0.1 / 2021-08-02 | — | 無（說明「Myfuncolandia」） | 與棒球無關 | 排除 |
| `fanko`、`punko`、`fangraphs-api`、`baseballdatabank`、`retrosheet` | PyPI 不存在 | — | — | — | — |

## 4. 驗證設計

通用規則：
- 比對單位一律是「一人一季（一層級）」，鍵為 `key_mlbam` + `season`（+ `level`），不用姓名。
- 每項檢查輸出三個數：**覆蓋率**（兩邊都有的比例）、**一致率**（在容忍度內的比例）、**差異分布**（分位數）。
- 不一致先分類再處置：(a) 對應錯（層級代碼、id）→ 修程式；(b) 聚合錯（跨隊、跨層加總）→ 修程式；(c) 來源定義不同（G 的計法、PA 含不含妨礙打擊）→ 寫明不修；(d) 來源本身錯 → 標記 `qc_flag`，不刪列。
- 容忍度達標寫 PASS，未達標寫 FAIL 並附前 20 筆差異最大的列。

### 4.1 小聯盟計數（`minors_bat_clean` / `minors_pit_clean`）

| 項目 | 內容 |
|---|---|
| 第二來源 | B-Ref 小聯盟 register。**階段 0** 用已快取的 2,738 人球員頁（`register_paths.parquet`，與 FG 打者表重疊 3,148 人季，另有投手列）；**階段 1** 分層抽樣球隊頁：每年 × 每層級抽 2 隊（7 層級 × 2 × 19 季 ≈ 266 頁），得該隊全部球員的 G、PA、H、BB、SO（打者）與 G、IP、BF、H、BB、SO（投手） |
| 連接鍵 | `key_mlbam`（FG `xMLBAMID`）→ Chadwick → `key_bbref_minors`；球隊頁用 B-Ref 球員連結取 `key_bbref_minors` 反查 |
| 層級對應表 | B-Ref `AAA/AA/A+/A/A-/Rk/FRk` ↔ FG `AAA/AA/A+/A/A-/R或CPX/DSL`；2021 起 B-Ref `Rk` = FG `CPX`、`FRk` = `DSL`。對不上的列單獨計數 |
| 檢查 1：列數 | 每季 × 層級：階段 1 抽樣球隊的球員人數 FG vs B-Ref，容忍 ±2 人（投手打擊列、兩棲球員） |
| 檢查 2：逐人計數 | PA、H、BB、SO（打者）；BF、SO、BB（投手）：**完全相等 ≥ 98%，|差| ≤ 1 的比例 ≥ 99.5%**。IP 以 1/3 局為單位，|差| ≤ 0.1 ≥ 99% |
| 檢查 3：G | 完全相等 ≥ 97%（兩站對代打、代跑出賽計法可能不同，門檻放寬） |
| 檢查 4：層級 | 同一人季在 FG 的層級集合 = B-Ref 的層級集合 ≥ 97%；差異集中在 2021 改制前後則歸類 (a) |
| 檢查 5：剔除規則 | 階段 0 樣本中被 `clean_minors.py` 剔除的「投手打擊列」，在 B-Ref 是否確為投手（有投球列）≥ 99% |
| 不達標時 | 先看是否集中在某年／某層級（對應錯）→ 修 `LEVEL_RANK` 或 `clean_minors.py`；若隨機散佈且差 ≤ 1 → 寫明為來源更新時差 |

### 4.2 大聯盟 fWAR 與結果變數（`fg_batting` / `fg_pitching` / `war_fwd`）

| 項目 | 內容 |
|---|---|
| 第二來源 | B-Ref `war_daily_bat.txt`、`war_daily_pitch.txt`（主）；Lahman `Batting`、`Pitching`（計數） |
| 連接鍵 | WAR 檔直接有 `mlb_ID` = `key_mlbam`；多 stint 先按 `year_ID` 加總。Lahman 用 `key_bbref` = `playerID` |
| 檢查 1：覆蓋 | 2005–2025 每季 FG 有而 B-Ref 無（或反之）的人數 ≤ 0.5%（FG qual=0 應幾乎全收） |
| 檢查 2：PA / BF | FG `PA` vs B-Ref `PA`：完全相等 ≥ 99%。Lahman 無 PA 欄，以 AB+BB+HBP+SH+SF 近似（少妨礙打擊），|差| ≤ 1 ≥ 99% |
| 檢查 3：WAR 相關 | 逐人季 fWAR vs bWAR：打者 Pearson r ≥ 0.85、投手 ≥ 0.75（投手 fWAR 以 FIP 為基礎，與 RA9 基礎的 bWAR 本就差較多）；報告散佈圖與斜率。**不是**要兩者相等 |
| 檢查 4：結果變數 | 以 bWAR 重算 `war_fwd`（本季+1 到 +6 合計）與 `reached_fwd`：`reached_fwd` 一致率 ≥ 99.5%；`war_fwd` 的 Spearman ρ ≥ 0.85；若模型主結論（負荷量差距）對換成 bWAR 敏感，寫入限制 |
| 檢查 5：Statcast 反推 | 2015–2025 由 Statcast 逐球取每人季最後一球事件數 ≈ PA，與 FG PA |差| ≤ 2 ≥ 98%（已在磁碟，零請求） |
| 不達標時 | PA 不等 → 查 FG 多隊球員是否重複列；WAR 相關低於門檻 → 分守位看（DH／捕手框架差異），寫明 |

### 4.3 選秀表（`draft` / `draft_linked`，含順位與簽約金）

| 項目 | 內容 |
|---|---|
| 第二來源 | MLB StatsAPI `api/v1/draft/{year}`（**僅參考對照**，每年一請求，共 21 次）；MLB.com 簽約追蹤文章人工抽 50 筆（2016+） |
| 連接鍵 | `year` + `pick_overall` ↔ `pickNumber`；再比 `person.id` ↔ `key_mlbam`（檢驗 `link_draft.py` 的連結與姓名配對） |
| 檢查 1：名單 | 每年 pick 數相等；`pick_overall` 1..N 連續無缺（補償順位編號方式若不同，寫對應） |
| 檢查 2：id | `person.id` = `key_mlbam` ≥ 99%；不等者列出，重點看 `link_source == 'name'` 的 5 筆與同名多候選 182 筆 |
| 檢查 3：簽約金 | 兩邊皆有值時完全相等 ≥ 95%（兩邊都可能四捨五入到千元，|差| ≤ 1,000 美元 ≥ 99%）；逐年列出 B-Ref 缺值但 StatsAPI 有值的比例 → 決定是否開新欄 `bonus_ref` 補值（補值欄只作敏感度分析，主模型仍用 B-Ref） |
| 檢查 4：來源類型 | `level`（HS/4Yr/JC）vs `school` 類型 ≥ 97% |
| 檢查 5：生涯 bWAR | B-Ref 選秀頁 `bwar_career` vs WAR daily 檔同人加總 |差| ≤ 0.2 ≥ 98%（確認選秀頁抓取日與 WAR 檔版本一致） |
| 不達標時 | id 不等 → 修 `link_draft.py` 或人工覆核；簽約金差 → 寫明以 B-Ref 為主、StatsAPI 僅參考 |

### 4.4 球員 id（`ids` 與各表的 `key_mlbam`）

| 項目 | 內容 |
|---|---|
| 第二來源 | B-Ref WAR 檔（`mlb_ID`↔`player_ID`）、Lahman `People`（`bbrefID`、`retroID`、`debut`）、FG 小聯盟表自帶的 `playerids`（= key_fangraphs）與 `minormasterid`、重新下載的 Chadwick 當日版 |
| 檢查 1：MLBAM↔BBREF | WAR 檔 `(mlb_ID, player_ID)` 對 vs Chadwick `(key_mlbam, key_bbref)`：一致 ≥ 99.9% |
| 檢查 2：MLBAM↔FG | 小聯盟表 `(key_mlbam, playerids)` vs Chadwick `(key_mlbam, key_fangraphs)`：一致 ≥ 99%；不一致者看是否為 Chadwick 落後的新人 |
| 檢查 3：唯一性 | 同一 `key_mlbam` 對到 ≥ 2 個 `playerids`、或同一 `playerids` 對到 ≥ 2 個 `key_mlbam` 的筆數 = 0；非 0 則列出 |
| 檢查 4：首登年 | Chadwick `mlb_played_first` vs Lahman `debut` 年 vs FG 大聯盟表首季（PA>0 或 IP>0）vs WAR 檔首季：四者一致 ≥ 99.5% |
| 檢查 5：漂移 | 重新下載 Chadwick，與 `ids.parquet` 比 `key_mlbam` 集合與 `mlb_played_first` 變動筆數（報告用，不設門檻） |

### 4.5 下一季去向（`player_season.next_status`）

| 項目 | 內容 |
|---|---|
| 第二來源 | B-Ref register（階段 0 的 2,738 人 + 新抽 200 名 `gone_exit`、200 名 `promoted`、100 名 `demoted` 的球員頁，共 500 請求）；首登年用 §4.4 檢查 4 的結果 |
| 連接鍵 | `key_mlbam` + `season` |
| 檢查 1：mlb | `next_status == 'mlb'` 的人季，B-Ref 該年或次年有 `Maj` 列 ≥ 99% |
| 檢查 2：promoted / demoted / same | 以 B-Ref 層級（對應表同 §4.1）重算 `next_max_rank` 比較，一致 ≥ 95% |
| 檢查 3：gone_exit | 200 名樣本中，B-Ref 顯示該人次季其實在 `Ind`、`Fgn`、`FgW`、`Maj` 的比例 → 這不是錯，而是**量化「離開 = 離開所屬體系」的定義**，寫進 `research_design.md` §10 的限制；若發現 ≥ 5% 其實仍在所屬體系（FG 漏人），則為 FAIL，查 FG 的 `lg` 參數清單 |
| 檢查 4：gone_return 間隔 | `gap_years` 與 B-Ref 空窗年數一致 ≥ 95%（階段 0 已有全部資料） |
| 檢查 5：censored | 2026 季為最後一季者比例合理（與 2025 相近） |

## 5. 執行計畫

### 5.1 腳本

| 順序 | 腳本 | 來源請求 | 輸入 | 輸出 | Arc 估時 |
|---|---|---|---|---|---|
| 1 | `src/validate_ids.py` | 重新下載 Chadwick 16 個 CSV（約 16 請求） | `ids.parquet`、`minors_*_clean`、`fg_*` | `reports/validation/ids.md`、`ids_mismatch.csv` | 3 分 |
| 2 | `src/validate_war_bref.py` | 下載 2 個 WAR 檔（2 請求，數十 MB） | `fg_batting`、`fg_pitching`、`model_bat`、`draft` | `war.md`、`war_scatter.png`、`war_fwd_bwar.parquet` | 5 分 |
| 3 | `src/validate_mlb_lahman.py` | `pip install pylahman`（Python ≥3.12，Arc 為 3.12） | `fg_*`、`ids` | `lahman.md` | 5 分 |
| 4 | `src/validate_minors_bref.py --phase 0` | 0（用 `cache/register/` 既有 2,738 頁） | `register_paths`、`minors_*_clean`、`player_season` | `minors_phase0.md` | 3 分 |
| 5 | `src/validate_minors_bref.py --phase 1` | 約 266 球隊頁 + 500 球員頁，`BRefSession` 6–7 秒一請求 | 同上 | `minors_phase1.md`、`cache/register_teams/` | 80–90 分 |
| 6 | `src/validate_draft_ref.py` | StatsAPI 21 請求（參考用，程式與報告都標註） | `draft_linked` | `draft.md`、`bonus_ref.parquet` | 3 分 |
| 7 | `src/validate_statcast_pa.py` | 0（讀 `data/statcast_*`，約 1 GB） | Statcast、`fg_batting` | `statcast_pa.md` | 10 分 |
| 8 | `src/validate_report.py` | 0 | 以上所有 | `reports/validation.md`（合併）、`reports/validation_summary.csv` | 1 分 |

每支腳本：讀 `rawdata/`（或 `data/`），不改寫任何既有 parquet；新增欄位一律寫到 `derived/qc_*.parquet`。

### 5.2 請求速率與平行

| 來源 | 限制 | 處置 |
|---|---|---|
| Baseball-Reference | 每分鐘 >20 次封 IP 一天；同一 IP 只能有一支程式在抓 | 腳本 2 與 5 **串行**，且不可與 `fetch_register.py`、`fetch_draft.py` 同時跑；沿用 `BRefSession` 與磁碟快取；403/429 立即停，不重試 |
| Chadwick（GitHub raw） | 無明顯限制 | 可平行 |
| MLB StatsAPI | 未公布；21 請求可忽略 | 可平行 |
| FanGraphs | **本計畫不新增任何 FG 請求** | — |
| Lahman | 本機套件 | 可平行 |

`screen` 分三個視窗（compute 節點 `srun -p compute1 -n 1 -t 04:00:00 --cpus-per-task=4`）：

```
視窗 A（B-Ref 串行）：  validate_war_bref.py → validate_minors_bref.py --phase 0 → --phase 1
視窗 B：               validate_ids.py → validate_mlb_lahman.py → validate_draft_ref.py
視窗 C：               validate_statcast_pa.py
全部結束後：           validate_report.py
```

總牆鐘時間約 1.5–2 小時，主要是視窗 A。

### 5.3 前置

- `requirements.txt` 加 `pylahman>=0.7`（不加 `fungo`、不加任何 StatsAPI 套件；StatsAPI 用裸 `requests`，程式頂端註明「reference only」）。
- `check_sources.py` 加兩項：`war_daily_bat.txt` HEAD 請求、`register/affiliate.cgi` 一頁，確認 Arc 可達。

## 6. 產出

### 6.1 `reports/validation.md` 結構

| 節 | 內容 |
|---|---|
| 0. 總表 | 一張表：每個檢查項 → 第二來源、覆蓋率、一致率、門檻、PASS/FAIL |
| 1. 小聯盟計數 | 階段 0 與階段 1 的一致率表（× 層級 × 年代）、差異分布、層級對應問題清單 |
| 2. 大聯盟 WAR | fWAR vs bWAR 散佈圖（打者、投手各一）、r、斜率；`war_fwd` 兩版的 ρ；PA 一致率 |
| 3. 選秀 | id 一致率、簽約金一致率 × 年、B-Ref 缺值而參考來源有值的比例 × 年 |
| 4. 球員 id | 四來源首登年一致率、不一致清單 |
| 5. 去向 | gone_exit 樣本的實際去處分布（Ind／Fgn／Maj／真消失）、promoted 重算一致率 |
| 6. 差異分類與處置 | 每筆 FAIL 的分類 (a)–(d)、已修／不修的理由、影響的人季數 |
| 7. 來源與條款 | 各來源取用方式、日期、版本、條款摘要；StatsAPI 僅參考的聲明 |

### 6.2 簡報中的一頁（「數據正確」證據）

一頁，標題「資料經三方交叉驗證」。左半：總表精簡版（5 列：小聯盟計數、大聯盟 PA、fWAR↔bWAR、選秀 id／簽約金、首登年；各列一個一致率）。右半：fWAR vs bWAR 散佈圖（打者）。底部一行：「原始記錄皆來自 MLB；驗證的是處理鏈與 id 連接；StatsAPI 僅作參考對照」。圖與表由 `validate_report.py` 直接輸出 PNG 與 markdown，簡報不另手填數字。

## 7. 風險與不做的事

| 項目 | 決定 | 理由 |
|---|---|---|
| MLB StatsAPI 當主來源 | **不做**。只在 §4.3 當參考對照，報告與程式皆標註 | 使用者限制 |
| Baseball Prospectus、Baseball America、The Baseball Cube | **不做** | 付費／訂閱，無程式介面；簽約金的補值價值不足以換取條款風險 |
| R-only 來源（`baseballr` 的 `fg_milb_*`、`Lahman` R 套件） | **不做** | 純 Python 限制；資料面皆有 Python 等價物（Lahman → `pylahman`） |
| Retrosheet、Kaggle | **不做** | Retrosheet 無小聯盟且與 Lahman／Statcast 重複；Kaggle 無維護中的小聯盟資料集 |
| 新增 FanGraphs 請求 | **不做** | 條款不明；驗證目的之一就是減少對 FG 的依賴 |
| B-Ref 全量抓球隊頁（≈4,500 頁） | **不做**，只抽 266 頁 + 500 球員頁 | Sports-Reference 政策禁止大量抓取；抽樣已足以估一致率（每層級每年 ≥ 50 人） |
| `fungo` 直接引入 | **不做**，只參考其 B-Ref WAR／Lahman 取法 | 其 FG 模組用手機 app UA 繞 Cloudflare、StatsAPI 模組違反限制；單人維護、3 顆星，長期穩定性未知 |
| 用 bWAR 取代 fWAR 當結果變數 | **不做**，只做敏感度 | 研究設計已定 fWAR；驗證是量化差異，不是換變數 |
| 風險：B-Ref 封 IP | 視窗 A 一旦 403 立即停並等 24 小時；階段 0 不需網路，可先交付 | — |
| 風險：Arc 到 StatsAPI 不通 | 退回 MLB.com 簽約追蹤文章人工抽 50 筆 | — |
| 風險：原始同源 | 小聯盟計數 FG 與 B-Ref 皆源自 MLB，一致率高不代表記錄無誤；報告寫明 | — |

## 8. 附註：關於「Funko」

使用者口述的套件名「Funko」在 PyPI 與常見棒球資料工具中都對不上，推測是語音轉文字誤差。查到的候選：

| 候選 | 出處 | 相似度 | 說明 |
|---|---|---|---|
| **`fungo`**（PyPI 2.0.0，2026-07-11） | https://pypi.org/project/fungo/ 、GitHub hedgertronic/fungo | 發音最接近（fungo = 棒球練習打擊用的「fungo bat」） | Python ≥3.12，MIT。覆蓋 Statcast、MLB StatsAPI、FanGraphs、Baseball-Reference（WAR daily、選秀頁、小聯盟 register）、Retrosheet、Lahman、Chadwick 查表，正好是本計畫的所有來源。注意：FG 部分靠偽裝 UA；StatsAPI 部分與使用者限制衝突 |
| `funko`（PyPI 0.0.1，2021-08-02） | https://pypi.org/project/funko/ | 拼法完全相同 | 說明文字「Myfuncolandia」，與棒球無關，排除 |
| **FanGraphs** | — | 語音上「Fan-graphs」與「Funko」兩音節相近 | 若使用者指的是「用 FanGraphs 的其他端點／套件驗證」，則對應 §3.2 的 `fangraphs`（已停更）或 FG 球員頁 API（同源，只能做內部一致性） |
| `pybbda` | PyPI 0.4.3，2023-07 | 低 | 「Python baseball data and analysis」，覆蓋 Lahman、B-Ref WAR、FG、Retrosheet、Statcast；停更 |
| `plyball` | PyPI 2.6.15，2025-06 | 低 | FG／Statcast 爬蟲 |
| `fanko`、`punko`、`phunko` | — | — | PyPI 不存在；網路搜尋無棒球相關結果 |

**請使用者確認**：您說的是 (1) `fungo` 這個套件、(2) FanGraphs 本身、還是 (3) 其他拼法？若是 (1)，本計畫建議只借用它的 B-Ref WAR daily 與 Lahman 取法（兩者都是公開下載檔，我們用 `requests`／`pylahman` 自行實作即可，不需引入套件）。
