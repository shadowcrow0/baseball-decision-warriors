# 主流賽伯計量模型的定義彙整：公式、輸入與常數校準方式

用途：釐清標準 sabermetric 模型「如何被定義」，以便定位我們的小聯盟打者貝氏潛在能力模型。所有定義皆附來源 URL；逐年變動的常數標明年份。查詢日期 2026-10-22。

---

## 1. Sabermetrics 的定義與起源

- **定義**：Bill James 於 1980 年定義 sabermetrics 為 *"the search for objective knowledge about baseball"*（對棒球客觀知識的探求）。
- **詞源**：James 創此詞部分是為了向 SABR（Society for American Baseball Research）致敬；SABR 的 Statistical Analysis Committee 由 James、Pete Palmer、Dick Cramer 於 1974 年共同成立。SABR 指出此詞已屬公共領域，泛指任何對棒球的數學／統計研究。
- 來源：https://sabr.org/sabermetrics

---

## 2. 得分估計式（Run Estimators）

### 2.1 Runs Created（Bill James，基本型）
- (a) 衡量一名打者「創造」的總得分。
- (b) 基本式：`RC = TB × (H + BB) / (AB + BB)`，即「上壘因子 × 推進因子 ÷ 機會數」。
- (c) 無逐年擬合的權重；為乘法型（非線性）會計式，進階版（技術型）另加 SB、GIDP 等項。
- 來源：https://www.mlb.com/glossary/advanced-stats/runs-created

### 2.2 Linear Weights（Pete Palmer）
- (a) 以每一事件的「平均得分價值」線性加總，估計打者相對聯盟平均的得分貢獻（Batting Runs）。
- (b) Palmer 1984 年版（*The Hidden Game of Baseball*）係數：`1B=.47, 2B=.83, 3B=1.02, HR=1.40, BB=.33`；出局（AB−H）係數設定為使「扣除投手打擊後的聯盟總和 = 0」，通常約 `−.25`。
- (c) **推導法**：以 24 個「壘包–出局數」狀態（8 壘包狀態 × 3 出局數）建立 run expectancy（RE）矩陣；每一打席的價值 = 事件後狀態 RE − 事件前狀態 RE（+ 該打席得分）；對同類事件取平均即得其線性權重（RE24 法）。FanGraphs 以 2015 年為例：相對出局，`BB=.55, HBP=.57, 1B=.70, 2B=1.00, 3B=1.27, HR=1.65`。Palmer 早期則用世界大賽逐球紀錄加模擬求得。
- 來源：https://sabr.org/journal/article/why-ops-works/ ；https://library.fangraphs.com/principles/linear-weights/

### 2.3 BaseRuns（David Smyth）
- (a) 團隊層級得分估計，結構模仿得分過程：上壘者 × 得分率 + 保證得分。
- (b) `Runs = A × B/(B+C) + D`。基本版：`A = H + BB − HR`；`B = (1.4·TB − .6·H − 3·HR + .1·BB) × 1.02`；`C = AB − H`；`D = HR`。
- (c) `B/(B+C)` 為「上壘者得分比例」的經驗近似，Tango wiki 明言「無深層理論，因實證有效而採用」。乘數（1.02 / 1.1）為校準係數。因公式非線性，**隱含權重隨球隊表現而變**，與固定線性權重不同。
- 來源：https://tangotiger.net/wiki_archive/Base_Runs.html

### 2.4 Pythagorean Expectation 與 Pythagenpat
- (a) 由得分／失分估計球隊預期勝率。
- (b) 原式（Bill James）：`W% = RS² / (RS² + RA²)`；Baseball-Reference 改用指數 **1.83**。
- (c) **Pythagenpat**（B-R 用於 WAR 之 runs→wins 轉換）：`W% = RS^x / (RS^x + RA^x)`，`x = (兩隊合計每場得分)^0.285`；例：雙方各 4.5 分/場 → `x = 9^0.285 ≈ 1.87`。
- 來源：https://www.mlb.com/glossary/advanced-stats/pythagorean-winning-percentage ；https://www.baseball-reference.com/about/war_explained_runs_to_wins.shtml

---

## 3. wOBA 家族（Tom Tango）

### 3.1 wOBA
- (a) 以得分價值加權的上壘率，將所有打擊事件放在 OBP 尺度上。
- (b) 2013 年權重（FanGraphs）：
  `wOBA = (0.690·uBB + 0.722·HBP + 0.888·1B + 1.271·2B + 1.616·3B + 2.101·HR) / (AB + BB − IBB + SF + HBP)`
- (c) **權重推導**：各事件之 RE24 線性權重（相對出局）→ 以該權重算出聯盟 wOBA → 乘上 **wOBA Scale** 使「聯盟 wOBA = 聯盟 OBP」。2015 年例：未縮放之聯盟值 .250、聯盟 OBP .313 → scale = 1.251。權重**逐年重算**；FanGraphs Guts! 頁 2025 年值：`wBB=.691, wHBP=.722, w1B=.882, w2B=1.252, w3B=1.584, wHR=2.037, wOBA Scale=1.232, lg wOBA=.313`。
- 來源：https://library.fangraphs.com/offense/woba/ ；https://library.fangraphs.com/principles/linear-weights/ ；https://www.fangraphs.com/guts.aspx?type=cn

### 3.2 wRAA、wRC、wRC+
- `wRAA = ((wOBA − lgwOBA) / wOBA Scale) × PA`（高於聯盟平均的得分）
- `wRC = (((wOBA − lgwOBA) / wOBA Scale) + lgR/PA) × PA`（絕對得分）
- `wRC+ = (((wRAA/PA + lgR/PA) + (lgR/PA − PF × lgR/PA)) / (AL 或 NL 不含投手之 wRC/PA)) × 100`
  其中 PF 為球場因子（頁面數值 ÷100，如 98 → 0.98）；分母為該聯盟（AL/NL）扣除投手打擊之 wRC/PA，構成聯盟調整。100 = 聯盟平均。
- 來源：https://library.fangraphs.com/offense/wrc/ ；https://library.fangraphs.com/offense/woba/

---

## 4. WAR

### 4.1 FanGraphs fWAR（野手）
`WAR = (Batting Runs + Base Running Runs + Fielding Runs + Positional Adjustment + League Adjustment + Replacement Runs) / Runs Per Win`

| 成分 | 定義 |
|---|---|
| Batting Runs | `wRAA + (lgR/PA − PF·lgR/PA)·PA + (lgR/PA − lg非投手wRC/PA)·PA`（球場＋聯盟調整） |
| Base Running (BsR) | `UBR + wSB + wGDP`。`wSB = SB·runSB + CS·runCS − lgwSB·(1B+BB+HBP−IBB)`，`runSB` 固定 .2，`runCS = 2·(Runs/Out) + .075`（2025：−.410）。UBR：每一非盜壘跑壘事件之「實際 RE 變化 − 該壘包出局狀態平均 RE 變化」，以 BIS 影像資料計算 |
| Fielding Runs | 非捕手用 **UZR**（RngR + ErrR + ARM + DPR；每顆球以「平均該守位球員接到之機率 × 該球得分價值」計貢獻，相對守位平均）；捕手用 rSB + RPP，不含 framing |
| Positional Adjustment | `((守備局數/9)/162) × 守位值`；每 162 場：C +12.5, SS +7.5, 2B/3B/CF +2.5, LF/RF −7.5, 1B −12.5, DH −17.5。由「換守位球員表現」之舊研究估得，FanGraphs 稱其為「估計值」 |
| League Adjustment | `(−1)·(lgBatting + lgBsR + lgFielding + lgPositional)/lgPA × PA`，使各聯盟總和歸零（每季約 0–5 分） |
| Replacement Runs | `570 × (MLB Games/2430) × (RPW/lgPA) × PA`。替代水準：替代球員球隊勝率約 .294（約 47–48 勝），全聯盟 2430 場共 **1000 WAR**，**57%** 歸野手（570）、43% 歸投手 |
| Runs Per Win | `RPW = 9 × (MLB Runs/MLB IP) × 1.5 + 3`，通常 9–10（2025：9.774；2023：10.028） |

- 來源：https://library.fangraphs.com/war/war-position-players/ ；https://library.fangraphs.com/misc/war/replacement-level/ ；https://library.fangraphs.com/misc/war/positional-adjustment/ ；https://library.fangraphs.com/misc/war/converting-runs-to-wins/ ；https://library.fangraphs.com/defense/uzr/ ；https://library.fangraphs.com/offense/ubr/

### 4.2 Baseball-Reference bWAR（野手）差異
- 成分：**Rbat**（以 Tango wOBA 架構之線性權重算 wRAA）、**Rbaser**（約 .20 分/壘、.48 分/出局）、**Rdp**（`.44 × (GIDP機會 × 聯盟GIDP率 − 球員GIDP)`）、**Rfield**（2003 起用 **DRS**，之前用 Total Zone）、**Rpos**（每 1350 局：C +9, SS +7, 2B +3, CF +2.5, 3B +2, LF/RF −7, 1B −9.5, DH −15）、**Rrep**。
- 替代水準同為 .294 勝率、`30×162×(.500−.294) = 1000 WAR`，但拆分為 **59% 野手／41% 投手**；野手替代乘數約 20.5 分/600 PA。
- runs→wins 用 **PythagenPat**（見 2.4）而非固定 RPW；`WAR = WAA + WAR_rep`。
- 與 fWAR 之主要差異：守備 DRS vs UZR；投手 RA9（加球隊守備調整）vs FIP。
- 來源：https://www.baseball-reference.com/about/war_explained_position.shtml ；https://www.baseball-reference.com/about/war_explained.shtml ；https://www.baseball-reference.com/about/war_explained_comparison.shtml ；https://library.fangraphs.com/war/differences-fwar-rwar

### 4.3 Baseball Prospectus WARP（簡述）
打擊用 DRC+、守備用 FRAA、投手用 DRA（混合模型估計之「應得失分」）；替代水準不以特定球隊勝率設定。來源：https://library.fangraphs.com/war/differences-fwar-rwar ；https://www.baseball-reference.com/about/war_explained_comparison.shtml

---

## 5. 投手指標

- **DIPS 理論**（Voros McCracken，1999 Usenet／2001 年 1 月 Baseball Prospectus）：投手對 K、BB、HR 有控制力，但對擊球進場安打率（BABIP）幾乎無差異（年際相關 r = 0.153）；BABIP 反映守備與球場。後續研究（Tippett、Lichtman）發現投手對 BABIP 有微弱影響。原始 DIPS ERA 以 Extrapolated Runs 計算預期失分再乘 0.9297。來源：https://sabr.org/journal/article/the-many-flavors-of-dips-a-history-and-an-overview/ ；https://www.baseballprospectus.com/news/article/878/pitching-and-defense-how-much-control-do-hurlers-have/
- **FIP**：`FIP = (13·HR + 3·(BB+HBP) − 2·K)/IP + C`，`C = lgERA − (13·lgHR + 3·(lgBB+lgHBP) − 2·lgK)/lgIP`，使聯盟 FIP = 聯盟 ERA（逐年變動，約 3.1；2025 cFIP = 3.135）。來源：https://library.fangraphs.com/pitching/fip/ ；https://www.fangraphs.com/guts.aspx?type=cn
- **xFIP**：以 `飛球數 × 聯盟 HR/FB%`（約 9–10%，資料自 2002 起）取代實際 HR。來源：https://library.fangraphs.com/pitching/xfip/
- **SIERA**（Matt Swartz）：以回歸估計 ERA，輸入含 K%、BB%、GB%、FB% 及其交互／平方項，並做球場與得分環境調整；FanGraphs 頁面未列完整係數。來源：https://library.fangraphs.com/pitching/siera/

---

## 6. 預測系統與小聯盟轉換

- **Marcel**（Tango，2004）：(1) 近三季以 **5/4/3** 加權；(2) 加入「縮放至 **1200 PA** 的聯盟平均」做均值迴歸；(3) `預測 PA = 0.5·PA(t−1) + 0.1·PA(t−2) + 200`；(4) 年齡調整：29 歲以下 `(29 − age) × .006`，以上 `(29 − age) × .003`；不做球場調整。來源：https://www.tangotiger.net/archives/stud0346.shtml ；https://www.tangotiger.net/marcel/
- **ZiPS**（Dan Szymborski）：以近 3–4 季加權建立 baseline（24–38 歲用四年），用 **Mahalanobis 距離之聚類分析**在歷史資料庫（含 1960 年代末以來的小聯盟轉換）找相似球員群，據以產生逐人老化曲線與分配範圍；非統計輸入含年齡、守位、左右打、身高體重。來源：https://blogs.fangraphs.com/the-2020-zips-projections/ ；https://www.mlb.com/glossary/projection-systems/szymborski-projection-system
- **Steamer**（Cross、Davidson、Rosenbloom）：加權過往表現＋均值迴歸＋老化曲線，並用投球追蹤資料預測投手；確切權重未公開。來源：https://www.mlb.com/glossary/projection-systems/steamer ；https://library.fangraphs.com/principles/projections/
- **PECOTA**（Nate Silver／BP）：**最近鄰（nearest-neighbour）比較法**，將球員與約 20,000 個二戰後 MLB 打者季與約 15,000 個轉換後小聯盟季比對；相似度考量產出、使用量、體型（左右手、身高）、守位四類；老化隱含於可比球員的後續軌跡；輸出為百分位分配。此為範例導向（exemplar-based）方法。來源：https://en.wikipedia.org/wiki/PECOTA
- **MLE**（Bill James，1985 Abstract）：將小聯盟成績依聯盟、層級、球場之得分環境調整並乘以層級轉換因子，換算為「等值大聯盟成績」；是**轉換**而非預測。ZiPS 與 BP 的 Davenport Translations 皆為此類。來源：https://www.baseballhq.com/articles/scouting/minor-leagues/2013-major-league-equivalencies ；https://blogs.fangraphs.com/the-2020-zips-projections/
- **KATOH**（Chris Mitchell，2014–2016）：以一系列 **probit 迴歸**估計小聯盟打者至 28 歲前達成各 WAR 門檻之機率，再加總為期望 WAR；輸入為年齡、聯盟調整後 K%、BB%、ISO、BABIP、盜壘嘗試率（各依穩定速度先向聯盟平均迴歸）、守位、業餘背景、身高；以 2000 年前資料訓練、之後資料驗證。來源：https://sabr.org/latest/mitchell-forecasting-major-league-hitting-with-minor-league-stats ；https://blogs.fangraphs.com/a-primer-on-a-new-and-improved-katoh
- **20–80 球探量表與 Future Value**：50 = 大聯盟平均，每 10 分為一個標準差（三個標準差涵蓋 99.7%）；野手五項工具：Hit、Power、Run、Field、Arm（Throw），各有 present/future 兩個分數（如 20/50）。**FV** 為單一總評，對應前六年可控年之年均 WAR（FanGraphs 2018 對照：40 = 0.0–0.7、50 = 1.6–2.4、60 = 3.4–4.9、70 = 5.0–7.0、80 > 7.0）。來源：https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/ ；https://blogs.fangraphs.com/the-new-fangraphs-scouting-primer

---

## 7. Statcast 時代描述性指標（簡述）

- **xwOBA**：每一擊球以擊球初速與仰角（2019 起對弱擊球加入打者 Sprint Speed），依 2015 年以來「可比擊球」的結果給出 1B/2B/3B/HR 機率，再套用當季 wOBA 權重；BB、K、HBP 用實際值。來源：https://www.mlb.com/glossary/statcast/expected-woba
- **Barrel**：可比擊球（初速×仰角）歷史上至少 .500 AVG 與 1.500 SLG；初速 ≥ 98 mph 且仰角 26–30°，每多 1 mph 區間擴大，至 116 mph 時為 8–50°。來源：https://www.mlb.com/glossary/statcast/barrel
- **Sprint Speed**：最快一秒窗內之 ft/s，取「合格跑壘」（兩壘以上推進、弱擊球之本壘到一壘）中最佳約三分之二平均。來源：https://www.mlb.com/glossary/statcast/sprint-speed
- **OAA**：外野以 Catch Probability（距離、時間、方向）計算，接到 75% 機率球得 +.25、漏接 −.75，逐球累加；內野加入到球點距離、時間、與壘包距離、跑者 Sprint Speed。來源：https://www.mlb.com/glossary/statcast/outs-above-average

---

## 8. 與潛在變數（factor／IRT 型）模型的結構差異

1. wOBA、wRAA、WAR、FIP 等皆為**確定性線性權重會計式**：權重來自 RE24 矩陣的事件平均，並以「聯盟總和歸零」或「聯盟 wOBA = OBP」、「聯盟 FIP = ERA」等約束校準，屬描述性價值核算，不含參數不確定性。
2. Runs Created、BaseRuns 為非線性但仍是**團隊層級校準的會計式**；Pythagorean 為勝率經驗式。
3. 這些指標把觀測成績**直接等同於能力**，沒有「真實能力 + 測量誤差」的分離；均值迴歸僅在 Marcel 類系統以固定 PA 常數（1200）外加。
4. 預測系統屬**回歸（Steamer、KATOH 之 probit）或相似度／最近鄰（PECOTA、ZiPS）**方法，輸出為條件期望或範例分配，而非層級貝氏後驗。
5. MLE 為確定性層級乘數轉換，**不建模升降級／篩選過程**；資料中只有被選上的球員，選擇偏差不在模型內。
6. 球探 20–80 分數與 FV 是獨立於統計系統的人為量表，現行模型未將其視為對同一潛在能力的另一組測量。
7. 守備（UZR/DRS/OAA）與跑壘指標各自以不同資料來源與基準計算後再線性相加，無共同潛在因子結構。

### 對照我們的模型（`docs/model_spec.md`）
- 我們的四（五）把尺是**預先定義的構念**（球探五項工具加選球、擊球品質），每把尺由多個成績單指標當作有測量誤差的指標去釘，是 factor／IRT 型；wOBA 類指標則是把指標用固定權重直接加總。
- 我們的 WAR 是 FanGraphs fWAR（第 4.1 節），只當**結果變數**，沒有拆它的零件；守備、跑壘的零件在小聯盟沒有同等資料，所以臂力與守備是量不到的尺。
- 球探評價 ζ 是由順位、簽約金釘住的潛在變數，與 FV 的精神相同（一個總評對應未來 WAR），但在我們的模型裡它跟成績單的尺放在同一個結構方程裡。
- 升級、登上時間、WAR 三段是我們獨有的篩選過程建模；KATOH 最接近（小聯盟成績 → 期望 WAR），但它是單段 probit、不含球團層級。
- PECOTA、ZiPS 的相似球員比較法就是 `decision_style.py` 裡的 exemplar 模型（GCM）；我們是拿它來問「球隊像不像這樣決策」，不是拿它做投射。

---

## 查詢備註
Baseball-Reference Bullpen 與 Baseball Prospectus 詞彙表受 bot 防護阻擋，故 BaseRuns 改引 Tango wiki、Pythagenpat 改引 B-R「WAR Explained: Runs to Wins」頁、PECOTA 改引 Wikipedia 所轉錄之 BP 詞彙表文字、MLE 改引 BaseballHQ。Steamer 官方頁回傳 500，僅能引 MLB Glossary 與 FanGraphs Library。FanGraphs 替代水準頁寫「約 29.7% 勝率」，B-R 寫 .294；兩站皆以 1000 WAR / 2430 場為基準，但野手／投手拆分不同（FG 57/43、B-R 59/41）。

## 來源
- https://sabr.org/sabermetrics
- https://www.mlb.com/glossary/advanced-stats/runs-created
- https://sabr.org/journal/article/why-ops-works/
- https://library.fangraphs.com/principles/linear-weights/
- https://tangotiger.net/wiki_archive/Base_Runs.html
- https://www.mlb.com/glossary/advanced-stats/pythagorean-winning-percentage
- https://www.baseball-reference.com/about/war_explained_runs_to_wins.shtml
- https://library.fangraphs.com/offense/woba/
- https://library.fangraphs.com/offense/wrc/
- https://www.fangraphs.com/guts.aspx?type=cn
- https://library.fangraphs.com/war/war-position-players/
- https://library.fangraphs.com/misc/war/replacement-level/
- https://library.fangraphs.com/misc/war/positional-adjustment/
- https://library.fangraphs.com/misc/war/converting-runs-to-wins/
- https://library.fangraphs.com/defense/uzr/
- https://library.fangraphs.com/offense/ubr/
- https://www.baseball-reference.com/about/war_explained_position.shtml
- https://www.baseball-reference.com/about/war_explained.shtml
- https://www.baseball-reference.com/about/war_explained_comparison.shtml
- https://library.fangraphs.com/war/differences-fwar-rwar
- https://sabr.org/journal/article/the-many-flavors-of-dips-a-history-and-an-overview/
- https://www.baseballprospectus.com/news/article/878/pitching-and-defense-how-much-control-do-hurlers-have/
- https://library.fangraphs.com/pitching/fip/
- https://library.fangraphs.com/pitching/xfip/
- https://library.fangraphs.com/pitching/siera/
- https://www.tangotiger.net/archives/stud0346.shtml
- https://www.tangotiger.net/marcel/
- https://blogs.fangraphs.com/the-2020-zips-projections/
- https://www.mlb.com/glossary/projection-systems/szymborski-projection-system
- https://www.mlb.com/glossary/projection-systems/steamer
- https://library.fangraphs.com/principles/projections/
- https://en.wikipedia.org/wiki/PECOTA
- https://www.baseballhq.com/articles/scouting/minor-leagues/2013-major-league-equivalencies
- https://sabr.org/latest/mitchell-forecasting-major-league-hitting-with-minor-league-stats
- https://blogs.fangraphs.com/a-primer-on-a-new-and-improved-katoh
- https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/
- https://blogs.fangraphs.com/the-new-fangraphs-scouting-primer
- https://www.mlb.com/glossary/statcast/expected-woba
- https://www.mlb.com/glossary/statcast/barrel
- https://www.mlb.com/glossary/statcast/sprint-speed
- https://www.mlb.com/glossary/statcast/outs-above-average
