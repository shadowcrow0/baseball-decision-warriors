# 小球 vs. 強打：在 WAR 的帳本裡各值多少？

回答的問題：「台灣追求短打、點，美國傾向打強打遠；這兩種打擊型態在 WAR 裡有什麼差異？」以 FanGraphs（fWAR）與 Baseball-Reference（bWAR）的公開定義為準，所有數字附來源；查不到的地方直接註明。查詢日期 2026-10-22。

## 1. WAR 的打擊帳本怎麼記小球與長打

### 1.1 基本公式

FanGraphs 的打擊價值來自 wOBA → wRAA（Batting Runs）：

```
wOBA = (wBB·uBB + wHBP·HBP + w1B·1B + w2B·2B + w3B·3B + wHR·HR)
       / (AB + BB − IBB + SF + HBP)
wRAA = (wOBA − lgwOBA) / wOBAscale × PA
```

2024 年 FanGraphs 權重：wBB .689、wHBP .720、w1B .882、w2B 1.254、w3B 1.590、wHR 2.050，wOBAscale 1.242，R/W 9.683（[FanGraphs Guts]）。

### 1.2 犧牲觸擊（SH）：零，不是負

- **FanGraphs**：分母是 AB + BB − IBB + SF + HBP，SH 不在分子也不在分母（[FanGraphs wOBA]）。一次成功犧短對 wOBA 的影響是零；觸擊失敗被封殺通常記為 AB 出局，才會是負值。
- **Baseball-Reference**：明文寫「SHs and IBBs are managerial decisions」，SH 不進 wOBA（rOBA），但保留在 wRAA 的 PA 基數 AB+BB+HBP+SF+SH 裡，假設打者在該打席「表現如常」（[B-R wRAA explained]）。兩家都把犧短當成「沒發生的打席」。
- **但依 RE24，平均的犧短是賠錢的**。Tango 2010–2015 run expectancy：一壘有人無人出局 0.859 → 二壘有人一出局 0.664，**−0.195 分**；二壘無人出局 1.100 → 三壘一出局 0.950，−0.150；一二壘無人出局 1.437 → 二三壘一出局 1.376，−0.061（[Tango RE24]）。Wikipedia 引 1993–2010 數字 .941 → .721，約 −23%（[Wikipedia Sacrifice bunt]）。
- **即使只求一分也不划算**：Dan Fox 用 1999–2002 資料，一壘無人出局得至少一分機率 .437，成功犧短後（二壘一出局）.406，位置球員的損益兩平成功率超過 100%；只有投手打擊時（損益兩平約 .679，實際成功率 .780）、一二壘或二壘無人出局且只需一分的情境才划算（[Fox, THT]）。James Click 的結論：投手幾乎都該點；位置球員只在二壘無人出局、晚局一分定勝負時點；其他情況皆有害（[Wikipedia Sacrifice bunt]）。
- 台灣資料也一致：高志綱（2014）用 2012–13 中職例行賽估計，一壘無人出局 → 成功犧短後，2013 年不得分機率 0.661 → 0.620，但得 2 分以上機率 0.087 → 0.029（[NTUS 論文]）：換一分、放棄大局，跟 MLB 的結論相同。

**小結**：WAR 給犧短的分數是 0，而 RE24 顯示它平均其實約 −0.1～−0.2 分。WAR 對小球戰術「不罰」，但也絕不「加分」。

### 1.3 觸擊安打、內野安打：就是一壘安打

觸擊安打進 1B，拿滿一壘安打的權重（.882）。B-R 自 2003 年起另把內野安打與外野安打分開給值，並把失誤上壘（ROE）當成「一壘安打 + 0.018」的可重複技能（[B-R wRAA explained]）。這是小球在 WAR 裡唯一能「賺」的打擊事件。

### 1.4 盜壘：wSB 與 UBR

```
wSB = SB·runSB + CS·runCS − lgwSB·(1B + BB + HBP − IBB)
runSB = +0.200（固定）；runCS = −(2·RunsPerOut + 0.075)
```

2024 runCS = −.405，2025 = −.410（[FanGraphs Guts]；[FanGraphs wSB]）。損益兩平 = .405/(.200+.405) ≈ **67%**（FanGraphs 的口語版：「SB 至少要是 CS 的兩倍」）。Ben Clemens 用 2024 Statcast 實測：成功盜壘 +0.169、被抓 −0.394，損益兩平約 70–71%（[Clemens, FanGraphs]）；依壘包出局數不同在 68%–90% 之間（[Mathemadage]）。UBR 另計推進額外壘包、補位、觸殺等，排除 SB/CS；BsR = wSB + UBR + wGDP（[FanGraphs UBR]）。

### 1.5 「有效出局」、打帶跑、接觸型打者

- wOBA 不區分出局種類。B-R 有區分：現代球季三振比其他出局**略好**（2025 AL 調整 SO +0.009、非 SO 出局 −0.005），因為滾地出局會帶來雙殺；B-R 另以 Rdp 給雙殺避免每次 .44 分（[B-R wRAA]；[B-R position WAR]）。Retrosheet（Ruane）舊表則 K −0.238、其他出局 −0.200、GIDP −0.688（[Retrosheet]）。
- 結論：三振與其他出局差距約 0.01–0.04 分，相對 1B 與 HR 的差距（約 1 分）可忽略。「推進跑者的滾地出局」在 wOBA 裡仍是出局；跑者的推進若超過平均，才由 UBR 記給**跑者**而非打者。

### 1.6 長打的線性權重

FanGraphs 2015 相對出局的線性權重：BB .55、1B .70、2B 1.00、3B 1.27、HR 1.65；一次出局約 −0.26（[FanGraphs Linear Weights]）。2002–04 的經典表：1B .465、2B .772、HR 1.394、K −.287、其他出局 −.250、雙殺 −.839（[True Blue LA]）。

**換算：一支 HR ≈ 2.3–2.4 支一壘安打**（2.050/.882 或 1.65/.70），而 **一支 HR ≈ 8–10 次成功犧短的 RE 損失**（1.65/0.195 ≈ 8.5）。

### 1.7 範例：OBP 相同、型態不同

600 PA、540 AB、50 BB、5 HBP、5 SF、149 H，OBP 皆 .340，用 2024 FanGraphs 權重：

| | 打者 A（接觸/單打型） | 打者 B（長打型） |
|---|---|---|
| 1B / 2B / 3B / HR | 125 / 20 / 2 / 2 | 85 / 30 / 2 / 32 |
| AVG / OBP / SLG | .276 / .340 / .331 | .276 / .340 / .517 |
| wOBA 分子 | 180.7 | 219.4 |
| wOBA | .301 | .366 |
| wRAA 差（÷1.242×600） | — | **+31.2 分** |
| ≈ WAR 差（÷9.683） | — | **≈ +3.2 WAR** |

若打者 A 再加 20 次成功犧短：wOBA 不變、wRAA 不變（FanGraphs 連分母都不計）；在 B-R 只是 PA 基數多 20。若 A 多 20 次盜壘、成功率 75%（15 SB / 5 CS）：15×.200 − 5×.405 = **+0.98 分**，約 0.1 WAR。這就是「小球最好只能打平、長打才產生 Batting Runs」的量化版本。

## 2. 聯盟風格差異的數字

### 2.1 MLB：犧短 60 年內跌掉八成、HR 翻倍

Baseball-Reference 每隊每場（[B-R League Batting]）：

| 年 | SH/G | HR/G | SB/G |
|---|---|---|---|
| 1960 | 0.48 | 0.86 | 0.37 |
| 1980 | 0.45 | 0.73 | 0.78 |
| 2000 | 0.34 | 1.17 | 0.60 |
| 2010 | 0.32 | 0.95 | 0.61 |
| 2019 | 0.16 | 1.39 | 0.47 |
| 2024 | 0.09 | 1.12 | 0.74 |
| 2025 | 0.12 | 1.16 | 0.71 |

1973 年美聯 DH 是第一次大跌，2022 國聯全面 DH 後再跌（[B-R Bullpen SH]）。

### 2.2 NPB：犧短是 MLB 的 8 倍以上

2024 年 12 隊各 143 場，球團打擊成績（[baseball-data.com]）：中央聯盟 HR 472、犠打 691；太平洋聯盟 HR 503、犠打 637。全聯盟 1,716 隊場次 → **HR 0.57/隊場、SH 0.77/隊場**（自行加總）。沒有 DH 的中央聯盟犧短較多（691 vs 637）。2017 年 Ringer/SABR 報導的 NPB 0.80 vs MLB 0.19 幾乎一致（[SABR/Lindbergh]）。

### 2.3 KBO

2024 年例行賽共 1,438 轟（2023 年 924）（[isplus]）；10 隊 × 144 場 = 1,440 隊場次（[B-R Bullpen 2024 KBO]）→ 約 **1.00 HR/隊場**。犧短 2024 總數未找到；2019 年 Pitcher List 的比較：KBO 犧打 0.31/9 局 vs MLB 0.16，HR 0.71 vs 1.40（[Pitcher List]）。

### 2.4 CPBL：聯盟層級總數**未能取得**

中職官網統計頁與台灣棒球維基館對自動擷取回傳 404/拒絕，cpblstats.com 於撰寫時 503，故**無法提供中職全聯盟犧短與全壘打的每場數字**，要直接查 cpbl.com.tw 補上。可用的片段：

- cpblstats（引用 Rebas 資料）：2024 年 4–5 月中信兄弟 33 場 33 次犧短，有望超越 2021 年單隊 106 次的近六年最高紀錄（[cpblstats bunting]）；106 次 / 120 場 ≈ 0.88/場，是 2024 MLB 全聯盟平均的近 10 倍。
- 2010–2015 年：有觸擊機會時，統一獅執行率 <8%，中信兄弟達 16%（[SETN/DailyView]）。該文宣稱「點的半局得分率較高」，但這是選樣偏誤（教練只在有利情境下點），非因果。
- 2024 年中職全壘打王 Moya 30 轟（[B-R Bullpen 2024 CPBL]）；聯盟 wOBA .330、3,015 分 / 27,419 PA（[HackMD]）。
- 用球：中職 2020 年把恢復係數區間收窄到 0.540–0.560（[NOWnews]）。

## 3. 台灣／亞洲有人算 CPBL 的 WAR 嗎？

- **中職官方**：供認證媒體用的資料庫含 wRC+ 與投手 WAR（[OCAC/Focus Taiwan]）；2026 年上線的 stats.cpbl.com.tw 目前只見擊球品質、揮空等排行，無 WAR 說明（[stats.cpbl.com.tw]）。
- **Rebas 野球革命**：2021 年起自建中職進階資料，2024 年 3 月宣布開發 wRC+、WAR 與守備指標（[CNA]）；方法文件未公開於可擷取頁面。
- **社群推算（HackMD, Wayway Pao）**：完全照 FanGraphs 公式，權重取自 PTT ldkrsi；2024 年 u1B .9528、uHR 2.1314，wOBAscale 1.2282，替代水準 0.029 R/PA，RPW 9.4；跑壘只算 wSB，UBR/wGDP 與守備因無資料設 0，Park Factor 設 1；**未提及犧短**，依公式即與 FanGraphs 相同：不計（[HackMD]）。
- cpblstats.com 2018 年曾刊投手 WAR（Roenicke 3.58）但無方法說明（[cpblstats 2018]）。

結論：目前所有可見的中職 WAR 都沿用 FanGraphs 框架，小球事件的處理與 MLB 相同。

## 4. 接觸型打者被 WAR 低估嗎？亞洲打者赴美的長打折損

- **Ichiro**：生涯 wOBA .328、wRC+ 104、57.8 WAR；Bobby Abreu .378 / 129 / 59.8。兩人 WAR 相當，是因為 Ichiro 守備遠勝（[Laurila, FanGraphs]）。Ichiro 2,514 支一壘安打史上第 6（[CBS]）：單打型打者的高 WAR 來自守備、跑壘與超長生涯，不是 wOBA 高估單打。
- **Arráez**：2024 年打擊王，但 1.1 rWAR、WAA −1.0，是 1871 年以來打擊王最低（[Fish On First]）。WAR 對「只會打單打」的評價就是如此，不是低估，而是定義。
- **NPB → MLB**：9 位 ≥500 AB 的日本打者，在 MLB 的 HR 率約為日職的 51%，SLG −90 點、OBP −25～30 點（[Bleacher Report]）；Ichiro、松井為保住長打的例外。
- **KBO → MLB**：Davenport 翻譯：姜正浩 2014 年 39 轟 → 24、朴炳鎬 53 → 33（[Davenport]）。
- 含意：亞洲打者赴美最先掉的是長打，而長打恰是 WAR 的主要來源；接觸能力保留較好，但接觸本身在 wOBA 只值一壘安打。

## 5. 對照本隊發現

本隊小聯盟模型的結果（`docs/factor_plan.md` §15）：犧短率對升級係數 17/17 季為負（合併 −0.10/SD）、MLB 初登板更慢（12/13 季）、WAR 係數略負（−0.07，11/13 季為負）；盜壘成功 +0.09 升級、+0.05 WAR；長打能力 +0.30、+0.14；接觸能力 +0.65、+0.26。

**給台灣讀者的三句話**：WAR 的帳本本身對犧短是零、對長打是每轟約 2.3 支單打的價值，所以模型裡「犧短負、長打正」與 WAR 的定義方向一致，而犧短的負係數更可能反映「被要求點的人是誰」（弱棒、角色球員）而非觸擊本身害人；接觸能力的係數最大，說明美式發展體系仍高度看重擊球能力，只是它要靠長打或守備、跑壘才能變成 WAR；因此「方法可以搬到台灣」，但若用中職資料重估，需先確認中職 WAR 對 SH/UBR/守備的處理（目前多為 0 或缺資料），否則小球元素的係數會被資料可得性而非戰術價值所決定。

## 查詢備註
中職全聯盟每場犧短與全壘打數未能取得（cpbl.com.tw 統計頁對自動擷取回 404、台灣棒球維基館擋 bot、cpblstats.com 503），投影片定稿前要人工從 cpbl.com.tw 補兩個數字。NPB 2024 每場數字是從 baseball-data.com 的 12 隊球團列自行加總（npb.jp 回 503）。True Blue LA 的線性權重表與 cpblstats 的兄弟 33 次／2021 年 106 次兩個數字來自搜尋摘要，頁面本身無法開啟，來源列已標註。

## 來源

- [FanGraphs Guts] https://www.fangraphs.com/tools/guts
- [FanGraphs wOBA] https://library.fangraphs.com/offense/woba/
- [FanGraphs wSB] https://library.fangraphs.com/offense/wsb/
- [FanGraphs UBR] https://library.fangraphs.com/offense/ubr/
- [FanGraphs Linear Weights] https://library.fangraphs.com/principles/linear-weights/
- [Clemens, FanGraphs] https://blogs.fangraphs.com/breaking-even-is-for-suckers/
- [Mathemadage] https://blogs.fangraphs.com/mathemadage-outs-at-third-base/
- [B-R wRAA explained] https://www.baseball-reference.com/about/war_explained_wraa.shtml
- [B-R position WAR] https://www.baseball-reference.com/about/war_explained_position.shtml
- [B-R League Batting] https://www.baseball-reference.com/leagues/majors/bat.shtml
- [B-R Bullpen SH] https://www.baseball-reference.com/bullpen/Sacrifice_hit
- [B-R Bullpen 2024 CPBL] https://www.baseball-reference.com/bullpen/2024_CPBL
- [B-R Bullpen 2024 KBO] https://www.baseball-reference.com/bullpen/2024_KBO
- [Tango RE24] https://www.tangotiger.net/re24.html
- [Fox, THT] https://tht.fangraphs.com/not-so-sweet-surrender/
- [Wikipedia Sacrifice bunt] https://en.wikipedia.org/wiki/Sacrifice_bunt
- [True Blue LA]（擷取時 403，數字取自搜尋摘要）https://www.truebluela.com/2007/4/14/141251/820
- [Retrosheet] https://retrosheet.org/Research/RuaneT/valueadd_art.htm
- [baseball-data.com] https://baseball-data.com/24/team/hitter.html
- [SABR/Lindbergh] https://sabr.org/latest/lindbergh-yusuke-okada-the-bill-james-of-japanese-baseball/
- [Pitcher List] https://pitcherlist.com/kbo-vs-mlb-a-breakdown/
- [isplus] https://isplus.com/article/view/isp202411210047
- [NTUS 論文] https://ir.ntus.edu.tw/entities/publication/a1a41906-1eed-47aa-ac03-2079ebc54bca/full
- [SETN/DailyView] https://www.setn.com/news/105054
- [cpblstats bunting]（擷取時 503，數字取自搜尋摘要）https://cpblstats.com/bunting-effectiveness-cpbl/
- [cpblstats 2018] https://cpblstats.com/top-pitchers-from-the-2018-cpbl-season/
- [NOWnews] https://www.nownews.com/news/5768540
- [HackMD] https://hackmd.io/t34XIJFnSIK2xcsGohZe5Q
- [CNA] https://www.cna.com.tw/news/aspt/202403230100.aspx
- [OCAC/Focus Taiwan] https://ocacnews.net/article/365238
- [stats.cpbl.com.tw] https://stats.cpbl.com.tw/
- [Laurila, FanGraphs] https://blogs.fangraphs.com/sunday-notes-did-bobby-abreu-have-a-better-mlb-career-than-ichiro-suzuki/
- [CBS] https://www.cbssports.com/mlb/news/leaderboarding-breaking-down-the-hall-of-fame-career-of-ichiro-suzuki-by-the-numbers
- [Fish On First] https://fishonfirst.com/news-rumors/major-league-baseball-mlb/luis-arraez-batting-title-stats-value-2024/
- [Bleacher Report] https://bleacherreport.com/articles/315874-japanese-hitters-in-the-us
- [Davenport] https://claydavenport.com/?p=333
