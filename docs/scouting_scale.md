# 球探評分體系（20–80 量表）與統計模型的關係：給 ζ 的定位

用途：說明「球探分數在主流做法裡是另一套系統」是什麼意思，以及我們模型裡的球探評價 ζ（由選秀順位、簽約金釘住的潛在變數）跟它差在哪。所有說法附來源；查詢日期 2026-10-22。

## 1. 20–80 球探量表是什麼

**起源。** FanGraphs 與 Baseball Prospectus 均稱此量表「一般歸功於 Branch Rickey」，但 BP 也承認「實際起源有點像個謎」；FiveThirtyEight 引 Kerrane《Dollar Sign on the Muscle》指出 Rickey 本人用的是 0–60（30 為平均），而 Al Campanis 另有 60–80 量表，誰創出 2–8/20–80 版本至今不清楚。部分球團仍用 2–8 制（無半級）。

**統計假設。** FanGraphs：「50 是大聯盟平均，每 10 分代表一個標準差」；20–80 而非 0–100 是因為 ±3 SD 已涵蓋常態分布約 99.7%，但 McDaniel 也說「並非每項工具都呈常態，多數只是接近」。Baseball America 的說法更保守：「量表鬆散地建立在統計分布上」，20 與 80 是「實務上的極端值」。FanGraphs 2018 年入門文另指出：球探常用 45/55 增加平均附近的解析度，卻很少用 35/65，因為肉眼在尾端沒有那種精度；分析師用 TrackMan 等客觀數據時才較常使用。

**誰在打分。** 球團內部由區域球探（area scout）寫報告並排序，區域/全國 cross-checker 複查並整合各區，球探總監（scouting director）定前段順位、GM 多只涉入前 15–20 順位與簽約預算（Blengino, FanGraphs 2014）。公開媒體方面，FanGraphs（Longenhagen/McDaniel）、Baseball America、MLB Pipeline、Baseball Prospectus 各自發布評分，MLB.com 稱其評分「大致符合球團的打分方式」。

**現在/未來分級與五項工具。** 野手五工具為 Hit、Power、Run、Field、Arm；每項有 present 與 future 兩個分級，記為「20/50」。高中球員 present 常在 20 左右，高階小聯盟球員兩者差距很小。Power 又分 raw power（打擊練習最深的一球）與 game power（成熟期每季全壘打數）；BA 則直接以「大聯盟預期全壘打數」定義 game power，以打擊率定義 Hit。McDaniel 稱 Hit tool「大約有十幾個輸入」，是最難評的一項。

**整體分級（FV/OFP）不是公式。** FanGraphs 明言 OFP/FV「不是各工具分級的簡單平均」，而是要反映球員在市場上的價值，納入風險與距離天花板的遠近；BP 詞彙表亦稱 OFP「是球探給的單一分數，不是各項分數的平均，權重因人而異」。2018 年 FanGraphs 入門文說評價來源是「自己看、與球探/高層的對話、公開數據的某種組合」，而「因傷病或人口特徵扣多少分是主觀的」。MLB Pipeline 稱 future overall grade 是「涵蓋一切的單一數字」，65 以上代表可能的 All-Star 級。

## 2. FV 校準到什麼

**FanGraphs。** FV 定義為「對應球員前六個服役年度平均年 WAR 的 20–80 分級」。2018 年入門文給出野手對照表（50 = 1.6–2.4 WAR「平均每日先發」；55 = 2.5–3.3；60 = 3.4–4.9「All Star」；70 = 5.0–7.0；80 > 7.0；45 = 0.8–1.5；40 = 0.0–0.7）。關鍵在於：這張表是用 **2018 年現役大聯盟球員 WAR 分布**（200 PA 以上 355 人取前 255 人）映射出來的，不是從新秀結果反推。新秀「結果機率」另外來自 Craig Edwards 的研究（An Update to Prospect Valuation，2018-11-12）：以 BA 1996–2010 年 Top 100 為基礎、依 Zimmerman 的方法回推歷史分級，求出各 FV 層的平均 WAR、bust 率與 star 率（例如 60 FV 野手 154 人平均 6.1 WAR、剩餘價值約 $55M；70 FV 野手 21 人平均 12.5 WAR、$112M；50 FV 野手 433 人平均 3.1 WAR、$28M；採 $9M/WAR、8% 折現）。2019 年 Top 100 文中便引用此基準，稱「平均 60 FV 野手有 26% 機率在六個控制年成為常年 5+ WAR 球員」，再依個案（如 Rookie 級 vs 3A）手動調整。2026 年 Ben Clemens 重做方法論：用 BA 2005–2016 加 FanGraphs 2017–2018 名單，把序位轉成 FV 使分布與現行打分一致，並承認「依球員最後結果回推分級，本質上把未來資訊帶進來了」。

**Baseball America。** 給一個 role grade（20–80，如 50 = Solid-average regular、60 = Occasional All-Star、65–70 = Perennial All-Star）加一個風險等級（Low 0、Mild −5、Average −10、High −15、Extreme −20），相減得 adjusted grade 用於排序；「60/High」即 60 扣 15 得 45。

**MLB Pipeline** 同用 20–80，以 future overall grade 為核心；**Baseball Prospectus** 以 OFP 加 Likely 兩個分級並附 Role/Risks 文字，Goldstein 稱這迫使評估者回答「你相信這球員的天花板版本是什麼」，不以百分比桶表示以免「假精確」。

## 3. 與統計模型的關係

**(a) 獨立的序數人為量表。** 上述所有分級都是人為判斷，並非從 box-score 推導；FanGraphs 的 The Board 甚至分「Scouting + Stats」與「Scouting Only」兩種檢視，並區分 present hit grade 在 Hi-A/AA 前後的意義。

**(b) 主流統計系統不以球探分級為輸入。** Marcel 作者 Tango 自述「用三年大聯盟數據、近期加權、回歸均值、年齡因子」，是「盡量少用智慧的最基本系統」。ZiPS 作者 Szymborski 列出輸入為多年統計、1960 年代末以來的小聯盟轉換、速球/Statcast、傷病、年齡位置等，並強調「ZiPS 純粹以數據為基礎」，未提球探分級。Steamer 的公開描述聚焦於幾乎所有小聯盟球員的 MLE 轉換。Davenport Translations 與 James 的 MLE 亦純由聯盟/球場難度調整統計而來。FanGraphs 的 projections 總覽頁把這些系統描述為「用過去數據、年齡與其他因素」，未提球探報告。據所查資料，未見任何主流系統以 20–80 分級為輸入。

**(c) 混合做法。**
- **KATOH / KATOH+**（Chris Mitchell）：純統計版「只考慮小聯盟數據，不考慮新秀排名」；KATOH+ 則「納入 BA 季中 Top 100 以及 Longenhagen 對未列名球員的季前 FV」。2018 年因 FV 未更新而暫停 KATOH+，顯示其對球探輸入的時效依賴。
- **Zimmerman（FanGraphs 2017）**：用 MLB.com 的 Power/Speed/Hit 未來分級加五項可觀測比率建投射，調整聯盟後仍輸給 Steamer，且發現「Hit tool 對預測 BABIP 無用」。
- **Edwards（2018）**、**Creagh & DiMiceli（Point of Pittsburgh, 2016）**：後者以 BA 1994–2006 Top 100 的序位分層，算六個控制年的 fWAR、$8M/WAR、8% 折現（野手 #1–10 平均 15.3 WAR、$73.5M；#76–100 平均 4.5 WAR、$20.6M）。兩者都是「名單 → 價值」轉換，不是把分級餵進統計投射。
- **Victor Wang（SABR 2007）**、**Arthur（BP 2015）**：以名單序位預測 WARP；Arthur 用 SVM 得 R² = .18，並稱「先前研究顯示優秀的人為評估者能勝過以小聯盟數據為基礎的投射」。
- **學術文獻**：Streib, Young & Sokol（Interfaces 2012）為某球團建立「球探意見經常顯著分歧」下的共識排序模型；Spurr（JSE 2000）發現長期而言各球團找出大聯盟球員的能力無顯著差異、市場曾長期低估大學經驗；Burger & Walters（JSE 2009）估計首輪中位數選秀年化報酬 44%，高中 36% vs 大學 57%，投手 34% vs 野手 52%，「與市場效率不一致」。

## 4. 選秀順位與簽約金作為球探共識的代理

**支持證據。** 球團的選秀板是由上述球探層級逐層匯整而成（Blengino），因此順位是 30 隊內部評估經競爭後的「揭示選擇」。順位價值曲線高度單調：Edwards（An Update on How to Value Draft Picks，2019，1993–2007 選秀）估第 1 順位現值 $45.5M、第 10 順位 $23.3M、第 30 順位 $10.1M、第 70 順位 $3.8M，第 11 輪後每人約 $1.0M 以下；Andrecheck 2009 年模型被 FanGraphs 稱為選秀產出的「黃金標準」。BA 甚至直接用「自家排名對應之 slot value」與實際簽約金比較（2024 年 Condon 排名第 1、slot $10,570,600，簽 $9,250,000）。

**雜訊來源。**
1. **Slot/bonus pool 制度**：2012 年起前 10 輪每個順位有指定金額，球團總池 = 各順位之和；超過 0–5% 課 75% 稅，超過 5% 即失去未來首輪籤，至今無隊超過 5%；第 11 輪後每人上限 $150,000；未簽約則扣除該 slot。
2. **池內挪移**：球隊常在前段簽低於 slot 以挪給後段（如 2012 太空人 Correa 低於 slot $2.4M），故簽約金 ≠ 順位 ≠ 評價。
3. **簽約意願（signability）**：區域球探報告本就記錄 signability、大學承諾；球團另設「unsignable」「overpay」板；高中球員握有升學籌碼，Burger & Walters 亦顯示高中/大學報酬差異。
4. **球團私有資訊**：BA 指出「每隊對球員都有比公開更多的資訊」，且人才並非依 slot 金額的簡單遞減分布。
5. **醫療與策略**：醫療分級、組合式風險分配（Blengino）都會改變順位，與工具分級無關。

## 5. 結語：平行體系 vs. 潛在變數 ζ

球探分級在主流實務中是一套**平行的、人為產生的序數量表**：FV 以 WAR 區間校準、以歷史名單結果估基準率，但不進入 Marcel/ZiPS/Steamer 等投射的結構方程；少數混合（KATOH+、Zimmerman）是事後把名單或分級當附加迴歸項。我們的 ζ 則不同：它是一個由**選秀順位與簽約金兩個指標**量測的潛在因子，與四（五）個統計能力同在一條結構方程中聯合估計。差異：
- **本質**：pick/bonus 是**市場在選秀日揭示的一次性估值**（含 slot 規則、簽約意願、池內挪移與球團私有資訊），不是球探逐項工具的 20–80 分級。
- **時間**：選秀日固定，不再更新；FV/BA 分級每年甚至季中更新。選秀之後的所有變化，在我們的模型裡由成績單的尺承擔。
- **粒度**：一個數字（兩個指標）vs. 五項工具 + present/future + 風險等級。我們看不到臂力與守備那兩項，這是量不到的尺（`docs/model_spec.md`）。
- **產生者**：30 隊競爭下的揭示選擇 vs. 個別評估者（球探、cross-checker、媒體分析師）的判斷。
- **可識別性**：ζ 是有兩個指標的潛在變數，尺度靠測量模型與統計能力的共變識別，沒有「50 = 平均」的絕對意義；球探分級是可直接觀測的序數量表。
- **用法**：他們分開算、在腦袋裡合；我們放在同一條方程式，才能問「成績單與球探評價各算幾分、哪個跟著球員比較久」（答案見 `docs/factor_plan.md` §15：球探評價對升級 +0.50、對 WAR +0.21，17 季全同號、無趨勢）。

## 來源
- https://blogs.fangraphs.com/?p=161314 （Scouting Explained: The 20-80 Scouting Scale）
- https://blogs.fangraphs.com/scouting-the-hit-tool-pt-6
- https://blogs.fangraphs.com/the-new-fangraphs-scouting-primer
- https://blogs.fangraphs.com/how-to-use-the-board-a-tutorial-2022/
- https://blogs.fangraphs.com/2019-top-100-prospects/
- https://blogs.fangraphs.com/?p=299295 （Edwards, An Update to Prospect Valuation, 2018-11-12）
- https://blogs.fangraphs.com/the-details-of-our-new-prospect-valuation-methodology/
- https://blogs.fangraphs.com/an-update-on-how-to-value-draft-picks
- https://blogs.fangraphs.com/what-happens-in-the-draft-room
- https://blogs.fangraphs.com/katohs-midseason-2017-top-100-lists
- https://blogs.fangraphs.com/2018-top-100-katoh-prospects
- https://fantasy.fangraphs.com/turning-scoutable-traits-into-projections
- https://blogs.fangraphs.com/the-2020-zips-projections/
- https://library.fangraphs.com/principles/projections/
- http://www.tangotiger.net/marcel/
- https://blogs.fangraphs.com/category/steamer-projects/
- https://legacy.baseballprospectus.com/other/bp1996/dtessay.html
- https://baseballamerica.com/?p=42636 （Explaining the 20-80 Scouting Scale）
- https://carloscollazo.substack.com/p/an-update-to-ba-grades
- https://baseballamerica.com/?p=1466920 （Best & Worst Values of the 2024 Draft）
- https://www.mlb.com/glossary/scouting/scouting-grades
- https://www.mlb.com/news/mlb-draft-2025-bonus-pick-values
- https://www.cbssports.com/mlb/news/2024-mlb-draft-slotting-system-explained-how-format-bonus-pools-work-and-who-has-the-most-money-to-spend
- https://www.bostonglobe.com/sports/2015/06/10/how-slot-values-and-bonus-pools-affect-mlb-draft-picks/fl0qcINq44GDFTShdFenRP/story.html
- https://www.bleachernation.com/?p=17245 （2012 首年 bonus pool 報導）
- https://legacy.baseballprospectus.com/glossary/index.php?search=OFP
- https://www.baseballprospectus.com/prospects/article/5870/future-shock-organizational-rankings/
- https://www.baseballprospectus.com/news/article/38784/deep-playable-trust-buckets-prospect-evaluation-frameworks/
- https://www.baseballprospectus.com/prospects/article/25764/moonshot-another-look-at-prospect-values/
- https://baseballprospectus.com/?p=38198 （Draft Value Calculator）
- https://fivethirtyeight.com/features/baseball-scouts-use-numbers-too/
- https://thepointofpittsburgh.com/mlb-prospect-surplus-values-2016-updated-edition/
- https://thepointofpittsburgh.com/ranking-farm-systems-by-prospect-surplus-value-method
- https://sabr.org/journal/article/how-much-is-a-top-prospect-worth/
- https://tht.fangraphs.com/how-much-is-a-draft-pick-worth-in-2014/
- https://ideas.repec.org/a/inm/orinte/v42y2012i2p119-130.html （Streib, Young & Sokol 2012）
- https://ideas.repec.org/a/sae/jospec/v1y2000i1p66-85.html （Spurr 2000）
- https://ideas.repec.org/a/sae/jospec/v10y2009i5p485-501.html （Burger & Walters 2009）

未能取得全文而未引用：Harvard Sports Analysis 2018〈How Predictive is the 20-80 Scale〉、FiveThirtyEight〈What to Expect from BA's Top 100〉、Journal of Sports Analytics〈Major League Draft WARs〉；Andrecheck 2009 原文僅經 FanGraphs/THT 轉述。
