# 進階指標對未來價值的預測力：文獻盤點與對五把尺的映射（2026-10-23）

問題：在個人層級，哪些「處理過」的指標（Statcast、選球、擊球、擊棒追蹤、跑壘、守備）比成績單上的黑白結果更能預測隔季 wOBA／WAR？以下只做文獻與證據整理，不跑計算；**文獻說的**與**我的推論**分開標示；查不到原始數字的標「未經證實」。

## 1. 證據表

| 指標 | 想取代的成績單指標 | 年際穩定度／穩定樣本 | 對隔季結果的預測效度 | 來源 | 狀態 |
|---|---|---|---|---|---|
| K%、BB%、ISO、HR/PA、BABIP、XBH、LD% | （基準） | KR-21 達 .70：K% 60 PA、BB% 120 PA、ISO 160 AB、HR 170 PA、GB/FB 80 BIP、LD 600 BIP、BABIP 820 BIP、XBH 1610 PA（估） | — | [Carleton, BP 2012](https://www.baseballprospectus.com/news/article/17659/baseball-therapy-its-a-small-sample-size-after-all/)；[FG Library](https://library.fangraphs.com/principles/sample-size/) | 已核 |
| xwOBA、xwOBAcon vs wOBA | wOBA | xwOBAcon 自相關高於 wOBAcon（MLB 技術部落格 Sharpe，數字未取到） | 2020–23、≥100 BBE：wOBAcon→隔季 wOBAcon r≈.50，xwOBA >.55，Tango 的 8 格 pwOBA .61；三者合併不再提升。FG 社群研究結論「不清楚誰較好」；BP 投手版 .35 vs .32（誤差內） | [Tango](http://tangotiger.com/index.php/site/comments/introducing-predictive-woba)；[FG Community](https://community.fangraphs.com/exploring-batter-xwoba-and-its-applications-part-2/)；[BP](https://www.baseballprospectus.com/news/article/40026/prospectus-feature-siren-song-statcasts-expected-metrics/) | 增益小但方向一致 |
| Barrel% | HR/AB、ISO | 年際 r .79（Brl/BBE，2016–19，≥300 PA）；迴歸量約 51–60 BBE | →隔季 wOBAcon .54–.58（MLB）、.55（2023→24）；同季對 ISO r .80 | [FG 2019](https://fantasy.fangraphs.com/which-statcast-measures-correlate-best-2019-refresh/)；[FG 3A](https://fantasy.fangraphs.com/a-quick-look-into-minor-league-statcast-for-hitters/)；[Salorio](https://adamsalorio.substack.com/p/evaluating-bat-speeds-predictive) | 已核 |
| HardHit%（≥95） | 同上 | 年際 .73；10.9% 的人季變動 ≥1 SD（三者中最吵） | →隔季 wOBAcon .49；被評為「三種 EV 整理法中最弱」 | [Andrews, FG](https://blogs.fangraphs.com/the-doomed-search-for-a-perfect-way-to-interpret-exit-velocity-data/) | 已核 |
| EV90／best speed／maxEV | 同上 | EV90 年際 Spearman .85 vs 平均 EV .77（2023–24）；maxEV .81；EV90 迴歸量 26 BBE | 最佳百分位隨樣本變：≥100 BIP 用 EV81（EV70–90 皆可），30–110 BIP 用 EV96；峰值對隔季 wOBAcon r≈.54、ISO .74。平均 EV 被 Tango 稱「最不可靠」 | [BP](https://www.baseballprospectus.com/news/article/95454/prospectus-feature-exit-velocity-modeling-average-maximum-athletic-effort/)；Andrews 同上 | 已核 |
| Sweet-spot%（LA 8–32°） | LD%、IFFB% | 年際 .40（925 人，2024）vs .59（另一分析）——互相矛盾 | 對隔季 BABIP r² 僅 .03 | [mlbdatawarehouse](https://www.mlbdatawarehouse.com/p/how-sticky-is-sweet-spot-rate)；[Pitcher List](https://pitcherlist.com/understanding-babip-pt-3-hitters/) | 非正式來源，**弱** |
| Pull-air%（拉打飛球率） | Pull%、HR/FB | 95–105 mph 飛球拉打率年際 R² .32（r≈.56）；多年 R² .62；約 340 BBE 穩定 | 無直接「→隔季 HR」檢定；拉打飛球 wOBA .812 vs 反方向 .289 | [Clemens, FG](https://blogs.fangraphs.com/an-meandering-examination-of-fly-ball-pull-rate-featuring-stars-of-the-game-and-also-isaac-paredes/)；[Rotowire](https://www.rotowire.com/baseball/article/leaderboard-of-the-week-pulled-air-rate-98296) | 預測 HR 屬推論 |
| 擊棒速度（2024 起） | 無對應 | 迴歸量 2–3 次揮棒（Tango），極穩 | 年 1 擊棒速度→年 2 wOBA r .224；xwOBA .446；合併 .455。→隔季 wOBAcon .48；同季對 wRC+ r 僅 .11 | [FG Fantasy](https://fantasy.fangraphs.com/a-way-too-early-look-at-the-importance-of-bat-speed/)；[Clemens](https://blogs.fangraphs.com/what-statcasts-new-bat-tracking-data-does-and-doesnt-tell-us/) | 已核 |
| Squared-up%、Blast% | 無對應 | 年際相關**未見發表** | 同季：squared-up% 對 K% r −.67、對 wOBAcon ≈0；blast% 對 wOBAcon .38、xwOBAcon .57 | Clemens 同上 | 未經證實 |
| Whiff%／Contact%、Z-Contact%、SwStr% | K% | 接觸率約 100 PA 穩定（二手）；Carleton 原表無揮棒類 | 接觸率對 K% R² .83（204 人，2016–18）；SwStr% .62；O-Contact >.67。3A whiff%→MLB K% 強於 3A K%（定性） | [Pitcher List](https://pitcherlist.com/going-deep-the-relationship-between-whiffs-and-ks/)；[Bouratoglou](https://maxbouratoglou.substack.com/p/which-minor-league-stats-actually) | 穩定樣本未經證實 |
| O-Swing%（追打）、Z-Swing% | BB%、P/PA | 同上 | O-Swing 對 BB% R² ≈.5，對 K% ≈0；Z-Swing 對兩者皆無關；3A chase→MLB BB% 強於 3A BB%（定性） | [Pitcher List](https://pitcherlist.com/a-beginners-guide-to-understanding-plate-discipline-metrics-for-hitters/)；Bouratoglou | 已核 |
| Sprint speed | SB、3B | 年際 r .59（2017–19）；MLB.com 稱「極強」但無數字 | 對盜壘嘗試率 r .60，對 SB% .30，對 SB/PA R² .33 | [FG](https://fantasy.fangraphs.com/should-i-care-about-sprint-speed/)；[FG](https://fantasy.fangraphs.com/how-sprint-speed-relates-to-stolen-bases/) | **MLB only** |
| OAA | 無 | 年際 Spearman .31–.33（2016–22）；DRS .23–.29；UZR .15；外野 .36–.44、內野 .15–.28 | — | [BP RDA](https://www.baseballprospectus.com/news/article/80209/prospectus-feature-introducing-range-defense-added/) | **MLB only** |
| Arm strength（2020 起） | 無 | 無任何發表的信度數字 | — | [Tango](https://tangotiger.com/index.php/site/comments/statcast-arm-strength-for-fielders) | **MLB only，未經證實** |

## 2. 過程 vs 結果

- **過程／技能量**：擊棒速度、EV90、whiff%、chase%、sprint speed、arm strength。每一次揮棒、每一球、每一次跑都是一筆觀測，分母是揮棒數或球數而不是打席，且不經過守備、球場、對手的二次過濾，所以**雜訊少、很快穩定**（擊棒速度 2–3 揮、EV90 26 BBE、K% 要 60 PA 但 whiff% 每打席有約兩次揮棒）。
- **處理過的結果**：xwOBA、Barrel%、pull-air%。是把結果用物理量重算一次，去掉守備與運氣，但仍是「這次揮棒打到了什麼」。
- **為什麼過程量能預測得比它餵出的結果好**：結果 = 技能 + 運氣，技能量直接量技能；樣本越小優勢越大（Andrews：小樣本看 EV96，大樣本看 EV81）。
- **什麼時候不行**：(1) 單一物理量不帶方向——平均 EV 沒有仰角、拉打方向，是「最不可靠」的；(2) 技能沒有轉成接觸——擊棒速度對 wRC+ 同季 r 只有 .11，年 1 擊棒速度加進 xwOBA 只把 .446 推到 .455，因為揮得快但打不到（squared-up% 與速度負相關 −.67）；(3) 過程量彼此替代——Z-Swing% 對 BB% 與 K% 都無關，只有 O-Swing 跟保送有關。**文獻一致的結論**（Clemens、Salorio、Tango）：Barrel%／xwOBAcon 這類「物理量 + 結果」的中間產物，對隔季 wOBAcon 仍是最強單一預測子，過程量是補充而非取代。

## 3. 小聯盟的轉換（3A 公開資料 2023 起，2022 僅 PCL＋夏洛特）

文獻說的：
- 2023 3A → 2024 MLB wOBAcon（≥50 BBE，n=108）：Barrel% .27、EV90 .27、wOBAcon .21、xwOBAcon .20；≥100 BBE（n=58）：xwOBAcon .55、Barrel% .50、EV90 .45、wOBAcon .39。Statcast 三者都勝過原始 wOBAcon；EV90 穩得最快（[FG](https://fantasy.fangraphs.com/a-quick-look-into-minor-league-statcast-for-hitters/)）。
- 3A whiff%、chase% 預測 MLB K%、BB% 強於 3A 的 K%、BB% 本身；3A HardHit% 解釋 MLB HardHit% 一半以上變異；但 swing%、Z-Swing%、Z-Contact% 對 MLB wOBA 不顯著（[Bouratoglou](https://maxbouratoglou.substack.com/p/which-minor-league-stats-actually)，只含登上者，存活者偏誤）。
- 2A 以下無 Statcast：K% 是最可轉換的結果統計，BB%、ISO 次之；LD%、HR/FB 幾乎無預測力。KATOH 同結論：ISO 與 K% 最重，BB% 在新人聯盟／低階 1A 不預測（[KATOH](https://tht.fangraphs.com/katoh-forecasting-a-hitters-major-league-performance-with-minor-league-stats/)）。
- 球探業界實務：Baseball America 用「接觸率 × EV90 × 追打率」三軸，加 xwOBAcon 做相似群比對，明言不比對 HR、SB 等結果（[BA](https://www.baseballamerica.com/stories/minor-league-hitters-to-target-whose-2024-statcast-data-compares-most-favorably-with-top-mlb-prospects/)）；Longenhagen 的 chat 以 EV90 107–108 當「MLB 前 40」的參考。MLB 隔季 K% ≈ 小聯盟 K% × 1.22（[mlbdatawarehouse](https://www.mlbdatawarehouse.com/p/predicting-mlb-k-with-milb-data)）。
- 未見任何發表數字：3A 擊棒速度、squared-up 的轉換；McDaniel 的 EV90 門檻（ESPN 付費牆）。

我的推論：轉換效度排序大約是 EV90 ≈ Barrel% > xwOBAcon（需 ≥100 BBE）> whiff%、chase%（對 K、BB 本身強，對 wOBA 弱）> 結果統計。PCL 海拔、3A 用球、ABS 判決會把同一組 EV／zone 的意義移位，`data_inventory.md` §2 的風險仍在。

## 4. 映射到五把尺

| 尺 | 現用指標 | 更好的指標（有文獻） | 3A 逐球可算？ | 尺以外的新維度 |
|---|---|---|---|---|
| 打得到球 | AVG、OBP、K%、whiff% | **Z-Contact%**、整體 whiff%（K% R² .83）；squared-up%（對 K% −.67，2024 起） | 是（description + zone）；squared-up 需 bat_speed 欄 | **bat-to-ball 與 swing decision 應拆成兩個維度**：Z-Swing 與 BB/K 無關，O-Swing 只管 BB，whiff 只管 K |
| 打得遠 | HR/AB、XBH/H、Pull%、GB% | **EV90**（年際 .85、26 BBE）、**Barrel%**、pull-air%（取代 Pull%：拉打本身在我們資料 12/12 負，拉打**飛球**才正） | 是（launch_speed、launch_angle、hc_x） | 擊棒速度＝原始力量，EV90＝遊戲力量；兩者差 ≈ 接觸品質 |
| 跑得快 | SB%、SBA/OBT、3B/H、GDP | sprint speed（年際 .59，對嘗試率 .60） | **否**（3A 無） | 無；只能繼續用結果 |
| 選得到球 | BB%、Ball%、P/PA | **O-Swing%**（對 BB% R² .5；P/PA、Ball% 被動且混了接觸） | 是（zone + description） | 「選球」拆成 chase（避壞球）與 in-zone aggressiveness，後者在文獻裡對 BB/K 都無關 |
| 擊球品質（第五把） | LD%、IFFB%、BABIP | **xwOBAcon**（≥100 BBE 轉換 .55）、sweet-spot%（弱，.40–.59 不一致） | 是 | 無 |
| 臂力、守備（缺口） | 無 | arm strength、OAA（年際僅 .31） | **否** | 只能在登上後驗證（現行 `validate_defense.py`） |

## 5. 建議（優先序）

1. **EV90**（3A 2023–24）：年際最穩、26 BBE 即可、轉換 .45；進 `hit_profile_statcast.py` 取代／並列 HardHit%。限制：PCL 海拔需球場因子。
2. **Barrel%**：對隔季 wOBAcon 最強單一子（.5+），已有係數 +0.71；保留為結果型指標。限制：需 ≥50–60 BBE。
3. **Z-Contact%** 與 **O-Swing%**：各自對 K%、BB% 的過程版，3A 可算；把選球拆成兩維，能直接檢驗第 4 節「選球沒幫忙」是 chase 還是 in-zone 的問題。限制：ABS 改變 3A 的 zone 定義，需按季標準化。
4. **xwOBAcon**：第五把尺的效標（factor_plan §16）；限制：需 ≥100 BBE，430 人季會掉一半。
5. **pull-air%**（拉打 × 仰角 ≥ 某值）：取代現用 Pull%，可解釋拉打率殘差為負的矛盾。限制：定義需自訂，3A 僅兩季。
6. **MLB 季彙總 Statcast 欄**（FanGraphs 473 欄裡的 EV90、Barrel%、O-Swing%、Z-Contact%、xwOBA）：A 段學權重用，10 月底前可抓。
7. 擊棒速度、squared-up：先確認 3A `bat_speed` 欄是否有值；文獻顯示增益小（.446→.455），列為可選。
8. sprint speed、OAA、arm：小聯盟無資料，僅作登上後驗證，不進升級模型。
