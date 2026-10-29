# 模型規格（貝氏生成式寫法）

> 配合 `research_design.md` 第 6–11 節。打者版；投手版把指標換成三振、保送、全壘打、滾地、揮空（分母 TBF、投球數），結構相同。

## 索引
- i 球員、t 球季、j 指標、o 球團、ℓ 層級、d 選秀年
- 觀測單位 (i, t)：一人一季。跨層級的季：分子分母跨層加總，ℓ 取主要層級。

## 測量模型（四 factor，見 `factor_plan.md`）
```
η_it ∈ R^4 = (contact, power, speed, discipline)
η_it ~ MultivariateNormal(μ_it, Ω)，Ω = L Lᵀ，L ~ LKJCholesky(η=2)，對角固定為 1（尺度識別）

二項指標 j（分子/分母）
  接觸：AVG H/AB、OBP OB/PA、K% SO/PA、whiff% whiff/Pitches
  力量：XBH/H、Pull% PULL/BIP、GB% GB/BIP
  速度：SB% SB/SBA、SBA/OBT、3B/H、GDP/AB
  選球：BB% BB/PA、Ball% BALLS/Pitches
  y_ijt ~ Binomial(n_ijt, p_ijt)
  logit(p_ijt) = γ_jℓ + Σ_k λ_jk · η_itk

計數指標 j
  力量：HR | AB；選球：Pitches | PA
  y_ijt ~ NegBinomial(μ_ijt, φ_j)
  log(μ_ijt) = log(n_ijt) + γ_jℓ + Σ_k λ_jk · η_itk
```
- λ_jk 只在指標 j 所屬 factor k 非零（簡單結構）；`--crossload` 另開 K%、whiff% → power，P/PA → contact。
- 符號識別：每個 factor 一個錨，K% 負、HR/AB 正、SBA/OBT 正、BB% 正。其他負荷量符號由資料決定。
- γ_jℓ 是指標 j 在層級 ℓ 的截距（不另設總截距 α_j：α 與七個 γ 只差一個常數，會形成取樣器走不完的山脊）。只對該季該指標有資料的層級建參數，沒資料的層級固定 0（例如 2015 年沒有 DSL）；整季沒有資料的指標不進模型（2006 年沒有揮空、滾地、拉打、壞球、投球數，選球只剩保送率一個指標）。每次 fit 開頭印「識別檢查」列出這些。打席少的人 n_ijt 小，似然平坦，自然低權重，不設門檻。
- `--factors 1/2/3` 是併合版本（`FOLD`），作 model comparison 的基準。

## factor 的均值（球員層、階層）
```
μ_it = a_i + b_i · (t − t_i0) + δ · age_rel_it
  a_i ~ Normal(θ_org(i) + θ_draft(i), σ_a)
  b_i ~ Normal(0, σ_b)
  θ_o = σ_org · (z_o − mean_o z_o)，z_o ~ Normal(0, 1)   每個 factor 一組，跨球團總和為零（factor 的位置由此識別）
  θ_d ~ Normal(0, σ_draft)
```

## 球探先驗 factor（`--scout`）
```
ζ_i ~ Normal(0, 1)                      每位選秀球員一個，不隨球季變；尺度固定 1
z(log pick_i)  ~ Normal(−λ ζ_i, σ_p)     標準化後的 log 順位；ζ 高 = 評價高 = 順位靠前
z(log bonus_i) ~ Normal(+λ ζ_i, σ_b)     標準化後的 log 簽約金，只對有資料的人；缺值不進似然
λ ~ HalfNormal(1)                        兩個指標負荷量大小相等（識別：兩指標一因子只有三個動差，需一個約束）
```
- 順位與簽約金是同一個潛在評價（球探＋資料團隊在簽約當下的共識）的兩次測量，當兩個獨立共變數會共線（r = 0.98）。
- 非選秀球員（國際自由球員）沒有 ζ，結構模型用 `undrafted` 的 0/1 給他們自己的基準率；出身 HS/JC 為 0/1 控制項（基準：大學）。
- ζ 進 M0、M1、Mh 三個模型：M0 = 只有球探先驗，M1 = 球探先驗 + 四個表現 factor。β_scout 對 β_contact 等四個的比較就是「順位之外還剩多少」。

## 結構模型 A（球團的決定）
```
z_it ∈ {promoted, same, demoted, exit, mlb}；censored 與 gone_return 間隔不進似然
z_it ~ Categorical(π_it)
log(π_it,k / π_it,same) = γ_k,ℓ + Σ_m β_km · f(η_itm) + β_scout,k · ζ_i  （或無 --scout 時：ρ_k · log(pick_i) + ω_k · log(bonus_i)）
                          + ψ_k · age_rel_it + u_k,org(i),t
  f(η) ∈ {η, η + η², spline(η)}                          model comparison
  u_k,o,t ~ Normal(0, τ_k)                               球團 × 球季隨機效應；exit 的 τ 是共享隨機性
```
二元版本：promoted ∪ mlb 對其他，Bernoulli，同一線性預測式。

## 結構模型 B（事後的真相）：兩段
```
第一段：幾年登上大聯盟（加速失敗時間，右設限）
  T_i = 首登年 − 本季 + 1 ∈ {1, …, 7}；六季窗口內沒登上 → T_i > 7（設限）
  log T_i ~ Normal(μ_t,i, σ_t)
  μ_t,i = γ_t,ℓ + Σ_k β_t,k η_ik + β_t,s ζ_i + 共變數 + u_t,org
  β_t 為負 = 該能力讓人更快登上；CDF P(T ≤ t) 直接可讀（三年內登上的機率）

第二段：登上後六季累積 fWAR（asinh 尺度）
  asinh(WAR_i) | 登上 ~ StudentT(ν, μ_w,i, σ_w)，ν = 3 + Gamma(2, 0.5)，μ_w 同型線性預測式
  （WAR 一半 ≤ 0、中位數 0.03、最高 30；不轉換時 t 的自由度估成 0.7，位置只盯著中位數，係數全為 0）
```
- 第一段取代原本的二元「登上與否」：同樣的資料，多了「多快」和「還沒登上 ≠ 不會登上」的訊息。
- 兩段的 pointwise log-lik 加總成 hurdle 的 LOO。λ^A（升級下估）與 λ^B（WAR 下估）的差距 = 球團評價與現實的落差。

## 構念從哪裡來（2026-10-19 定稿）

構念**事先定義**，不是從資料裡 feed 出來再取名：球探的五項工具（打擊、力量、速度、臂力、守備）加賽伯計量的選球。成績單量得到其中四個：打得到球、打得遠、跑得快、選得到球。十五個指標依定義歸屬（`INDICATORS` 表第五欄），負荷量符號用棒球常識錨定（`INIT_SIGN`、`ANCHORS`），資料只估負荷量的大小，先驗 Normal(0, 1)。臂力與守備在成績單上沒有任何指標，構念掛空不估；這是測量的缺口，不是分析的選擇。守備那一塊用登上後的 Savant OAA 與 FanGraphs 純守備分事後驗證（`validate_defense.py`）。

探索性分析（factor_plan §2：相關群、平行分析、varimax）只用來回答「十五個數字撐得起幾個維度、選球要不要從接觸分出來」，沒有用來命名構念。

**驗證性檢查**（`--verify`，tag 加 v）：錨指標（K%、HR/AB、SBA/OBT、BB%、BABIP）以外的每個指標都可以載到定義之外的構念，先驗 Normal(0, 0.3)；錨指標的交叉負荷固定 0，每把尺至少有一個只屬於它的指標，因子才不能任意旋轉（第一版全部放開，四個因子方向沒釘住，主負荷量被拉到零、r-hat 2–3，不能用）。`construct_M1.csv` 每個指標一列：定義上的構念、先驗符號、後驗主負荷、對其他構念的交叉負荷（附相對主負荷的百分比）、判定（89% 區間不含 0 而且 |交叉| > |主負荷| 的一半 = 跑過去）。跑過去的指標代表定義有問題，要改歸屬或承認它是兩個構念的混合（已知：三振率、揮空率同時載到打得遠）。

## Prior
```
λ_jk ~ Normal(0, 1)，每個 factor 的錨指標限定符號（K% 負、HR/AB 正、SBA/OBT 正、BB% 正）
γ_jℓ, γ_s,ℓ ~ Normal(0, 2)
β, ρ, ω, ψ, δ ~ Normal(0, 1)
σ_a, σ_b, σ_org, σ_draft, τ_k ~ HalfNormal(1)
φ_j ~ Gamma(2, 0.5);  ν ~ Gamma(2, 0.1)
```

## 序列更新
```
prior^(t) = posterior^(t−1)，變異數 × (1 + κ)，κ 遺忘係數（候選 0.1 / 0.25 / 0.5，用留出年份挑）
2019 → 2021：κ 加倍
輸出：λ_j^(t) 逐年軌跡；‖λ^A(t) − λ^B(t)‖ 逐年
```

## 跨球團
```
λ_jo = λ_j + ε_jo,  ε_jo ~ Normal(0, σ_λ)      球團專屬負荷量偏移（收縮）
距離共識：‖ε_o‖；距離真相：‖λ_j + ε_jo − λ_j^B‖
```
樣本限制：2015 年後才可能有足夠訊號。

### 球團層（`--club`，已實作於 fit_year.py）

球員在球團裡，球團是第二層。第二層有兩件事：

```
截距：u_o = a_club · ClubWAR_o + Σ_k a_k · θ_org[o, k] + τ · z_o
      ClubWAR_o  母隊當季團隊 WAR（標準化）；a_club < 0 = 弱隊多升（缺人）
      θ_org[o,k] 這個球團手上球員的能力平均（測量模型估的）；a_k < 0 = 農場人才越多越不升（塞車）
      前兩項 = u_org_ctx（截距被解釋的部分），τ·z_o = 解釋完剩下的（文化、政策）
個人：logit P(升_i) = … + b_need · 缺口_{o(i),pos(i)} + b_top · 主力WAR + b_above · 農場同守位層級較高人數 + b_total · 農場同守位人數
                      + Σ_k (β_k + b_need×k · 缺口) · η_ik + (β_s + b_need×s · 缺口) · ζ_i
      缺口 = −(母隊當季該守位群 WAR 加總)，標準化；守位群來自小聯盟 positionDB，大聯盟主守位來自 fg_positions.parquet
      b_need > 0        缺什麼補什麼（守位缺人就多升）
      b_need×k < 0      缺人時這個能力的權重變小（門檻降低）；> 0 缺人時反而更挑這個能力
      b_top < 0         主力強就卡住；b_above < 0 農場上面有人就塞住
```
三段（升級、登上時間、WAR|登上）各有自己一組；WAR 段的 b_need×k 回答「缺人補上去的人後來打得好不好」。
tag 加 c；需要 `data/fg_positions.parquet`（`fetch_fg_positions.py`）。

## Model comparison
PSIS-LOO 為主、WAIC 為對照，兩者皆報 ELPD 差與標準誤（`az.compare`）。留出單位 = 球員 i（整條路徑）。
LOO 的 Pareto-k > 0.7 的球員列出來，是對模型影響最大的個案（預期含海外回歸者）。候選維度：
- f(η)：線性 / 二次 / 樣條
- η 維度：1 / 2 / 3 / 4 / 4+crossload
- pick：有 / 無
- u_org,t：有 / 無
- outside_return：有 / 無（球員頁補完後）
主結果：「pick + η」對「只有 pick」的 ΔELPD，與留出年份 AUC。

## 待定
- 盜壘放成功率（Binomial）還是頻率（NegBinomial offset PA），或兩個都放。
- 投手版是否與打者版共用結構模型的 β。

## 實作
pymc ≥ 5，NUTS；每年一個 fit，約 5–8 千人季、十個指標，預估每年 10–20 分鐘。


## 第二階段：四年合併的階層模型（規格，可直接寫成 `src/fit_pooled.py`）

**資料**：2010、2015、2019、2023 四個球季的人季疊起來（約 15,500 人季、約 12,000 人）。同一個人可能出現在兩個年份（例如 2010 與 2015）。每人季的結果同單年：升級（0/1）、幾年後登上（設限）、WAR|登上（只有登上的人）。

**測量模型**：與單年相同（15 指標 → 4 能力），但負荷 λ、層級截距 γ 加一個年份維度：λ 共用（能力的定義跨年固定，這是可以比較的前提），γ[年份, 層級] 各自估（投手環境逐年變）。θ_org 與 δ 也各年一組。

**結構模型**：三個結果各一條預測式，係數分兩層：
```
β[年份, k] = β_0[k] + β_year[年份, k],  β_year ~ Normal(0, σ_β[k])
```
β_0 是「四年共同的效果」（小效果目錄要的那個合併估計），β_year 是各年的偏離，σ_β 是「這個效果跨年穩不穩」。選得到球在升級段的 β_year 應該會顯示 2010 → 2023 的單調縮小。共變數 b、球探評價 β_ζ 同樣處理。

**共同隨機來源 u_i**（球員層）：
```
u_i ~ Normal(0, τ_u)                      每個球員一個
升級:     lin_p += a_p · u_i               a_p 固定 = 1（定尺度）
幾年後登上: lin_t += a_t · u_i             a_t 自由
WAR|登上:  lin_w += a_w · u_i              a_w 自由
```
τ_u 是「成績單與球探都量不到、但同時推三個結果的東西」有多大；a_t、a_w 的符號說它推的方向是否一致。跨年出現兩次的人：u_i 拆成 u_i = v_i + e_{i,年份}，v_i 跨年穩定（球員本身）、e 逐年（當年的傷與環境）；只出現一次的人兩者分不開，靠出現兩次的人釘住 τ_v 與 τ_e 的比例。

**登上段的交互作用**（強弱隊檢查過線的兩條，四年同號才留）：
```
lin_t += c_1 · 跑得快 × 農場弱 + c_2 · 選得到球 × 母隊強
```
母隊強弱 = 當季大聯盟團隊 WAR 高於中位；農場強弱 = 該球團球員期望 WAR 平均高於中位（第一階段的分數算）。

**識別**：u_i 要有兩個以上的結果才釘得住，升級與登上時間每人都有，WAR 只有登上者有；a_p = 1 定尺度；η 與 u_i 的區別靠「η 由 15 個指標釘住、u_i 不碰指標」。

**取樣**：參數量約單年的四倍加 12,000 個 u_i；ADVI 初始化照舊；先 2 年（2015、2019）試跑 200 抽樣看時間，再四年。預估單一 job 超過 24 小時，要用 `SRUN_TIME` 拉長或分段存檔。

**輸出**：β_0 與 σ_β 的表（小效果目錄）、τ_u 與 a_t、a_w、τ_v/τ_e 的比例、每人 u_i 的後驗（錯放名單的第四個欄位：成績單、球探、共同來源各貢獻多少）。

## 進階指標與球隊型態偏好（2026-10-23）

- `--statcast` 五種模式（2026-10-25）：`add` 成績單 + Statcast、全樣本（S）；`same` 成績單、只留有 Statcast 的 3A 人季（N）；`only` Statcast 自己組尺（加成績單的速度四個）、同一批人（O）；`both` 兩個都放、同一批人（B）。**問「進階量組的尺 vs 成績單組的尺哪個比較會預測」要用 same / only / both 三個 fit**，同一批人、同一個結果，升級段 LOO 直接比（`compare_fits.py`）。跑這三個時不要加 `--extra`、`--direct`，否則成績單的比率會從結構式漏進 only 那邊。
- `--statcast add|only`（舊說明）：3A 逐球（Savant）算出的過程量當同一組構念的指標。add = 成績單 15 個 + 區內接觸率（打得到球）、EV90、barrel 率、拉打飛球率（打得遠）、追打率（選得到球，負）、EV10、接觸後 xwOBA（擊球品質）；只有 3A 人季有，其他層級的格子不建參數。連續量層級內標準化後進常態似然，sigma × sqrt(100 / 進場球)。only = 只用 3A、只用進階量加成績單的速度四個，錨換成 Z-Con%、EV90、SBA/OBT、O-Swing%（負）、xwOBAcon。區內接觸率與追打率要有進壘點欄的資料（2026-10-23 後抓的），沒有自動不進模型。
- `--club-type`（需 `--club`）：升級段加 母隊強弱 × (η_power − η_contact) 與每隊自己對型態的斜率；tag 加 k。鏡像對齊時若打得到球或打得遠有翻轉的 draw 會印警告（型態的定義在那些 draw 不成立）。
- `src/compare_fits.py`：同一季兩個 tag 的 M1 後驗並排（負荷量、β 的 B − A 後驗與 P(B > A)、球員 η 相關與移動、球團、LOO）。

## 後驗預測檢查（`src/ppc.py`，2026-10-24）

模型假設的分布跟實際資料合不合，不靠 r-hat，靠把資料重新生一次：從 M1 後驗抽 200 組參數，照每個指標的似然（二項、負二項、常態）把每個人的計數生出來，跟真實比——跨人比率的分位數、Pearson 卡方的離散比（真實 ÷ 模型生的，> 1.1 且 p < 0.05 = 過度離散，該換 beta-binomial 或放寬 φ）、零的比例、最大值。結構段：升級的十分位校準；WAR tag 的 asinh(WAR) 分位數與 ≤ 0 比例（StudentT）、幾年登上的分布與設限比例（log-normal AFT）。圖：每個指標一格，黑 = 真實、紅 = 20 份模型生的。`python src/ppc.py outputs/ data/ <season> <tag> [--statcast add]`，輸出 `outputs/<season>/<tag>/ppc.md` 與 `outputs/figures/ppc_<season>_<tag>.png`。

## 證據強度：Bayes factor，不用 p < .05（2026-10-25）

- 係數類（`src/bf.py`，所有後處理表的 BF10 欄）：Savage–Dickey 近似。H1 的先驗 β ~ N(0, τ²)，τ = 0.5（標準化 logit 尺度上的中等效果）；似然用 N(b̂, se²) 近似；BF10 = 先驗在 0 的密度 ÷ 後驗在 0 的密度。讀法（Jeffreys）：> 100 極強、10–100 強、3–10 中等、1/3–3 分不出來、< 1/3 支持零。敏感度：τ = 0.2、1.0 同方向才算穩（`bf.sensitivity`）。
- 兩群平均差（`dist_shapes.py`）：同一個 Savage–Dickey，差的先驗是 0.5 個合併標準差。
- 主模型結構係數（`src/bayes_factors.py`）：從 idata 的後驗抽樣算，先驗就是模型裡的 N(0, 1)，後驗在 0 的密度用 KDE；跨季 log10 BF 相加 = 「所有季皆零」vs「皆非零」。
- 模型比較（`decision_bayes.py`、`compare_fits.py` 的 LOO）：pseudo-Bayes factor，log10 BF = Δelpd ÷ ln 10（Geisser–Eddy）。
- 後驗預測檢查的 p（`ppc.py`）是 posterior predictive p，不是假設檢定的 p，保留；判定仍以離散比的大小為準。
