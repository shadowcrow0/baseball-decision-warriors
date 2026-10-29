#!/usr/bin/env bash
# Arc 上一眼看完：哪些結果已經有了（✓）、哪些有 log 但還沒有結果（⏳，在跑或失敗，印 log 最後一行）、哪些根本沒開（✗）。
#   bash scripts/arc_status.sh
# 只讀檔案，不動任何 job。最後印 squeue 的 R / PD 數與 screen 數。
cd "$HOME/baseball" || exit 1
P=promoted_f4sed_linear_p2; W=war_f4sed_linear_p2; L=outputs/logs

check() {  # check <名稱> <結果檔> <log 檔>
  local name="$1" out="$2" log="$3"
  if [ -s "$out" ]; then
    printf "✓  %-34s %s\n" "$name" "$(date -r "$out" '+%m-%d %H:%M')"
  elif [ -e "$log" ]; then
    printf "⏳ %-34s %s\n" "$name" "$(tail -n 1 "$log" 2>/dev/null | cut -c1-90)"
  else
    printf "✗  %-34s 沒開\n" "$name"
  fi
}
count() {  # count <名稱> <glob> <應有數>
  local n; n=$(find $2 -size +0 2>/dev/null | wc -l)
  printf "%s  %-34s %s / %s\n" "$([ "$n" -ge "$3" ] && echo ✓ || echo ⏳)" "$1" "$n" "$3"
}

echo "== 後處理（全季）"
check "all_seasons discipline_split" outputs/discipline_split_all_$P.md $L/run_ds_all.log
check "all_seasons club_need"        outputs/club_need_all_$P.md        $L/run_cn_all.log
check "all_seasons org_background"   outputs/org_background_all_$P.md   $L/run_ob_all.log
check "all_seasons validate"         outputs/validate_abilities_all_$P.md $L/run_va_all.log
check "variable_hunt 17 季"          outputs/variable_hunt_$P.md        $L/run_vh.log
check "ppc_all 升級段"               outputs/ppc_all_$P.md              $L/run_ppc_all.log
check "ppc_all WAR 段"               outputs/ppc_all_$W.md              $L/run_ppcw_all.log
check "forward_check"                outputs/forward_check_$P.md        $L/run_fwd.log
check "bayes_factors 主模型"         outputs/bayes_factors_promoted_war.md $L/run_bfm.log
check "club_risk"                    outputs/club_risk_$P.md            $L/run_crisk.log
check "learn_weights"                outputs/learn_weights_$P.md        $L/run_lw.log
check "eta_quantiles"                outputs/eta_quantiles_$P.md        $L/run_etaq.log

echo "== decision_bayes（WAIC／LOO）"
check "2008–2011" outputs/decision_bayes_${P}_2008_2011.md $L/run_dbayes1.log
check "2012–2015" outputs/decision_bayes_${P}_2012_2015.md $L/run_dbayes2.log
check "2016–2019" outputs/decision_bayes_${P}_2016_2019.md $L/run_dbayes3.log
check "2021–2024" "outputs/decision_bayes_${P}_2021_2024.md" $L/run_dbayes4.log
[ -e outputs/decision_bayes_$P.md ] && echo "   （另有舊檔名 decision_bayes_$P.md = 改名前的 2021–2024）"

echo "== 參數還原"
count "主模型 binom 複本"      "outputs/sim/recover_2015_promoted_f4s_binom_rep*.json" 20
count "主模型 betabinom 複本"  "outputs/sim/recover_2015_promoted_f4s_betabinom_rep*.json" 20
count "WAR 段 複本"            "outputs/sim/recover_2015_war_f4s_binom_rep*.json" 20
count "五把尺 複本"            "outputs/sim/recover_2015_promoted_f5s_binom_rep*.json" 10
count "進階量 N 複本"          "outputs/sim/recover_2024_promoted_f5sN_binom_rep*.json" 10
count "進階量 O 複本"          "outputs/sim/recover_2024_promoted_f5sO_binom_rep*.json" 5
count "進階量 B 複本"          "outputs/sim/recover_2024_promoted_f5sB_binom_rep*.json" 5
count "球團斜率 複本"          "outputs/sim/recover_2015_promoted_f4so_binom_rep*.json" 10
count "型態 複本"              "outputs/sim/recover_2015_promoted_f4sk_binom_rep*.json" 5
check "彙整" outputs/sim_recover_summary.md $L/run_rec_sum.log
count "決策程序還原（5 真程序 × 4）" "outputs/sim/decision_2015_*_rep*.json" 20
check "決策程序彙整" outputs/sim_decision_summary.md $L/run_dec_sum.log

echo "== 進階量 vs 成績單（同一批 3A 人）"
for s in 2023 2024; do for m in N O B; do check "fit $s f5s$m" outputs/$s/promoted_f5s${m}_linear_p2/summary.md $L/${s}_promoted_f5s${m}_linear_p2.log; done; done
for s in 2023 2024; do
  check "compare $s N vs O" outputs/$s/compare_promoted_f5sN_linear_p2_vs_promoted_f5sO_linear_p2.md $L/run_cmpNO${s: -2}.log
  check "compare $s N vs B" outputs/$s/compare_promoted_f5sN_linear_p2_vs_promoted_f5sB_linear_p2.md $L/run_cmpNB${s: -2}.log
  check "compare $s f5sed vs f5sedS" outputs/$s/compare_promoted_f5sed_linear_p2_vs_promoted_f5sedS_linear_p2.md $L/run_cmp${s: -2}.log
done

echo "== 主模型補全季"
count "五把尺 升級段（應 17）" "outputs/20*/promoted_f5sed_linear_p2/summary.md" 17
count "五把尺 WAR 段（應 13）"  "outputs/20*/war_f5sed_linear_p2/summary.md" 13
count "球隊 × 型態（應 17）"    "outputs/20*/promoted_f4sedck_linear_p2/summary.md" 17
count "構念驗證 v2"             "outputs/20*/promoted_f4sedv_linear_p2/construct_M1.csv" 1
count "球團各自的尺 f4so（應 17）" "outputs/20*/promoted_f4so_linear_p2/org_M1.csv" 17
check "org_rulers 跨季合併"        outputs/org_rulers_promoted_f4so_linear_p2.md $L/run_orgr.log
check "sim_policy"                  outputs/sim_policy_promoted_f4sed_linear_p2.md $L/run_simpol.log

echo "== 隊列"
printf "R %s  PD %s  screens %s\n" "$(squeue -u "$USER" -h -t R | wc -l)" "$(squeue -u "$USER" -h -t PD | wc -l)" "$(screen -ls 2>/dev/null | grep -c Detached)"
echo "== 失敗的 log（最後一行含 Error/Traceback/找不到）"
grep -l "Error\|Traceback\|找不到\|No such file" $L/*.log 2>/dev/null | while read -r f; do printf "   %s: %s\n" "$(basename "$f")" "$(tail -n 1 "$f" | cut -c1-80)"; done
