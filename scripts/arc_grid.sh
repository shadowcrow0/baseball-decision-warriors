#!/usr/bin/env bash
# 正式網格：每個 fit 拆 M0/M1/Mh 三個節點（arc_fit3），1000+1000 × 4 鏈。
# 帳號同時最多 10 個 job，多的會排隊（AssocMaxJobsLimit），不用管，screen 視窗會等。
# 用法：scripts/arc_grid.sh            全部（升級四年 + WAR 三年 = 21 個 job）
#       scripts/arc_grid.sh promoted   只升級
#       scripts/arc_grid.sh war        只 WAR
cd "$(dirname "$0")/.."
MODEL=(--factors 4 --scout --extra --direct)
what="${1:-all}"
if [ "$what" = all ] || [ "$what" = promoted ]; then
  for s in 2010 2015 2019 2023; do scripts/arc_fit3.sh $s "${MODEL[@]}"; done
fi
if [ "$what" = all ] || [ "$what" = war ]; then
  for s in 2010 2015 2019; do scripts/arc_fit3.sh $s --outcome war "${MODEL[@]}"; done
fi
echo; echo "已送出；scripts/arc_status.sh 看狀態。每個 tag 三個 idata 都到了之後："
echo "  scripts/arc_fit.sh <season> [同樣參數] --reuse   彙整"
