#!/usr/bin/env bash
# 全年份正式網格：升級 2007–2024（跳 2006 缺指標、2020 停賽）、WAR 2007–2019（六季窗口）。
# 已經有 idata 的 (年份, 模型) 跳過，所以可以重複執行、斷掉再接。WAR 的 job 時限 48 小時（分區上限 3 天），升級 24 小時。
# 帳號同時最多 10 個 job，其餘排隊；全部約 4–5 天。
# 用法：scripts/arc_grid_all.sh            全部
#       scripts/arc_grid_all.sh promoted   只升級
#       scripts/arc_grid_all.sh war        只 WAR
#       DRY_RUN=1 scripts/arc_grid_all.sh  只印不送
cd "$(dirname "$0")/.."
MODEL=(--factors 4 --scout --extra --direct)
TAG=f4sed_linear_p2
what="${1:-all}"
sent=0; skipped=0
launch() {   # launch <season> <outcome> <time>
  local s=$1 outcome=$2 t=$3
  for mode in M0 M1 Mh; do
    if [ -f "outputs/$s/${outcome}_${TAG}/idata_${mode}.nc" ]; then skipped=$((skipped+1)); continue; fi
    log="outputs/logs/${s}_${outcome}_${TAG}_${mode}.log"
    # log 兩天內有寫入、又還沒有「已存」= 正在跑（screen 在別的登入節點也看得到 log），跳過
    if [ -f "$log" ] && [ -n "$(find "$log" -mmin -2880)" ] && ! grep -q "已存" "$log"; then echo "跳過（正在跑）: $s $outcome $mode"; skipped=$((skipped+1)); continue; fi
    if [ -n "$DRY_RUN" ]; then echo "would launch: $s $outcome $mode ($t)"; else
      SRUN_TIME=$t scripts/arc_fit.sh $s --outcome $outcome "${MODEL[@]}" --only $mode >/dev/null
    fi
    sent=$((sent+1))
  done
}
if [ "$what" = all ] || [ "$what" = war ]; then
  for s in 2019 2018 2017 2016 2015 2014 2013 2012 2011 2010 2009 2008 2007; do launch $s war 48:00:00; done
fi
if [ "$what" = all ] || [ "$what" = promoted ]; then
  for s in 2024 2023 2022 2021 2019 2018 2017 2016 2015 2014 2013 2012 2011 2010 2009 2008 2007; do launch $s promoted 24:00:00; done
fi
echo "送出 $sent 個 job，跳過 $skipped 個已有的。scripts/arc_status.sh 看狀態；三個 idata 都到了之後 --reuse 彙整。"
