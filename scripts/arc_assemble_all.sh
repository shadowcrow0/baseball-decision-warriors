#!/usr/bin/env bash
# 全年份彙整：哪一年的三個後驗檔（M0、M1、Mh）都到了、而且還沒有 summary.md，就用 --reuse 彙整那一年（不取樣）。
# 一個 srun job 跑完全部，log 在 outputs/logs/assemble_all.log。
#   scripts/arc_assemble_all.sh            升級 + WAR
#   DRY_RUN=1 scripts/arc_assemble_all.sh  只印不送
cd "$(dirname "$0")/.."
MODEL="--factors 4 --scout --extra --direct"
TAG=f4sed_linear_p2
todo=()
for outcome in promoted war; do
  for d in outputs/20*/${outcome}_${TAG}; do
    s=$(echo "$d" | cut -d/ -f2)
    [ -f "$d/idata_M0.nc" ] && [ -f "$d/idata_M1.nc" ] && [ -f "$d/idata_Mh.nc" ] || continue
    [ -f "$d/summary.md" ] && continue
    todo+=("$s $outcome")
  done
done
echo "要彙整 ${#todo[@]} 個（年份 結果）："; printf '  %s\n' "${todo[@]}"
[ "${#todo[@]}" -eq 0 ] && exit 0
[ -n "${DRY_RUN:-}" ] && exit 0
cmds=""
for item in "${todo[@]}"; do
  set -- $item
  cmds+="echo '== $1 $2'; python src/fit_year.py data/ $1 outputs/ --outcome $2 $MODEL --reuse 2>&1 | tail -3; "
done
inner="type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || true;
module load anaconda3/2024.10-1 && source ~/pyb_env/bin/activate && cd ~/baseball && { $cmds } 2>&1 | tee outputs/logs/assemble_all.log"
screen -dmS assemble_all bash -c "srun -p ${SRUN_PART:-compute1} -n 1 -t ${SRUN_TIME:-06:00:00} --cpus-per-task=8 bash -lc \"$inner\"; echo '=== 彙整結束，Enter 關閉 ==='; read -r"
echo "已啟動 screen assemble_all；看進度：tail -f outputs/logs/assemble_all.log"
