#!/usr/bin/env bash
# 在 Arc 上用 screen + srun 開一個 fit。從 login 節點執行，不用先 srun。
#   scripts/arc_fit.sh <season> [fit_year.py 的其他參數...]
# 例： scripts/arc_fit.sh 2015 --outcome war --factors 4 --crossload
#      scripts/arc_fit.sh 2010 --factors 4 --crossload --quick
# 每個 fit 一個 screen 視窗，名稱 fit_<season>_<tag>；log 在 outputs/logs/。
# 時限預設 24 小時，可用 SRUN_TIME="10:00:00" 改；partition 預設 compute1，可用 SRUN_PART=compute2 改。要指定 srun 額外參數用 SRUN_EXTRA，例：SRUN_EXTRA="--exclusive"（整台機器）或 SRUN_EXTRA="--exclude=c043"。
# 從 login 節點執行；在 compute 節點的配置裡執行會變成同一配置的 job step，共用那 8 核。
set -euo pipefail
season="$1"; shift
args=("$@")
outcome=promoted; factors=1; feta=linear; poly=2; cross=""; nc=""; only=""; sc=""; ex=""; dr=""; av=""; wo=""; os=""; cl=""; ct=""; vf=""; st=""
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[$i]}" in
    --outcome) outcome="${args[$((i+1))]}";;
    --factors) factors="${args[$((i+1))]}";;
    --feta) feta="${args[$((i+1))]}";;
    --poly) poly="${args[$((i+1))]}";;
    --crossload) cross=x;;
    --scout) sc=s;;
    --extra) ex=e;;
    --direct) dr=d;;
    --avg) av=a;;
    --woba) wo=w;;
    --org-slopes) os=o;;
    --verify) vf=v;;
    --club) cl=c;;
    --club-type) ct=k;;
    --statcast) case "${args[$((i+1))]}" in add) st=S;; same) st=N;; only) st=O;; both) st=B;; esac;;
    --param) [ "${args[$((i+1))]}" = noncentered ] && nc=_nc;;
    --only) only="_${args[$((i+1))]}";;
  esac
done
tag="${outcome}_f${factors}${cross}${sc}${ex}${dr}${av}${wo}${os}${cl}${ct}${vf}${st}_${feta}_p${poly}${nc}"
name="fit_${season}_${tag}${only}"
mkdir -p "$HOME/baseball/outputs/logs"
log="$HOME/baseball/outputs/logs/${season}_${tag}${only}.log"
if screen -list | grep -q "\.${name}[[:space:]]"; then
  echo "已有視窗 ${name}，先 screen -r ${name} 看看"; exit 1
fi
inner="type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || true;
module load anaconda3/2024.10-1 && source ~/pyb_env/bin/activate && cd ~/baseball &&
python src/fit_year.py data/ ${season} outputs/ ${args[*]} 2>&1 | tee ${log}"
outer="srun -p ${SRUN_PART:-compute1} -n 1 -t ${SRUN_TIME:-24:00:00} --cpus-per-task=8 ${SRUN_EXTRA:-} bash -lc '${inner}';
echo; echo '=== srun 結束（節點已釋放），log 在 ${log}；Enter 關閉視窗 ==='; read -r"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "screen -dmS ${name} bash -c <<<"; echo "${outer}"; exit 0; fi
screen -dmS "${name}" bash -c "${outer}"
echo "已啟動 ${name}；看進度：screen -r ${name}（Ctrl-A D 離開）或 tail -f ${log}"
