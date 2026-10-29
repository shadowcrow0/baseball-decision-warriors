#!/usr/bin/env bash
# 在 compute 節點上跑任意一支 python 腳本（screen + srun），給抓取與後處理用。
#   scripts/arc_run.sh <名稱> python src/xxx.py 參數...   （環境變數 CPUS=16 SRUN_TIME=06:00:00 SRUN_PART=compute2 可調核心數、時限、分區；log 即時寫出）
set -euo pipefail
name="run_$1"; shift
mkdir -p "$HOME/baseball/outputs/logs"; log="$HOME/baseball/outputs/logs/${name}.log"
inner="type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh 2>/dev/null || true;
module load anaconda3/2024.10-1 && source ~/pyb_env/bin/activate && cd ~/baseball && PYTHONUNBUFFERED=1 $* 2>&1 | tee ${log}"
screen -dmS "${name}" bash -c "srun -p ${SRUN_PART:-compute1} -n 1 -t ${SRUN_TIME:-02:00:00} --cpus-per-task=${CPUS:-2} bash -lc '${inner}'; echo; echo '=== 結束，Enter 關閉 ==='; read -r"
echo "已啟動 ${name}；tail -f ${log}"
