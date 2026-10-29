#!/usr/bin/env bash
# 一個 fit 拆成三個節點：M0、M1、Mh 各開一個 screen + srun 同時跑。
#   scripts/arc_fit3.sh <season> [fit_year.py 參數...]
# 三個都出現 idata 之後彙整：scripts/arc_fit.sh <season> [同樣參數] --reuse
cd "$(dirname "$0")/.."
for mode in M0 M1 Mh; do scripts/arc_fit.sh "$@" --only $mode; done
