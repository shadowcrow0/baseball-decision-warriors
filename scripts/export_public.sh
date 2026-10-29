#!/usr/bin/env bash
# 做公開版：把程式、輸出、文件、投影片放進孤兒分支 public-export（不含 share/ data/ cache/ 原始資料、不含內部文件），
# 推到私庫後，再用一行推到公開庫：
#   git push https://github.com/shadowcrow0/baseball-decision-warriors.git public-export:main
#   bash scripts/export_public.sh        （可重複跑，每次重建分支）
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
tmp=$(mktemp -d)
git -c core.quotepath=off ls-files -z | while IFS= read -r -d '' f; do
  case "$f" in
    share/*|data/*|cache/*|derived/*|rawdata/*|.claude/*|docs/judges.md|docs/deck_plan.md|README.md|README_PUBLIC.md|*.nc|*.log|*__pycache__*) continue;;
  esac
  mkdir -p "$tmp/$(dirname "$f")"; cp "$f" "$tmp/$f"
done
cp README_PUBLIC.md "$tmp/README.md"
printf 'data/\ncache/\nshare/\n*.nc\n*.log\n__pycache__/\n' > "$tmp/.gitignore"
git branch -D public-export 2>/dev/null || true
git checkout --orphan public-export >/dev/null 2>&1
git rm -rfq --cached . >/dev/null 2>&1 || true
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -a "$tmp"/. .
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-Decision WARriors}" -c user.email="${GIT_AUTHOR_EMAIL:-noreply@example.com}" commit -qm "Public export: code, derived outputs, docs, deck (no raw data)"
git checkout -q -f main
git clean -fdq
rm -rf "$tmp"
echo "分支 public-export 建好：$(git rev-list --count public-export) commit，$(git ls-tree -r public-export --name-only | wc -l) 個檔"
