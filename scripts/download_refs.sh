#!/usr/bin/env bash
# 下载课题 C09 参考文献 PDF（开放获取，全部取自 arXiv）
#
# 清单来源: refs/arxiv_ids.txt（<分组> <arXiv ID> <key> <简述>）
# 输出:     refs/pdf/<key>.pdf
# 幂等:     已存在且为合法 PDF 的文件会跳过；用 -f 可强制重新下载
#
# 用法: bash scripts/download_refs.sh [-f]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDS="$ROOT/refs/arxiv_ids.txt"
PDFDIR="$ROOT/refs/pdf"
mkdir -p "$PDFDIR"

FORCE=0
[[ "${1:-}" == "-f" ]] && FORCE=1

ok=0; fail=0; skip=0
while read -r group id key note; do
  [[ -z "${group:-}" ]] && continue
  [[ "$group" == \#* ]] && continue
  dest="$PDFDIR/$key.pdf"
  if [[ $FORCE -eq 0 && -s "$dest" ]] && head -c 4 "$dest" | grep -q '%PDF'; then
    echo "[skip] $key.pdf"
    skip=$((skip+1)); continue
  fi
  echo "[get ] $id -> $key.pdf"
  if curl -fL --retry 5 --retry-delay 3 --retry-all-errors -C - \
          --connect-timeout 20 --max-time 600 \
          -o "$dest.part" "https://arxiv.org/pdf/$id"; then
    if head -c 4 "$dest.part" | grep -q '%PDF'; then
      mv "$dest.part" "$dest"; ok=$((ok+1))
    else
      echo "[FAIL] $id 返回的不是 PDF"; rm -f "$dest.part"; fail=$((fail+1))
    fi
  else
    echo "[FAIL] $id 下载失败"; rm -f "$dest.part"; fail=$((fail+1))
  fi
done < "$IDS"

echo "[done] 新下载 $ok 篇，跳过 $skip 篇，失败 $fail 篇"
echo "[gen ] refs/pdf/MANIFEST.sha256"
( cd "$ROOT/refs/pdf" && find . -name '*.pdf' | sort | xargs sha256sum ) > "$ROOT/refs/pdf/MANIFEST.sha256"

[[ $fail -eq 0 ]]
