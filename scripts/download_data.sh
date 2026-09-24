#!/usr/bin/env bash
# 课题 C09 数据集下载脚本（幂等，可重复运行；已存在且非空的文件会跳过）
#
# 数据来源
#   SQuAD v2.0        : 官方发布页 https://rajpurkar.github.io/SQuAD-explorer/
#   HotpotQA distractor: HuggingFace 数据集 hotpotqa/hotpot_qa（经 hf-mirror.com 镜像）
#                       原始发布页 https://hotpotqa.github.io/ （本机网络不可达，故走镜像）
#
# 用法: bash scripts/download_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SQUAD_DIR="$ROOT/data/raw/squad"
HOTPOT_DIR="$ROOT/data/raw/hotpotqa"
mkdir -p "$SQUAD_DIR" "$HOTPOT_DIR"

# 部分站点仅 IPv4 可达 / 镜像偶发断连，故统一重试 + 断点续传
curl_get() { # <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "[skip] $(basename "$dest") 已存在（$(du -h "$dest" | cut -f1)）"
    return 0
  fi
  echo "[get ] $url"
  curl -fL --retry 5 --retry-delay 3 --retry-all-errors \
       -C - --connect-timeout 20 --max-time 3600 \
       -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

SQUAD_BASE="https://rajpurkar.github.io/SQuAD-explorer/dataset"
HF_BASE="https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor"

# SQuAD v2.0：train 130319 问 / dev 11873 问（含不可回答样本）
curl_get "$SQUAD_BASE/train-v2.0.json" "$SQUAD_DIR/train-v2.0.json"
curl_get "$SQUAD_BASE/dev-v2.0.json"   "$SQUAD_DIR/dev-v2.0.json"

# HotpotQA distractor：train 90447 例（2 个 parquet 分片）/ validation 7405 例
curl_get "$HF_BASE/train-00000-of-00002.parquet"      "$HOTPOT_DIR/train-00000-of-00002.parquet"
curl_get "$HF_BASE/train-00001-of-00002.parquet"      "$HOTPOT_DIR/train-00001-of-00002.parquet"
curl_get "$HF_BASE/validation-00000-of-00001.parquet" "$HOTPOT_DIR/validation-00000-of-00001.parquet"

echo "[gen ] data/raw/MANIFEST.sha256"
( cd "$ROOT/data/raw" && find . -type f \( -name '*.json' -o -name '*.parquet' \) | sort | xargs sha256sum ) \
  > "$ROOT/data/raw/MANIFEST.sha256"

echo "[done] 数据集下载完成，校验清单见 data/raw/MANIFEST.sha256"
