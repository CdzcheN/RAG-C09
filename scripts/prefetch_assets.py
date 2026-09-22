#!/usr/bin/env python3
"""预热下载课题所需模型与数据集（Ubuntu / Windows 通用，替代 shell heredoc）。

用法:
  python scripts/prefetch_assets.py            # 模型 + 数据集全部预热
  python scripts/prefetch_assets.py --models   # 只下模型
  python scripts/prefetch_assets.py --data     # 只下数据集

前置: 建议先设置 HF_ENDPOINT=https://hf-mirror.com（本机 huggingface.co 不可达），
      跨平台设置方式见 docs/项目启动与实施指南.md §1.1。
说明: 首次需联网（经镜像），之后可设置 HF_HUB_OFFLINE=1 离线运行；
      缓存目录由 HF_HOME 控制，不要放进协作仓库。
"""
from __future__ import annotations

import argparse
import os
import sys

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",              # 生成模型（主）
    "sentence-transformers/all-MiniLM-L6-v2",  # 稠密检索（可选支线）
    "cross-encoder/nli-deberta-v3-xsmall",     # NLI 蕴含特征判别器
]

# (数据集标识, 配置名)
DATASETS = [
    ("rajpurkar/squad_v2", None),
    ("hotpotqa/hotpot_qa", "distractor"),
]

MIRROR_HINT = "https://hf-mirror.com"


def warn_if_no_mirror() -> None:
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if MIRROR_HINT not in endpoint:
        print(
            f"[warn] HF_ENDPOINT 未指向镜像（当前 {endpoint or '未设置'}）；"
            f"若外网不可达请先设为 {MIRROR_HINT}（见文档 §1.1 的跨平台对照表）"
        )


def prefetch_models() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[fail] 缺少 huggingface_hub，请先执行 python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    for repo in MODELS:
        print(f"[get ] 模型 {repo}")
        print(f"[ok  ] 缓存于 {snapshot_download(repo)}")
    return 0


def prefetch_datasets() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[fail] 缺少 datasets，请先执行 python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    for name, config in DATASETS:
        label = name if config is None else f"{name} ({config})"
        print(f"[get ] 数据集 {label}")
        ds = load_dataset(name, config) if config else load_dataset(name)
        splits = ", ".join(f"{k}={len(v)}" for k, v in ds.items())
        print(f"[ok  ] {label}: {splits}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="预热下载课题 C09 所需模型与数据集")
    ap.add_argument("--models", action="store_true", help="只下载模型")
    ap.add_argument("--data", action="store_true", help="只下载数据集")
    args = ap.parse_args()

    warn_if_no_mirror()
    both = not (args.models or args.data)
    rc = 0
    if args.models or both:
        rc |= prefetch_models()
    if args.data or both:
        rc |= prefetch_datasets()
    print("\n[done] 预热结束；之后可设置 HF_HUB_OFFLINE=1 离线运行")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
