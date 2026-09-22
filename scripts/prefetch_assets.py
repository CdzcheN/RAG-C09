#!/usr/bin/env python3
"""预热下载课题所需模型与数据集（Ubuntu / Windows 通用，替代 shell heredoc）。

用法:
  python scripts/prefetch_assets.py            # 模型 + 数据集全部预热
  python scripts/prefetch_assets.py --models   # 只下模型
  python scripts/prefetch_assets.py --data     # 只下数据集

前置: 建议先设置 HF_ENDPOINT=https://hf-mirror.com（本机 huggingface.co 不可达），
      跨平台设置方式见 项目资料/项目启动与实施指南.md §1.1。
说明: 首次需联网，之后可设置 HF_HUB_OFFLINE=1 离线运行；
      缓存目录由 HF_HOME 控制，不要放进协作仓库。

网络注意（本机实测，2026-09）:
  hf-mirror 对 Xet 存储的大文件会 302 到 cas-bridge.xethub.hf.co，实测仅约 0.2 MB/s
  且频繁 Read timed out。因此本脚本：
    1. 默认禁用 Xet（HF_HUB_DISABLE_XET=1），回退到可续传的经典单流下载；
    2. 放宽单次读超时（HF_HUB_DOWNLOAD_TIMEOUT）；
    3. HF 通道仍失败时，自动改用 ModelScope 镜像（实测约 5 MB/s）补齐同一份 HF 缓存，
       使后续 transformers / sentence-transformers 代码无需任何改动；
    4. 数据集同样适用：HF datasets 的 parquet 也在 Xet 上（实测约 0.13 MB/s），
       回退时优先复用仓库里已有的原始文件（data/raw/hotpotqa/*.parquet 与 HF 声明哈希一致，
       零下载），再退到镜像；补齐后 datasets.load_dataset 仍按原标识加载。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# 必须在导入 huggingface_hub / transformers 之前设置（本模块顶层即生效）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

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
MODELSCOPE_HINT = "https://www.modelscope.cn"
UA = "RAG-C09-prefetch/1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 本地已有的原始数据文件：与 HF 声明的哈希一致时直接复用（硬链接），省去重复下载。
# 键为 HF 仓库标识，值为 {仓库内相对路径: 相对项目根的本地路径}。
LOCAL_DATASET_FILES: dict[str, dict[str, str]] = {
    "hotpotqa/hotpot_qa": {
        "distractor/train-00000-of-00002.parquet": "data/raw/hotpotqa/train-00000-of-00002.parquet",
        "distractor/train-00001-of-00002.parquet": "data/raw/hotpotqa/train-00001-of-00002.parquet",
        "distractor/validation-00000-of-00001.parquet": "data/raw/hotpotqa/validation-00000-of-00001.parquet",
    },
}


def local_sources_for(repo_id: str) -> dict[str, Path]:
    """返回该仓库可复用的本地文件映射（{仓库内路径: 绝对路径}）。"""
    return {rel: PROJECT_ROOT / p for rel, p in LOCAL_DATASET_FILES.get(repo_id, {}).items()}


# 纯元数据文件：缺失不影响加载，取不到时只告警不算失败
OPTIONAL_FILES = {".gitattributes", "README.md"}


def needed_file(path: str, config: str | None) -> bool:
    """数据集仓库常含多个配置（如 hotpot_qa 的 distractor / fullwiki），只取目标配置的文件。"""
    if config is None:
        return True
    return "/" not in path or path.startswith(config + "/")


def warn_if_no_mirror() -> None:
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if MIRROR_HINT not in endpoint:
        print(
            f"[warn] HF_ENDPOINT 未指向镜像（当前 {endpoint or '未设置'}）；"
            f"若外网不可达请先设为 {MIRROR_HINT}（见文档 §1.1 的跨平台对照表）"
        )
    print(f"[info] HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}"
          f"（大文件改走经典下载，避免 Xet 端点超时）")


# --------------------------------------------------------------------------- 下载工具
def http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def curl_download(url: str, dest: Path, retries: int = 5) -> bool:
    """用 curl 下载（支持断点续传）；Windows 10+ 自带 curl.exe。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-L", "-C", "-", "-sS",
        "--retry", str(retries), "--retry-delay", "3", "--retry-all-errors",
        "--connect-timeout", "20", "--max-time", "1800",
        "-o", str(dest), url,
    ]
    try:
        return subprocess.call(cmd) == 0 and dest.exists()
    except FileNotFoundError:
        print("[fail] 未找到 curl；Ubuntu 请安装 curl，Windows 请使用 Win10 及以上", file=sys.stderr)
        return False


def file_digest(path: Path, is_lfs: bool) -> str:
    """LFS 文件按 sha256 校验；普通文件按 git blob sha1 校验（与 HF 的 oid 一致）。"""
    if is_lfs:
        h = hashlib.sha256()
    else:
        h = hashlib.sha1()
        h.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_blob(blob: Path, link: Path) -> None:
    """按 HF 缓存约定建立 snapshots/<commit>/<file> 指向 blobs/<hash> 的链接。

    优先符号链接；Windows 无权限时退回硬链接，再不行则复制。
    """
    link.parent.mkdir(parents=True, exist_ok=True)  # 仓库里可能存在 1_Pooling/ 这类子目录
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(os.path.relpath(blob, link.parent))
        return
    except OSError:
        pass
    try:
        os.link(blob, link)
    except OSError:
        shutil.copyfile(blob, link)


# --------------------------------------------------------------------------- HF 通道失败时的 ModelScope 回退
def hf_cache_dir() -> Path:
    """与 huggingface_hub 一致的缓存根目录推导（不依赖其内部 API）。"""
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    )
    return Path(os.environ.get("HUGGINGFACE_HUB_CACHE") or base / "hub")


def fill_hf_cache(
    repo_id: str,
    endpoint: str,
    repo_type: str = "model",
    local_sources: dict[str, Path] | None = None,
    config: str | None = None,
) -> Path | None:
    """补齐同一个 HF 缓存目录（模型或数据集）；成功返回快照目录，失败返回 None。

    取源顺序：本地已有同名文件（哈希一致，零下载）→ ModelScope（大文件，与 HF 同哈希）
    → hf-mirror（小配置，内容权威）→ ModelScope 宽松兜底。

    步骤：从 HF 元数据接口取 commit 与文件清单（含每个文件的 blob 名与大小）
    → 逐文件按 HF 声明的哈希校验后放进 blobs/ → 建立 snapshots/<commit>/<file> 链接。
    完成后 transformers / datasets 仍按原标识加载，无需任何代码改动。
    """
    endpoint = (endpoint or "https://huggingface.co").rstrip("/")
    api = "models" if repo_type == "model" else "datasets"
    try:
        info = http_json(f"{endpoint}/api/{api}/{repo_id}")
        commit = info.get("sha") or "main"
        tree = http_json(f"{endpoint}/api/{api}/{repo_id}/tree/{commit}?recursive=1")
    except Exception as exc:  # noqa: BLE001
        print(f"[fail] 读取 HF 元数据失败（回退需用它确定 blob 名与哈希）: {exc}", file=sys.stderr)
        return None

    prefix = "models--" if repo_type == "model" else "datasets--"
    cache = hf_cache_dir() / (prefix + repo_id.replace("/", "--"))
    blobs, snapshots = cache / "blobs", cache / "snapshots" / commit
    blobs.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir(parents=True, exist_ok=True)

    files = [f for f in tree if f.get("type") == "file" and f.get("path") and needed_file(f["path"], config)]
    ok = True

    def finalize(blob: Path, link: Path, rel: str) -> bool:
        """清掉 HF 通道残留的 .incomplete，并建立 snapshots 链接。"""
        for stale in blobs.glob(blob.name + ".incomplete*"):
            stale.unlink(missing_ok=True)
        try:
            link_blob(blob, link)
            return True
        except OSError as exc:  # noqa: BLE001 - 单个文件失败不应中断整仓
            print(f"[fail] 建立缓存链接失败 {rel}: {exc}", file=sys.stderr)
            return False

    for meta in files:
        rel, size = meta["path"], meta.get("size") or 0
        lfs = meta.get("lfs") or {}
        want = lfs.get("oid") or meta["oid"]
        blob = blobs / want
        link = snapshots / rel

        # 已存在且哈希正确的 blob 直接复用；哈希复验能发现"内容被镜像改写"的文件
        if blob.exists() and blob.stat().st_size == size and file_digest(blob, bool(lfs)) == want:
            if not finalize(blob, link, rel):
                ok = False
            continue

        # 本地已有同名文件（例如先前下载的原始数据）：哈希一致就直接复用，零下载
        local = (local_sources or {}).get(rel)
        if local and local.exists() and local.stat().st_size == size and file_digest(local, bool(lfs)) == want:
            print(f"[local     ] {repo_id} :: {rel}  ({size / 1e6:.1f} MB)")
            try:
                os.link(local, blob)  # 同分区时硬链接，不占额外空间
            except OSError:
                shutil.copyfile(local, blob)
            if not finalize(blob, link, rel):
                ok = False
            continue

        # 取源策略：大权重（LFS）走 ModelScope（实测与 HF 同哈希）；
        # 小配置文件优先从 HF 取——ModelScope 会改写部分小文本
        # （.gitattributes / tokenizer_config.json 等），直接采用会破坏可复现性。
        ms_kind = "models" if repo_type == "model" else "datasets"
        ms_url = f"{MODELSCOPE_HINT}/{ms_kind}/{repo_id}/resolve/master/{rel}"
        hf_url = f"{endpoint}/{repo_id}/resolve/{commit}/{rel}"
        sources = [(ms_url, "modelscope")] if lfs else [(hf_url, "hf-mirror"), (ms_url, "ms-lenient")]

        part = blobs / (blob.name + ".part")
        fetched = False
        for url, tag in sources:
            # 本地可能已有完整残片（例如先前用 curl 手工下好）：直接进入校验，避免 416
            if not (part.exists() and part.stat().st_size == size):
                print(f"[{tag:<10}] {repo_id} :: {rel}  ({size / 1e6:.1f} MB)")
                if not curl_download(url, part):
                    print(f"[warn] {tag} 下载失败: {rel}", file=sys.stderr)
                    part.unlink(missing_ok=True)
                    continue
            if part.stat().st_size != size:
                print(f"[warn] {tag} 大小不符: {rel}", file=sys.stderr)
                part.unlink(missing_ok=True)
                continue
            if file_digest(part, bool(lfs)) != want:
                if tag == "ms-lenient":
                    print(f"[warn] {rel}: 镜像内容与 HF 不一致，仅按大小采用（{tag}）", file=sys.stderr)
                else:
                    print(f"[warn] {tag} 哈希校验不通过: {rel}", file=sys.stderr)
                    part.unlink(missing_ok=True)
                    continue
            part.replace(blob)
            fetched = True
            break
        if not fetched:
            if rel in OPTIONAL_FILES:
                print(f"[warn] 可选元数据文件未取到，忽略: {rel}", file=sys.stderr)
                continue
            print(f"[fail] 两个源均无法取得该文件: {rel}", file=sys.stderr)
            ok = False
            continue
            continue

        if not finalize(blob, link, rel):
            ok = False

    if not ok:
        return None
    (cache / "refs").mkdir(parents=True, exist_ok=True)
    (cache / "refs" / "main").write_text(commit, encoding="utf-8")
    return snapshots


# --------------------------------------------------------------------------- 预热入口
def prefetch_models() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[fail] 缺少 huggingface_hub，请先执行 python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    rc = 0
    for repo in MODELS:
        print(f"[get ] 模型 {repo}")
        try:
            print(f"[ok  ] 缓存于 {snapshot_download(repo)}")
            continue
        except Exception as exc:  # noqa: BLE001 - 网络类异常都要回退
            print(f"[warn] HF 通道失败：{type(exc).__name__}: {str(exc)[:160]}")
        snap = fill_hf_cache(repo, endpoint, "model")
        if snap:
            print(f"[ok  ] 已由 ModelScope 补齐 HF 缓存：{snap}")
        else:
            print(f"[fail] {repo}：HF 与 ModelScope 两条通道均未成功", file=sys.stderr)
            rc = 1
    return rc


def prefetch_datasets() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[fail] 缺少 datasets，请先执行 python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    rc = 0
    for name, config in DATASETS:
        label = name if config is None else f"{name} ({config})"
        print(f"[get ] 数据集 {label}")

        # 本机已备有原始文件时优先直接补齐缓存：HF datasets 的大 parquet 也在 Xet 上，
        # 本机实测会长时间停在 0 字节，而本地文件与 HF 声明哈希一致（可零下载复用）。
        local = local_sources_for(name)
        if local and all(p.exists() for p in local.values()):
            print(f"[info] 检测到本地原始文件，直接补齐缓存：{', '.join(sorted(local))}")
            if snap := fill_hf_cache(
                name, endpoint, "dataset", local_sources=local, config=config
            ):
                ds = load_dataset(name, config) if config else load_dataset(name)
                splits = ", ".join(f"{k}={len(v)}" for k, v in ds.items())
                print(f"[ok  ] {label}（本地文件补齐缓存）: {splits}")
                continue
            print("[warn] 本地补齐失败，改用 HF 通道")

        try:
            ds = load_dataset(name, config) if config else load_dataset(name)
            splits = ", ".join(f"{k}={len(v)}" for k, v in ds.items())
            print(f"[ok  ] {label}: {splits}")
            continue
        except Exception as exc:  # noqa: BLE001 - 网络类异常都要回退
            print(f"[warn] HF 通道失败：{type(exc).__name__}: {str(exc)[:160]}")

        # HF 通道拿不到（本机大文件长期停在 0 字节）：用「本地已有文件 → ModelScope → hf-mirror」补齐缓存
        snap = fill_hf_cache(name, endpoint, "dataset", local_sources=local_sources_for(name))
        if not snap:
            print(f"[fail] {label}：HF 与本地/镜像回退均未成功", file=sys.stderr)
            rc = 1
            continue
        ds = load_dataset(name, config) if config else load_dataset(name)
        splits = ", ".join(f"{k}={len(v)}" for k, v in ds.items())
        print(f"[ok  ] {label}（回退补齐缓存后加载）: {splits}")
    return rc


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
