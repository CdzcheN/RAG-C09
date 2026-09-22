#!/usr/bin/env python3
"""课题 C09 环境自检 —— 校验当前 conda 环境是否满足 requirements.txt（Ubuntu / Windows 通用）。

检查内容:
  1. 是否处于 conda 环境（sys.prefix 下存在 conda-meta/），并报告环境名与平台；
  2. Python 版本是否为 3.10.x；
  3. 逐条校验 requirements.txt：`==` 的锚点包严格比对公共版本号（`2.10.0+cu128` 视同 `2.10.0`），
     `>=` 的包检查已安装且不低于下限；
  4. PyTorch 的 CUDA 可用性、GPU 名称、可用显存、cuDNN 版本与 bf16 支持；
  5. 模型与数据集下载镜像 HF_ENDPOINT（huggingface.co 在部分网络不可达，须走 hf-mirror.com）。

用法:
  python scripts/check_env.py                             # 打印检查表，退出码 0/1
  python scripts/check_env.py --report results/env_report.json
  python scripts/check_env.py --allow-missing             # 装依赖之前：缺包仅告警，退出码仍为 0

退出码: 0 全部通过；1 存在失败项（非 conda 环境 / Python 版本不符 / 锚点版本不符 / CUDA 不可用 / 依赖缺失）
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from importlib import metadata, util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
PYTHON_SERIES = "3.10"
EXPECTED_ENV_NAME = "rag-c09"
MIN_VRAM_GB = 3.0
HF_MIRROR = "hf-mirror.com"

# 发行包名 -> import 名（不一致时）
IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
}

try:
    from packaging.specifiers import SpecifierSet
except ImportError:  # 无 packaging 时退化为字符串比较
    SpecifierSet = None


class Check:
    """收集检查结果，负责打印与落盘。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.fatal = False
        self.warn = False

    def add(self, status: str, item: str, detail: str, fatal: bool = False) -> None:
        self.rows.append((status, item, detail))
        if status == "FAIL" and fatal:
            self.fatal = True
        if status == "WARN":
            self.warn = True

    def show(self) -> None:
        print("== 课题 C09 环境自检 ==")
        width = max(len(item) for _, item, _ in self.rows)
        for status, item, detail in self.rows:
            print(f"[{status:^4}] {item:<{width}}  {detail}")


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    """解析 requirements.txt，返回 [(包名, 约束串)]。

    约束串按 pip 原样保留，支持 `>=1.5,<2` 这类组合约束（早期版本只认 `==`/`>=`，
    会把 `5.17,<6` 整体当成版本号，导致合法版本被误判为不满足）。
    """
    out: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # 形如：包名 / 包名==1.2.3 / 包名>=1.0,<2 / 包名[extra]>=1.0；分号后的环境标记忽略
        m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*([<>=!~][^;]*)?$", line)
        if m and m.group(2):
            out.append((m.group(1), m.group(2).strip()))
    return out


def version_ok(got: str, spec: str) -> bool:
    """按 PEP 440 判断已安装版本是否满足约束串。

    说明：`==2.10.0` 能匹配 `2.10.0+cu128`（PEP 440 规定未写 local 段时忽略 local），
    因此 torch 的 `+cu128` 本地版本号不会造成误判。
    """
    if SpecifierSet is None:  # 退化路径
        return any(got.split("+")[0].startswith(p) for p in re.findall(r"\d+(?:\.\d+)*", spec))
    try:
        return SpecifierSet(spec).contains(got, prereleases=True)
    except Exception:
        return False


def check_environment(c: Check) -> None:
    prefix = Path(sys.prefix)
    if (prefix / "conda-meta").is_dir():
        if prefix.name == EXPECTED_ENV_NAME:
            c.add("OK", "环境归属", f"conda 环境 {prefix.name}：{prefix}")
        else:
            c.add("WARN", "环境归属", f"当前 conda 环境为 {prefix.name}（团队约定名 {EXPECTED_ENV_NAME}）")
    else:
        c.add(
            "FAIL",
            "环境归属",
            f"sys.prefix={prefix} 不是 conda 环境；请先 conda activate {EXPECTED_ENV_NAME}",
            fatal=True,
        )
    c.add("OK", "平台", f"{platform.system()} {platform.release()} | Python {platform.python_version()}")
    if platform.python_version().startswith(PYTHON_SERIES):
        c.add("OK", "Python", f"{platform.python_version()}（要求 {PYTHON_SERIES}.x）")
    else:
        c.add("FAIL", "Python", f"{platform.python_version()} ≠ {PYTHON_SERIES}.x", fatal=True)


def check_requirements(c: Check, allow_missing: bool) -> None:
    if not REQUIREMENTS.exists():
        c.add("FAIL", "requirements.txt", f"未找到 {REQUIREMENTS}", fatal=True)
        return
    specs = parse_requirements(REQUIREMENTS)
    if not specs:
        c.add("FAIL", "requirements.txt", "未解析到任何依赖条目", fatal=True)
        return
    c.add("OK", "requirements.txt", f"解析到 {len(specs)} 条依赖")

    missing: list[str] = []
    for pkg, spec in specs:
        mod = IMPORT_NAME.get(pkg, pkg.replace("-", "_"))
        if util.find_spec(mod) is None:
            missing.append(pkg)
            continue
        try:
            got = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            got = getattr(__import__(mod), "__version__", "unknown")
        if version_ok(got, spec):
            c.add("OK", f"依赖 {pkg}", f"{got}（要求 {spec}）")
        else:
            c.add("FAIL", f"依赖 {pkg}", f"{got} 不满足 {spec}", fatal=True)

    if missing:
        c.add(
            "WARN" if allow_missing else "FAIL",
            "缺失依赖",
            f"{len(missing)} 个未安装：{', '.join(missing)} → python -m pip install -r requirements.txt",
            fatal=not allow_missing,
        )


def check_cuda(c: Check) -> dict:
    info: dict = {}
    try:
        import torch
    except Exception as exc:
        c.add("FAIL", "CUDA", f"无法导入 torch：{exc!r}", fatal=True)
        return info

    info["torch"] = torch.__version__
    info["cuda_compiled"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    if not torch.cuda.is_available():
        c.add(
            "FAIL",
            "CUDA 可用",
            "torch.cuda.is_available() == False（检查 NVIDIA 驱动，或确认装的是 CUDA 版 torch）",
            fatal=True,
        )
        return info

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024 ** 3
    info.update(
        {
            "cuda_available": True,
            "gpu": props.name,
            "vram_gb": round(vram_gb, 2),
            "capability": list(torch.cuda.get_device_capability(0)),
            "bf16": bool(torch.cuda.is_bf16_supported()),
        }
    )
    c.add("OK", "CUDA 可用", f"CUDA {info['cuda_compiled']} | cuDNN {info['cudnn']}")
    c.add("OK", "GPU", f"{props.name} | 可用显存 {vram_gb:.2f} GB | CC {info['capability']}")
    if vram_gb < MIN_VRAM_GB:
        c.add("WARN", "显存", f"{vram_gb:.2f} GB 低于建议值 {MIN_VRAM_GB} GB（按 0.5B fp16 + 贪心解码选型）")
    c.add("OK" if info["bf16"] else "WARN", "bf16", str(info["bf16"]))
    return info


def check_hf_env(c: Check) -> None:
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if HF_MIRROR in endpoint:
        c.add("OK", "HF_ENDPOINT", endpoint)
    elif endpoint:
        c.add("WARN", "HF_ENDPOINT", f"{endpoint}（建议 https://{HF_MIRROR}）")
    else:
        c.add("WARN", "HF_ENDPOINT", f"未设置 → Ubuntu: export HF_ENDPOINT=https://{HF_MIRROR}；"
                                     f"Windows: $env:HF_ENDPOINT=\"https://{HF_MIRROR}\"")


def main() -> int:
    ap = argparse.ArgumentParser(description="课题 C09 环境自检（Ubuntu / Windows 通用）")
    ap.add_argument("--report", type=Path, default=None, help="将环境指纹写入该 JSON")
    ap.add_argument("--allow-missing", action="store_true", help="缺包仅告警（装依赖之前用）")
    args = ap.parse_args()

    c = Check()
    check_environment(c)
    check_requirements(c, args.allow_missing)
    cuda_info = check_cuda(c)
    check_hf_env(c)
    c.show()

    if args.report:
        report = {
            "sys_prefix": sys.prefix,
            "env_name": Path(sys.prefix).name,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "checks": [{"status": s, "item": i, "detail": d} for s, i, d in c.rows],
            **cuda_info,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            shown = args.report.resolve().relative_to(ROOT)
        except ValueError:
            shown = args.report
        print(f"\n环境指纹已写入 {shown}")

    if c.fatal:
        print("\n结果：存在失败项（见上方 FAIL 行）")
        return 1
    print("\n结果：通过" + ("（有告警，见 WARN 行）" if c.warn else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
