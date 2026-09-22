#!/usr/bin/env python3
"""冒烟测试：结构 + 契约 + 公共层行为；数据/模型未就绪的环节标记 SKIP。

用法:
  python scripts/smoke_test.py                 # 结构与契约自检 + 已存在产物的校验
  python scripts/smoke_test.py --with-tests    # 额外运行 tests/ 下的单元测试

退出码: 0 通过（允许 SKIP）；1 结构/契约失败；3 产物校验不通过
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import io, validate  # noqa: E402
from src.common.schema import FEATURE_COLUMNS, Prediction, RAGSample  # noqa: E402

REQUIRED_PATHS = [
    "README.md", "requirements.txt",
    "docs/开发文档索引.md", "docs/接口契约.md", "docs/数据构造规范.md", "docs/实验与评估规范.md",
    "configs/default.yaml", "configs/models.yaml", "configs/challenge.yaml", "configs/detect.yaml",
    "src/common/schema.py", "src/common/seeding.py", "src/common/io.py", "src/common/validate.py",
    "src/datasets/challenge_builder.py", "src/retrieval/bm25.py", "src/generation/pipeline.py",
    "src/features/overlap.py", "src/detection/train.py", "src/evaluation/metrics.py",
]
ARTIFACTS = [
    ("samples", "data/processed/challenge_set.jsonl"),
    ("features", "results/features/w4-consistency-logreg-13.parquet"),
]

ok = skip = 0
failures: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    global ok, skip
    if status == "OK":
        ok += 1
    elif status == "SKIP":
        skip += 1
    else:
        failures.append(f"{name}: {detail}")
    print(f"[{status:^4}] {name}" + (f"  {detail}" if detail else ""))


def check_structure() -> None:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    if missing:
        report("FAIL", "目录与文件结构", f"缺少 {missing}")
    else:
        report("OK", "目录与文件结构", f"{len(REQUIRED_PATHS)} 个关键路径齐备")


def check_configs() -> None:
    try:
        cfgs = {n: io.load_config(ROOT / "configs" / f"{n}.yaml")
                for n in ("default", "models", "challenge", "detect")}
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "配置解析", repr(exc))
        return
    hashes = {io.config_hash(c) for c in cfgs.values()}
    if len(hashes) != 4 or cfgs["default"]["generation"]["do_sample"] is not False:
        report("FAIL", "配置解析", "配置缺失或 do_sample 未固定为 False")
    else:
        report("OK", "配置解析", "四份配置可解析、哈希互异、解码为贪心")


def check_contract() -> None:
    try:
        s = RAGSample(sample_id="s-1", source="squad", split="normal", challenge_type="none",
                      question="q", gold_answer="a", gold_context=("e",))
        assert RAGSample.from_dict(s.to_dict()).to_dict() == s.to_dict()
        p = Prediction(sample_id="s-1", exp_id="smoke-13", answer="a", baseline_confidence=0.5)
        assert Prediction.from_dict(p.to_dict()).to_dict() == p.to_dict()
        assert len(FEATURE_COLUMNS) == 14
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "契约往返", repr(exc))
        return
    report("OK", "契约往返", "RAGSample/Prediction 往返一致，特征列 14 个")


def check_artifacts() -> None:
    any_checked = False
    for kind, rel in ARTIFACTS:
        path = ROOT / rel
        if not path.exists():
            continue
        any_checked = True
        problems = validate.validate_file(kind, path)
        if problems:
            report("FAIL", f"产物校验 {rel}", f"{len(problems)} 条问题，例：{problems[0]}")
        else:
            report("OK", f"产物校验 {rel}", "通过契约校验")
    if not any_checked:
        report("SKIP", "产物校验", "尚未生成 challenge_set / features 产物（W3、W4 阶段产出）")


def check_pipeline_ready() -> None:
    """检查管道前置条件，但不实际下载模型（避免冒烟拖成长任务）。"""
    try:
        import datasets  # noqa: F401
    except ImportError:
        report("SKIP", "管道前置（datasets）", "未安装 datasets；建环境后重跑本项")
        return
    try:
        import torch
    except ImportError:
        report("SKIP", "管道前置（torch）", "未安装 torch；建环境后重跑本项")
        return
    if torch.cuda.is_available():
        report("OK", "管道前置", f"torch {torch.__version__} + CUDA 可用，可跑端到端冒烟")
    else:
        report("SKIP", "管道前置", "CUDA 不可用；仍可在 CPU 上冒烟，但需调整 batch/长度")


def run_unit_tests() -> None:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if result.wasSuccessful():
        report("OK", "单元测试", f"{result.testsRun} 个用例通过")
    else:
        report("FAIL", "单元测试", f"失败 {len(result.failures)}、错误 {len(result.errors)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="课题 C09 冒烟测试")
    ap.add_argument("--with-tests", action="store_true", help="同时运行 tests/ 下的单元测试")
    args = ap.parse_args()

    print("== 课题 C09 冒烟测试 ==")
    check_structure()
    check_configs()
    check_contract()
    check_artifacts()
    check_pipeline_ready()
    if args.with_tests:
        run_unit_tests()

    print(f"\n结果：OK {ok}，SKIP {skip}，FAIL {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
