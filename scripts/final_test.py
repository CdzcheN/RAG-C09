#!/usr/bin/env python3
"""课题 C09 最终验收测试：L0 静态自检 → L1 单元测试 → L2 数据链路 → L3 生成管道 → L4 判别与评估。

定位: `scripts/smoke_test.py` 是**开发期的快速自检**（结构/配置/契约/产物），本脚本是**交付前的
端到端验收**：把 A（数据）→ B（生成与特征）→ C（判别与评估）三段真实串起来跑，并保留每段产物的
契约校验结果，供 G6「可追溯性」清单核对。

用法:
  python scripts/final_test.py                       # 默认全跑 L0–L4（L3/L4 需模型与依赖，首次会下载权重）
  python scripts/final_test.py --fast                # 只跑 L0–L2（秒级，无需模型）
  python scripts/final_test.py --level 3             # 跑到指定层级
  python scripts/final_test.py --limit 2             # 覆盖 L3/L4 的样本条数（默认 2）
  python scripts/final_test.py --artifacts-dir /tmp/c09-final   # 改产物目录（默认保留在仓库内）
  python scripts/final_test.py --figures-dir results/figures    # 把验证图也放到真实结果图目录

约定:
- 重依赖（torch/transformers/rank_bm25/pyarrow/sklearn…）缺失时对应检查记 SKIP 并说明安装方式，
  不计为失败（交付到没有 GPU 的机器上也能给出结构 + 数据链路的结论）；
- L3 的 G1 检查口径（见 `g1_compare`）：`sample_id` / `answer` / `prompt` / `passages` / `decode`
  必须逐字节一致；`baseline_confidence` 允许浮点尾差（容差 1e-6，报告最大绝对差）；`latency_ms`
  是计时噪声、`_meta.timestamp` 是时间戳，均不参与比较（把计时噪声算进"逐字节"会让门禁永远无法通过）；
- L4 在 L3 未产出真实特征表时使用**合成特征**验证 C 批链路，报告中会标注来源，避免"看起来通过"；
- 每次运行前清理上一次遗留的中间产物（`results/final_test/l*`）与本脚本上次产出的图
  （`<figures_dir>/final-*.png`），避免读到 stale 结果而误报 OK、也避免陈旧图累积；
  产物**默认保留**且路径固定，便于人工核对：样本/预测/特征/指标/汇总在 `results/final_test/`
  （`--artifacts-dir` 可改），图表在 `results/final_test/figures/`（`--figures-dir` 可改）。
  注意 L4 在缺少真实特征表时用**合成特征**跑链路，因此这些图是链路验证图，默认不与契约约定的
  真实结果图目录 `results/figures/` 混淆；需要时可以 `--figures-dir results/figures` 覆盖。
  集成测试（`tests/test_end_to_end.py`）的四张链路验证图单独放在
  `results/final_test/test_figures/`，与本脚本的六张必备图分目录，避免两类图混在一起。
  脚本只清理本次新增的 `results/logs/*.log`。

退出码: 0 全通过（允许 SKIP）/ 1 存在失败 / 3 产物契约校验不通过
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import importlib
import importlib.util
import io as _io
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = sys.executable
LEVELS = ("L0", "L1", "L2", "L3", "L4")
LEVEL_DESC = {
    "L0": "静态自检：编译/路径/配置/契约/无骨架残留",
    "L1": "单元测试：tests/ 全部用例",
    "L2": "数据链路：读取→算子→标签→划分→特征表(无 NLI)→契约校验",
    "L3": "生成管道：构造集→预测→特征表→G1 逐字节一致性",
    "L4": "判别与评估：判别器→分组指标→配对检验→汇总表与图表",
}

REQUIRED_PATHS = [
    "README.md", "requirements.txt",
    "configs/default.yaml", "configs/models.yaml", "configs/challenge.yaml", "configs/detect.yaml",
    "src/common/schema.py", "src/common/io.py", "src/common/validate.py", "src/common/text.py",
    "src/datasets/reader.py", "src/datasets/challenge_builder.py", "src/datasets/split.py",
    "src/retrieval/index.py", "src/retrieval/bm25.py", "src/retrieval/evaluate.py",
    "src/generation/model.py", "src/generation/decode.py", "src/generation/pipeline.py",
    "src/features/entailment.py", "src/features/consistency.py", "src/features/build_features.py",
    "src/detection/model.py", "src/detection/train.py", "src/detection/predict.py",
    "src/evaluation/metrics.py", "src/evaluation/grouped.py", "src/evaluation/significance.py",
    "src/evaluation/plots.py", "src/demo/app.py", "scripts/smoke_test.py", "tests/test_contract.py",
]
BUSINESS_MODULES = [
    "src.datasets.reader", "src.datasets.challenge_builder", "src.datasets.split",
    "src.datasets.operators", "src.datasets.slots", "src.datasets.construct_log",
    "src.retrieval.index", "src.retrieval.bm25", "src.retrieval.dense", "src.retrieval.evaluate",
    "src.generation.model", "src.generation.decode", "src.generation.pipeline",
    "src.generation.baseline_confidence", "src.features.entailment", "src.features.consistency",
    "src.features.build_features", "src.features.overlap", "src.detection.model",
    "src.detection.train", "src.detection.calibrate", "src.detection.predict",
    "src.evaluation.metrics", "src.evaluation.grouped", "src.evaluation.significance",
    "src.evaluation.tables", "src.evaluation.plots", "src.demo.app",
]
HEAVY_DEPS = ["torch", "transformers", "datasets", "rank_bm25", "pyarrow", "sklearn", "scipy",
              "matplotlib", "numpy", "pandas"]
OPTIONAL_DEP_HINTS = {
    "torch": "python -m pip install -r requirements.txt（或按指南 §1.1 指定 CUDA 索引）",
    "rank_bm25": "python -m pip install rank_bm25",
    "pyarrow": "python -m pip install pyarrow",
    "sklearn": "python -m pip install scikit-learn",
    "scipy": "python -m pip install scipy",
    "matplotlib": "python -m pip install matplotlib",
}


class Report:
    """逐项记录 OK / SKIP / FAIL，并统计产物契约校验问题数（决定退出码 3）。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.contract_failures = 0

    def add(self, status: str, level: str, name: str, detail: str = "") -> None:
        self.rows.append((status, level, name, detail))
        if status == "FAIL" and ("契约" in name or "校验" in name):
            self.contract_failures += 1
        print(f"[{status:^4}] {level}  {name}" + (f"  {detail}" if detail else ""))

    def section(self, level: str) -> None:
        print(f"\n=== {level} · {LEVEL_DESC[level]} ===")

    def exit_code(self) -> int:
        if any(status == "FAIL" for status, _, _, _ in self.rows):
            return 3 if self.contract_failures else 1
        return 0

    def totals(self) -> dict[str, int]:
        counts = {"OK": 0, "SKIP": 0, "FAIL": 0}
        for status, _, _, _ in self.rows:
            counts[status] += 1
        return counts


class Ctx:
    """运行上下文：产物目录、依赖可用性、需清理的日志文件。"""

    def __init__(self, artifacts_dir: pathlib.Path, figures_dir: pathlib.Path, limit: int) -> None:
        self.limit = limit
        self.artifacts_dir = artifacts_dir
        self.figures_dir = figures_dir
        self.available = {name: importlib.util.find_spec(name) is not None for name in HEAVY_DEPS}
        self.log_dir = ROOT / "results" / "logs"
        self.logs_before = set(self.log_dir.glob("*.log")) if self.log_dir.exists() else set()
        self.samples_path: pathlib.Path | None = None      # L3 用的样本文件
        self.features_path: pathlib.Path | None = None     # L4 用的特征表
        self.features_source = ""                          # 特征表来源（真实/合成）
        self.dataset_origin = ""
        self.extra_checks: list[tuple[str, bool]] = []      # 数据链路派生的小结论

    def has(self, *names: str) -> bool:
        return all(self.available.get(name, False) for name in names)

    def skip_reason(self, *names: str) -> str:
        missing = [n for n in names if not self.available.get(n, False)]
        hints = {OPTIONAL_DEP_HINTS[n] for n in missing if n in OPTIONAL_DEP_HINTS}
        return f"缺少 {missing}；安装：{'；'.join(sorted(hints)) if hints else 'python -m pip install -r requirements.txt'}"

    def subdir(self, name: str) -> pathlib.Path:
        path = self.artifacts_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup(self) -> None:
        """只清理本次运行新增的日志；产物目录默认保留，便于人工核对。"""
        for path in sorted(set(self.log_dir.glob("*.log")) - self.logs_before):
            try:
                path.unlink()
            except OSError:
                pass


def run_cli(module: str, args: list[str], timeout: int = 3600) -> tuple[int, str]:
    """跑一个 CLI 子进程，返回 (退出码, 末尾输出)。"""
    proc = subprocess.run([PY, "-m", module, *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, check=False)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-3:])
    return proc.returncode, tail


def sha256_text(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_lines(path: pathlib.Path) -> list[str]:
    """JSONL 的数据行（排除 `_meta`，因其中含 timestamp，无法逐字节稳定比较）。"""
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and '"_meta"' not in line]


# ———————————————————————— L0 静态自检 ————————————————————————

def check_compile(_ctx: Ctx) -> tuple[str, str]:
    ok = compileall.compile_dir(str(ROOT / "src"), quiet=2) and compileall.compile_dir(str(ROOT / "scripts"), quiet=2)
    return ("OK", "src 与 scripts 全部编译通过") if ok else ("FAIL", "编译失败：见上方输出")


def check_paths(_ctx: Ctx) -> tuple[str, str]:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    return ("FAIL", f"缺少 {missing}") if missing else ("OK", f"{len(REQUIRED_PATHS)} 个关键路径齐备")


def check_configs(_ctx: Ctx) -> tuple[str, str]:
    from src.common import io

    cfgs = {name: io.load_config(ROOT / "configs" / f"{name}.yaml")
            for name in ("default", "models", "challenge", "detect")}
    hashes = {io.config_hash(cfg) for cfg in cfgs.values()}
    greedy = cfgs["default"]["generation"]["do_sample"] is False
    if len(hashes) != 4 or not greedy:
        return "FAIL", "四份配置哈希未互异或 do_sample 未固定为 False"
    return "OK", "四份配置可解析、哈希互异、解码为贪心"


def check_contract(_ctx: Ctx) -> tuple[str, str]:
    from src.common.schema import FEATURE_COLUMNS, METHOD_FEATURE_GROUPS, Prediction, RAGSample, RetrievedPassage

    sample = RAGSample(sample_id="s-1", source="squad", split="normal", challenge_type="none",
                       question="q", gold_answer="a", gold_context=("e",))
    prediction = Prediction(sample_id="s-1", exp_id="final-13", answer="a",
                            passages=(RetrievedPassage(rank=1, title="t", text="x", score=1.0),))
    flat = {c for cols in METHOD_FEATURE_GROUPS.values() for c in cols}
    problems = []
    if RAGSample.from_dict(sample.to_dict()).to_dict() != sample.to_dict():
        problems.append("RAGSample 往返不一致")
    if Prediction.from_dict(prediction.to_dict()).to_dict() != prediction.to_dict():
        problems.append("Prediction 往返不一致")
    if len(FEATURE_COLUMNS) != 14:
        problems.append(f"特征列 {len(FEATURE_COLUMNS)} 个（应为 14）")
    if "baseline_confidence" in flat:
        problems.append("baseline_confidence 混入了主方法特征组")
    return ("FAIL", "；".join(problems)) if problems else ("OK", "schema 往返一致、14 列、baseline 不入主方法组")


def check_imports(_ctx: Ctx) -> tuple[str, str]:
    broken = []
    for name in BUSINESS_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{name}: {exc!r}")
    return ("FAIL", "；".join(broken[:3])) if broken else ("OK", f"{len(BUSINESS_MODULES)} 个业务模块可导入")


#: 允许保留的占位：`cli_utils.todo()` 是公共层给未来模块用的统一"未实现"出口，无调用者
ALLOWED_PLACEHOLDER_FILES = {pathlib.Path("src/common/cli_utils.py")}


def check_no_skeleton(_ctx: Ctx) -> tuple[str, str]:
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative in ALLOWED_PLACEHOLDER_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("状态: 骨架", "状态: 待实现", "raise NotImplementedError"):
            if marker in text:
                offenders.append(f"{relative} 含 {marker}")
    return ("FAIL", "；".join(offenders[:3])) if offenders else ("OK", "无占位实现残留（骨架标记/NotImplementedError）")


def check_cli_help(_ctx: Ctx) -> tuple[str, str]:
    modules = ["src.datasets.cli", "src.retrieval.cli", "src.generation.cli", "src.features.cli",
               "src.detection.cli", "src.evaluation.cli", "src.demo.app"]
    broken = [m for m in modules if run_cli(m, ["--help"])[0] != 0]
    return ("FAIL", f"--help 失败：{broken}") if broken else ("OK", f"{len(modules)} 个 CLI 入口 --help 正常")


# ———————————————————————— L1 单元测试 ————————————————————————

def check_unit_tests(_ctx: Ctx) -> tuple[str, str]:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    stream = _io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    skipped = len(result.skipped)
    if result.wasSuccessful():
        return "OK", f"{result.testsRun} 个用例通过（跳过 {skipped} 个，缺依赖时正常）"
    detail = f"失败 {len(result.failures)}、错误 {len(result.errors)}"
    print(stream.getvalue()[-2000:])
    return "FAIL", detail


# ———————————————————————— L2 数据链路 ————————————————————————

def _synthetic_example(index: int) -> dict:
    """HotpotQA 原始字段格式（{title, sentences} 双列表），用于顺带验证 normalize_hotpotqa 的转换。"""
    titles = [f"Topic{index}Part{j}" for j in range(6)]
    return {
        "id": f"row{index}",
        "question": f"In which year was {titles[0]} founded? (q{index})",
        "answer": "1990",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": [titles[0]], "sent_id": [0]},
        "context": {"title": titles,
                    "sentences": [[f"The {titles[j]} article started in {1990 + j}.",
                                   f"Its record was {100 + j} units."] for j in range(6)]},
    }


def check_readers(ctx: Ctx) -> tuple[str, str]:
    from src.datasets.reader import SQUAD_ID, load_dataset_split

    try:
        view = load_dataset_split(SQUAD_ID, split="validation")
    except Exception as exc:  # noqa: BLE001
        if ctx.has("datasets"):
            return "FAIL", repr(exc)
        return "SKIP", f"既无 datasets 也无本地副本可用：{exc!r}"
    ctx.dataset_origin = view.origin
    detail = f"SQuAD dev {len(view)} 行（origin={view.origin}）"
    if view.origin == "local" and view.local_sha256:
        detail += f"，sha256={view.local_sha256[:12]}"
    return "OK", detail


def check_operators(_ctx: Ctx) -> tuple[str, str]:
    from src.datasets.challenge_builder import (apply_evidence_conflict, apply_outdated,
                                                apply_retrieval_failure, make_control)
    from src.datasets.reader import normalize_hotpotqa

    problems = []
    counts: dict[str, int] = {}
    for index in range(6):
        example = normalize_hotpotqa(_synthetic_example(index))
        pool = [{"title": f"Pool{index}-{j}", "text": f"Pool{index} passage {j} about Topic{index}."}
                for j in range(3)]
        attacks = {"retrieval_failure": apply_retrieval_failure(
                        example, {"n_replace": 2, "distractor_pool": "native",
                                  "min_lexical_overlap": 0.0, "_pool": pool}, seed=1000 + index),
                   "evidence_conflict": apply_evidence_conflict(
                        example, {"slot_type": ["date", "number", "person", "place"],
                                  "n_inject": 1, "inject_position": "middle"}, seed=1000 + index),
                   "outdated": apply_outdated(example, {"time_gap_years": 2}, seed=1000 + index)}
        for name, result in attacks.items():
            if result is None:
                continue
            counts[name] = counts.get(name, 0) + 1
            if name == "retrieval_failure" and any(g in p["text"] for p in result["passages_for_retrieval"]
                                                   for g in example["gold_context"]):
                problems.append("检索失败算子没有移除 gold 句")
            if name == "evidence_conflict" and not result["injected_values"]:
                problems.append("证据冲突算子没有记录注入值")
            if name == "outdated" and "outdated_version" not in result["construct_params"]:
                problems.append("过时算子没有记录过期版本")
        control = make_control(example, seed=1000 + index)
        if control["challenge_type"] != "none" or not control["construct_params"]["is_control"]:
            problems.append("对照版构造不符合约定")
    if problems:
        return "FAIL", "；".join(sorted(set(problems))[:3])
    return "OK", f"三类算子可构造（{counts}）+ 对照版正常"


def check_labels(_ctx: Ctx) -> tuple[str, str]:
    from src.datasets.challenge_builder import derive_label, is_refusal

    cases = [
        (derive_label("I don't know", "1990", (), "retrieval_failure"), 0, "检索失败+拒答→0"),
        (derive_label("1990", "1990", (), "retrieval_failure"), 1, "检索失败+具体答案→1"),
        (derive_label("in 1985", "1990", ("1985",), "evidence_conflict"), 1, "冲突+注入值→1"),
        (derive_label("1990", "1990", ("1985",), "evidence_conflict"), 0, "冲突+gold→0"),
        (derive_label("unrelated", "1990", ("1985",), "evidence_conflict"), None, "冲突+都不匹配→None"),
        (derive_label("1985", "1990", ("1985",), "outdated"), 1, "过时+过期值→1"),
        (derive_label("1990", "1990", (), "outdated", is_control=True), 0, "对照版+gold→0"),
        (derive_label("x", "1990", (), "none", is_control=True), 1, "对照版+不匹配→1"),
    ]
    wrong = [name for got, want, name in cases if got != want]
    if not is_refusal("无法回答") or not is_refusal(""):
        wrong.append("中文/空答案拒答识别")
    return ("FAIL", f"规则不符：{wrong}") if wrong else ("OK", f"{len(cases)} 条标签规则 + 拒答识别全部正确")


def check_split(_ctx: Ctx) -> tuple[str, str]:
    from src.common.schema import RAGSample
    from src.datasets.split import base_key, split_by_base_sample

    samples = []
    for index in range(10):
        for challenge_type in ("evidence_conflict", "none"):
            samples.append(RAGSample(
                sample_id=f"hotpotqa-dev-{index:05d}-{challenge_type}", source="hotpotqa",
                split="challenge", challenge_type=challenge_type, question="q", gold_answer="a",
                gold_context=("s",), is_hallucination=0 if challenge_type == "none" else 1,
                construct_params={"is_control": challenge_type == "none", "seed": 1000 + index},
                provenance={"base_sample_key": f"hotpotqa-dev-{index:05d}"}))
    fit, test = split_by_base_sample(samples, {"dev_fit": 0.7, "dev_test": 0.3}, seed=13)
    fit_keys = {base_key(s) for s in fit}
    test_keys = {base_key(s) for s in test}
    if fit_keys & test_keys:
        return "FAIL", "污染版与对照版跨侧（数据泄漏）"
    if len(fit) + len(test) != len(samples):
        return "FAIL", "划分后样本数不一致"
    again, _ = split_by_base_sample(samples, {"dev_fit": 0.7}, seed=13)
    if [s.sample_id for s in again] != [s.sample_id for s in fit]:
        return "FAIL", "同种子划分不可复现"
    return "OK", f"10 基样本 → {len(fit_keys)}/{len(test_keys)}，成对同侧且可复现"


def check_features_contract(ctx: Ctx) -> tuple[str, str]:
    from src.common import io, validate
    from src.common.schema import FEATURE_COLUMNS, Prediction, RAGSample
    from src.features.build_features import build_feature_table

    samples, predictions = [], []
    for index in range(4):
        for challenge_type, injected, answer in (("evidence_conflict", ("1985",), "1985"),
                                                 ("retrieval_failure", (), "I don't know"),
                                                 ("none", (), "1990")):
            sample_id = f"hotpotqa-dev-{index:05d}-{challenge_type}"
            samples.append(RAGSample(
                sample_id=sample_id, source="hotpotqa", split="challenge",
                challenge_type=challenge_type, question="q", gold_answer="1990",
                gold_context=("The Topic article started in 1990.",),
                passages_for_retrieval=({"title": "t", "text": "The Topic article started in 1990."},),
                is_hallucination=1 if challenge_type != "none" else 0,
                construct_params={"is_control": challenge_type == "none", "seed": 1000 + index,
                                  "injected_values": list(injected)}))
            predictions.append(Prediction(sample_id=sample_id, exp_id="final-features-13", answer=answer,
                                          baseline_confidence=0.5, latency_ms=10.0, prompt="p"))
    stats: dict = {}
    rows = build_feature_table(predictions, samples, None, {"features": {}}, stats=stats)
    if len(rows) != len(samples):
        return "FAIL", f"特征行数 {len(rows)} != 样本数 {len(samples)}（stats={stats}）"
    if any(set(row.to_dict()) != set(FEATURE_COLUMNS) for row in rows):
        return "FAIL", "特征列与 FEATURE_COLUMNS 不一致"
    path = ctx.subdir("l2") / "final_features.parquet"
    written = io.write_table(path, [row.to_dict() for row in rows], columns=list(FEATURE_COLUMNS))
    problems = validate.validate_file("features", written)
    if problems:
        return "FAIL", f"产物校验不通过：{problems[0]}"
    return "OK", f"标签回填 + {len(rows)} 行特征表通过 validate（{written.suffix.lstrip('.')}）"


def check_missing_deps(ctx: Ctx) -> tuple[str, str]:
    missing = sorted(name for name, ok in ctx.available.items() if not ok)
    if not missing:
        return "OK", "全部依赖可用（可跑完整链路）"
    return "SKIP", f"缺失依赖：{missing}（相关检查将记 SKIP）"


# ———————————————————————— L3 生成管道 ————————————————————————

def _build_samples(ctx: Ctx) -> tuple[pathlib.Path | None, str]:
    """优先构造挑战集（需 pyarrow/datasets），退化为常规集（SQuAD 本地副本，纯标准库）。"""
    out = ctx.subdir("l3-samples")
    code, tail = run_cli("src.datasets.cli", ["--config", "configs/challenge.yaml", "--seed", "1000",
                                              "--limit", str(max(4, ctx.limit * 3)),
                                              "--out", str(out), "--exp-id", "final-1000",
                                              "--audit-out", str(out / "construct_audit.csv")])
    challenge = out / "challenge_set.jsonl"
    if code == 0 and challenge.exists():
        return challenge, "challenge_set（含检索失败/证据冲突/过时三类）"
    normal = out / "normal_set.jsonl"
    if normal.exists():
        return normal, "normal_set（挑战集构造降级）"
    return None, (tail.splitlines() or [""])[-1]  # 只留最后一行错误信息，避免刷屏


def check_dataset_cli(ctx: Ctx) -> tuple[str, str]:
    path, note = _build_samples(ctx)
    if path is None:
        return "SKIP", f"无可用数据集（缺 pyarrow 且无本地副本）：{note}"
    from src.common import validate

    problems = validate.validate_file("samples", path)
    if problems:
        return "FAIL", f"samples 契约校验：{problems[0]}"
    ctx.samples_path = path
    n_lines = len(data_lines(path))
    return "OK", f"{path.name}：{n_lines} 条，来源={note}，契约校验通过"


def check_pipeline_cli(ctx: Ctx) -> tuple[str, str]:
    if ctx.samples_path is None:
        return "SKIP", "上游样本文件缺失，跳过生成管道"
    if not ctx.has("torch", "transformers", "rank_bm25"):
        return "SKIP", ctx.skip_reason("torch", "transformers", "rank_bm25")
    out = ctx.subdir("l3-pred")
    code, tail = run_cli("src.generation.cli", ["--config", "configs/default.yaml", "--seed", "13",
                                                "--samples", str(ctx.samples_path),
                                                "--limit", str(ctx.limit),
                                                "--out", str(out), "--exp-id", "final-pipeline-13"])
    path = out / "final-pipeline-13.jsonl"
    if code != 0 or not path.exists():
        return "FAIL", f"退出码 {code}：{tail}"
    from src.common import validate

    problems = validate.validate_file("predictions", path)
    if problems:
        return "FAIL", f"predictions 契约校验：{problems[0]}"
    return "OK", f"{len(data_lines(path))} 条预测通过 validate（含 _meta、latency、自评置信度）"


#: G1 比较口径：这些字段必须逐字节一致（答案与证据是"可复现"的实质内容）
G1_STRICT_FIELDS = ("sample_id", "answer", "prompt", "passages", "decode")
#: 允许浮点尾差的字段（CUDA fp16 归约顺序差异会带来极小偏差）→ 容差
G1_TOLERANT_FIELDS = {"baseline_confidence": 1e-6}
#: 计时/时间戳类字段，不参与一致性比较
G1_IGNORED_FIELDS = ("latency_ms",)


def read_rows(path: pathlib.Path) -> list[dict]:
    """读 JSONL 的数据行（排除 `_meta`）。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and '"_meta"' not in line]


def g1_compare(rows_a: list[dict], rows_b: list[dict]) -> tuple[bool, str]:
    """按门禁 G1 的口径比较两次运行的数据行，返回 (是否一致, 差异说明)。

    语义字段（`G1_STRICT_FIELDS`）逐字节比较；`G1_TOLERANT_FIELDS` 允许给定容差内的浮点尾差；
    `G1_IGNORED_FIELDS`（计时等）与 `_meta.timestamp` 不参与比较。
    """
    if len(rows_a) != len(rows_b):
        return False, f"数据行数不同：{len(rows_a)} vs {len(rows_b)}"
    max_delta = 0.0
    for index, (left, right) in enumerate(zip(rows_a, rows_b), start=1):
        if left.get("sample_id") != right.get("sample_id"):
            return False, (f"第 {index} 条 sample_id 不同："
                           f"{left.get('sample_id')!r} vs {right.get('sample_id')!r}")
        for field in G1_STRICT_FIELDS:
            if left.get(field) != right.get(field):
                return False, (f"第 {index} 条字段 {field!r} 不同："
                               f"{str(left.get(field))[:120]!r} vs {str(right.get(field))[:120]!r}")
        for field, tolerance in G1_TOLERANT_FIELDS.items():
            try:
                delta = abs(float(left.get(field)) - float(right.get(field)))
            except (TypeError, ValueError):
                return False, f"第 {index} 条字段 {field!r} 不可比较：{left.get(field)!r}"
            if delta != delta:  # NaN：两侧都缺失，视为一致
                continue
            max_delta = max(max_delta, delta)
            if delta > tolerance:
                return False, f"第 {index} 条字段 {field!r} 超出容差 {tolerance:g}：|Δ|={delta:.3g}"
    return True, f"语义字段逐字节一致；baseline_confidence 最大浮点差 {max_delta:.3g}"


def check_g1_determinism(ctx: Ctx) -> tuple[str, str]:
    if ctx.samples_path is None or not ctx.has("torch", "transformers", "rank_bm25"):
        return "SKIP", "需要生成管道可用（torch/transformers/rank_bm25）"
    paths: list[pathlib.Path] = []
    for tag in ("a", "b"):
        out = ctx.subdir(f"l3-g1-{tag}")
        code, tail = run_cli("src.generation.cli", ["--config", "configs/default.yaml", "--seed", "13",
                                                    "--samples", str(ctx.samples_path),
                                                    "--limit", str(ctx.limit), "--out", str(out),
                                                    "--exp-id", "final-g1-13"])
        path = out / "final-g1-13.jsonl"
        if code != 0 or not path.exists():
            return "FAIL", f"第 {tag} 次运行失败（退出码 {code}）：{tail}"
        paths.append(path)
    ok, detail = g1_compare(read_rows(paths[0]), read_rows(paths[1]))
    for field in G1_IGNORED_FIELDS:
        detail += f"；{field} 不参与比较"
    detail += f"（对比文件：{paths[0].relative_to(ROOT)} / {paths[1].relative_to(ROOT)}）"
    return ("OK" if ok else "FAIL"), detail


def check_features_cli(ctx: Ctx) -> tuple[str, str]:
    pred = ctx.subdir("l3-pred") / "final-pipeline-13.jsonl"
    if not pred.exists():
        return "SKIP", "上游预测文件缺失（生成管道未跑）"
    out = ctx.subdir("l3-features")
    code, tail = run_cli("src.features.cli", ["--config", "configs/default.yaml", "--seed", "13",
                                              "--pred", str(pred), "--samples", str(ctx.samples_path),
                                              "--no-nli", "--out", str(out / "final-pipeline-13.parquet")])
    written = out / "final-pipeline-13.parquet"
    if not written.exists():
        written = out / "final-pipeline-13.csv"
    if code != 0 or not written.exists():
        return "FAIL", f"退出码 {code}：{tail}"
    from src.common import validate

    problems = validate.validate_file("features", written)
    if problems:
        return "FAIL", f"features 契约校验：{problems[0]}"
    ctx.features_path = written
    ctx.features_source = "真实管道产物（--no-nli，语义列 NaN）"
    stats = out / "final-pipeline-13-features.json"
    extra = f"，{json.loads(stats.read_text(encoding='utf-8'))['n_undetermined']} 条 undetermined 被剔除" \
        if stats.exists() else ""
    return "OK", f"{written.name} 通过 validate{extra}"


# ———————————————————————— L4 判别与评估 ————————————————————————

def _synthetic_features(ctx: Ctx) -> tuple[pathlib.Path, pathlib.Path]:
    """造合成特征表 + dev-fit 划分（在 L3 未产出真实特征表时使用，报告中会标注）。"""
    from src.common import io
    from src.common.schema import FEATURE_COLUMNS, FeatureRow

    rng = random.Random(13)
    rows = []
    for index in range(30):
        label = index % 2
        for suffix, challenge_type in (("evidence_conflict", "evidence_conflict"), ("none", "none")):
            rows.append(FeatureRow.from_dict({
                "sample_id": f"hotpotqa-dev-{index:05d}-{suffix}", "exp_id": "final-logreg-13",
                "challenge_type": challenge_type, "is_hallucination": label,
                "entailment_max": 0.85 - 0.6 * label + rng.uniform(-0.05, 0.05),
                "entailment_mean": 0.8 - 0.55 * label + rng.uniform(-0.05, 0.05),
                "contradiction_max": 0.1 + 0.5 * label + rng.uniform(-0.05, 0.05),
                "overlap_em": float(1 - label), "overlap_f1": 0.9 - 0.6 * label,
                "citation_cov": 0.9 - 0.3 * label, "conflict_count": float(rng.randint(0, 2)),
                "retrieval_top_score": rng.uniform(2, 9), "answer_len": float(rng.randint(1, 12)),
                "baseline_confidence": 0.6 - 0.10 * label + rng.uniform(-0.4, 0.4),
            }))
    out = ctx.subdir("l4-synth")
    features = out / "final-logreg-13.parquet"
    written = io.write_table(features, [row.to_dict() for row in rows], columns=list(FEATURE_COLUMNS))
    fit_split = out / "split_dev_fit.jsonl"
    io.write_jsonl(fit_split, [{"sample_id": row.sample_id} for row in rows
                               if int(row.sample_id.split("-")[2]) % 10 < 7])
    return written, fit_split


def _true_features_usable(ctx: Ctx) -> tuple[bool, str]:
    """判断 L3 的真实特征表能否支撑 5 折分层分组 CV（行数与两类标签都要够）。"""
    if ctx.features_path is None:
        return False, "L3 未产出真实特征表"
    from src.common import io

    try:
        rows = io.read_table(ctx.features_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"读取失败（{exc!r}）"
    labels = {int(row.get("is_hallucination", -1)) for row in rows}
    if len(rows) < 20 or labels != {0, 1}:
        return False, (f"真实特征表仅 {len(rows)} 行、标签 {sorted(labels)}，不足以做 5 折分层分组 CV"
                       f"（要覆盖真实特征链路请把 --limit 提到 ≥ 30）")
    return True, f"真实管道产物 {len(rows)} 行"


def _fit_split_of(ctx: Ctx, features: pathlib.Path) -> pathlib.Path:
    """按基样本 7:3 生成 dev-fit 划分（键取 sample_id 去掉末段），保证报告侧非空。"""
    from src.common import io
    from src.detection.train import base_key as sample_base_key  # 按 sample_id 派生基样本键

    groups: dict[str, list[str]] = {}
    for row in io.read_table(features):
        sample_id = str(row["sample_id"])
        groups.setdefault(sample_base_key(sample_id), []).append(sample_id)
    keys = sorted(groups)
    n_fit = max(1, min(len(keys) - 1, int(round(len(keys) * 0.7))))
    fit_ids = {sample_id for key in keys[:n_fit] for sample_id in groups[key]}
    path = ctx.subdir("l4-real") / "split_dev_fit.jsonl"
    io.write_jsonl(path, [{"sample_id": sample_id} for sample_id in sorted(fit_ids)])
    return path


def check_detection_cli(ctx: Ctx) -> tuple[str, str]:
    if not ctx.has("sklearn", "numpy"):
        return "SKIP", ctx.skip_reason("sklearn", "numpy")
    usable, why = _true_features_usable(ctx)
    if usable:
        features, fit_split, source = ctx.features_path, _fit_split_of(ctx, ctx.features_path), why
    else:
        features, fit_split = _synthetic_features(ctx)
        source = f"合成特征（{why}）"

    out = ctx.subdir("l4-metrics")
    common = ["--config", "configs/detect.yaml", "--seed", "13", "--features", str(features),
              "--fit-split", str(fit_split), "--out", str(out)]
    code, tail = run_cli("src.detection.cli", [*common, "--classifier", "logreg",
                                               "--exp-id", "final-logreg-13"])
    if code != 0:
        return "FAIL", f"主方法 run 退出码 {code}：{tail}"
    code_b1, tail_b1 = run_cli("src.detection.cli", [*common, "--feature-set", "self_confidence",
                                                     "--exp-id", "final-selfconf-13"])
    if code_b1 != 0:
        return "FAIL", f"B1 基线 run 退出码 {code_b1}：{tail_b1}"

    metrics = out / "metrics" / "final-logreg-13.json"
    if not metrics.exists():
        return "FAIL", "未产出 MetricRecord"
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    required = ("exp_id", "stage", "seed", "model", "split", "n_samples", "auc", "pr_auc", "precision",
                "recall", "f1_macro", "f1_micro", "confusion", "threshold", "config_hash", "env_report",
                "timestamp")
    missing = [key for key in required if key not in payload]
    if missing:
        return "FAIL", f"MetricRecord 缺字段：{missing}"
    calib = out / "metrics" / "final-logreg-13-calibration.json"
    if not calib.exists():
        return "FAIL", "未产出校准对比（Abl-3）"
    ctx.extra_checks.append(("detection 产出分组指标与 CV 指标",
                             bool(payload.get("grouped")) and bool(payload.get("cv_metrics"))))
    return "OK", (f"来源={source}；AUC={payload['auc']:.4f}, F1(macro)={payload['f1_macro']:.4f}, "
                  f"n={payload['n_samples']}；校准与分组指标齐备")


def check_evaluation_cli(ctx: Ctx) -> tuple[str, str]:
    if not ctx.has("sklearn", "numpy", "scipy"):
        return "SKIP", ctx.skip_reason("sklearn", "numpy", "scipy")
    root = ctx.subdir("l4-metrics")
    metrics = root / "metrics" / "final-logreg-13.json"
    if not metrics.exists():
        return "SKIP", "上游 detection 未产出指标（先跑 L4 判别器检查）"
    out = ctx.subdir("l4-eval")
    code, tail = run_cli("src.evaluation.cli", [
        "--config", "configs/detect.yaml", "--seed", "13", "--exp-ids", str(metrics),
        "--baseline-exp-id", "final-selfconf-13", "--out", str(out),
        "--scores-dir", str(root / "scores"), "--figures-dir", str(ctx.figures_dir),
        *(["--no-figures"] if not ctx.has("matplotlib") else [])])
    if code != 0:
        return "FAIL", f"退出码 {code}：{tail}"
    for name in ("summary.csv", "summary.json", "summary.md"):
        if not (out / name).exists():
            return "FAIL", f"缺少 {name}"
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    primary_exp_id = next(iter(payload.get("runs") or ["final-logreg-13"]))
    sig = payload.get("significance") or {}
    if sig.get("n_pairs", 0) <= 0:
        return "FAIL", "配对检验未在共同 sample_id 上完成"
    delta = sig.get("delta_auc")
    detail = (f"配对 n={sig['n_pairs']}，ΔAUC={delta if delta is None else f'{delta:.4f}'}，"
              f"p_t={sig['ttest']['p_value']:.3g}，p_wilcoxon={sig['wilcoxon']['p_value']:.3g}")
    figures = sorted(ctx.figures_dir.glob(f"{primary_exp_id}-*.png"))
    if ctx.has("matplotlib"):
        if len(figures) < 6:
            return "FAIL", f"必备图不足（{len(figures)}/6，目录 {ctx.figures_dir}）"
        detail += (f"；六张图齐备（{sum(f.stat().st_size for f in figures) // 1024} KB，"
                   f"目录 {ctx.figures_dir}）")
    else:
        detail += "；matplotlib 缺失，跳过出图"
    return "OK", detail


CHECKS: list[tuple[str, str, object]] = [
    ("L0", "编译通过（src/scripts）", check_compile),
    ("L0", "关键路径齐备", check_paths),
    ("L0", "配置解析与贪心解码", check_configs),
    ("L0", "接口契约往返与特征列", check_contract),
    ("L0", "业务模块可导入", check_imports),
    ("L0", "无占位实现残留", check_no_skeleton),
    ("L0", "CLI 入口 --help", check_cli_help),
    ("L1", "tests/ 单元测试", check_unit_tests),
    ("L2", "依赖可用性清点", check_missing_deps),
    ("L2", "数据集读取（在线/本地降级）", check_readers),
    ("L2", "三类退化算子 + 对照版", check_operators),
    ("L2", "标签规则化推导", check_labels),
    ("L2", "按基样本划分（成对同侧）", check_split),
    ("L2", "特征表与契约校验", check_features_contract),
    ("L3", "构造集/常规集落盘 + 契约校验", check_dataset_cli),
    ("L3", "生成管道 + predictions 契约校验", check_pipeline_cli),
    ("L3", "门禁 G1：同种子两次运行一致", check_g1_determinism),
    ("L3", "特征表落盘 + 契约校验", check_features_cli),
    ("L4", "判别器训练 + 校准（Abl-3）", check_detection_cli),
    ("L4", "分组指标 + 配对检验 + 图表", check_evaluation_cli),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="课题 C09 最终验收测试（L0–L4）")
    parser.add_argument("--level", type=int, default=4, choices=[0, 1, 2, 3, 4],
                        help="跑到第几层（默认 4 = 全跑）")
    parser.add_argument("--fast", action="store_true", help="等价于 --level 2（只跑静态/单元/数据链路）")
    parser.add_argument("--limit", type=int, default=2, help="L3/L4 的样本条数（冒烟尺度）")
    parser.add_argument("--artifacts-dir", default=str(ROOT / "results" / "final_test"),
                        help="验收中间产物目录（默认 results/final_test，默认保留）")
    parser.add_argument("--figures-dir", default=str(ROOT / "results" / "final_test" / "figures"),
                        help="图表输出目录（默认 results/final_test/figures；真实结果图请用 results/figures）")
    args = parser.parse_args(argv)

    max_level = 2 if args.fast else args.level
    ctx = Ctx(artifacts_dir=pathlib.Path(args.artifacts_dir),
              figures_dir=pathlib.Path(args.figures_dir), limit=max(1, args.limit))
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    # 清理上一次运行留下的中间产物：否则 L3/L4 会读到 stale 结果而"误报 OK"（图片目录不动，按名覆盖）
    stale = [d for d in sorted(ctx.artifacts_dir.glob("l*")) if d.is_dir()]
    for directory in stale:
        shutil.rmtree(directory, ignore_errors=True)
    # 只在本次会出图的层级（L4）清理旧图，避免 --fast 把上次全跑的图删掉
    stale_figures = (sorted(ctx.figures_dir.glob("final-*.png"))
                     if max_level >= 4 and ctx.figures_dir.exists() else [])
    for figure in stale_figures:
        figure.unlink()
    report = Report()
    print("== 课题 C09 最终验收测试 ==")
    print(f"仓库: {ROOT}")
    print(f"层级: L0–L{max_level}｜样本条数: {ctx.limit}")
    print(f"产物目录: {ctx.artifacts_dir}")
    print(f"验收图目录: {ctx.figures_dir}｜集成测试图目录: {ctx.artifacts_dir / 'test_figures'}")
    if stale or stale_figures:
        print(f"已清理上次遗留：中间产物 {', '.join(d.name for d in stale) or '无'}"
              f"｜旧图 {len(stale_figures)} 张")
    if ctx.available and not all(ctx.available.values()):
        missing = [name for name, ok in ctx.available.items() if not ok]
        print(f"缺失依赖（相关检查将 SKIP）: {missing}")

    # 用 try/finally 保证即使输出被管道截断（BrokenPipeError）也能清理本次新增的日志
    try:
        for level, name, func in CHECKS:
            if LEVELS.index(level) > max_level:
                continue
            if not report.rows or report.rows[-1][1] != level:
                report.section(level)
            try:
                status, detail = func(ctx)  # type: ignore[operator]
            except Exception as exc:  # noqa: BLE001 - 单项异常不应中断整轮验收
                status, detail = "FAIL", repr(exc)
            report.add(status, level, name, detail)

        for name, ok in ctx.extra_checks:
            report.add("OK" if ok else "FAIL", "L4", f"派生结论：{name}", "")

        counts = report.totals()
        print(f"\n结果：OK {counts['OK']}，SKIP {counts['SKIP']}，FAIL {counts['FAIL']}")
        if counts["FAIL"]:
            print("失败项：")
            for status, level, name, detail in report.rows:
                if status == "FAIL":
                    print(f"  - [{level}] {name}：{detail}")
    finally:
        ctx.cleanup()
    print(f"产物保留在：{ctx.artifacts_dir}")
    if ctx.figures_dir.exists():
        figures = sorted(ctx.figures_dir.glob("*.png"))
        if figures:
            print(f"验收图保留在：{ctx.figures_dir}（{len(figures)} 张，"
                  f"{sum(f.stat().st_size for f in figures) // 1024} KB）")
    test_figures = ctx.artifacts_dir / "test_figures"
    if test_figures.exists():
        count = len(sorted(test_figures.glob("*.png")))
        print(f"集成测试图目录：{test_figures}（{count} 张，来自 python -m unittest tests.test_end_to_end）")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
