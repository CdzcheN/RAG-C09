#!/usr/bin/env python3
"""挑战子集构造：三类退化算子与标签规则化推导。

负责人: A
对应文档: docs/数据构造规范.md §2、§3
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def apply_retrieval_failure(example: Mapping[str, Any], params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """算子 A：移除 gold 证据句并替换为同主题干扰段。

    状态: 待实现 —— T2.2/T3.1：优先用 HotpotQA distractor 自带干扰文；受 min_lexical_overlap 约束
    """
    raise NotImplementedError("T2.2/T3.1：优先用 HotpotQA distractor 自带干扰文；受 min_lexical_overlap 约束")


def apply_evidence_conflict(example: Mapping[str, Any], params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """算子 B：在上下文中部注入与真值矛盾的槽位句。

    状态: 待实现 —— T2.2/T3.1：槽位类型与位置见 configs/challenge.yaml
    """
    raise NotImplementedError("T2.2/T3.1：槽位类型与位置见 configs/challenge.yaml")


def apply_outdated(example: Mapping[str, Any], params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """算子 C：把时间敏感事实改写为过期版本。

    状态: 待实现 —— T2.2/T3.1：样本不足时按指南 §6 降级并如实报告
    """
    raise NotImplementedError("T2.2/T3.1：样本不足时按指南 §6 降级并如实报告")


def derive_label(answer: str, gold_answer: str, injected_values: Sequence[str], challenge_type: str, is_control: bool = False) -> int | None:
    """规则化标签推导；无法判定返回 None（记 undetermined 并排除）。

    状态: 待实现 —— T2.1：严格按数据构造规范 §3 实现，禁止改用 NLI 等模型信号（避免标签与特征同源）
    """
    raise NotImplementedError("T2.1：严格按数据构造规范 §3 实现，禁止改用 NLI 等模型信号（避免标签与特征同源）")


def build_challenge_set(cfg: Mapping[str, Any]) -> list[RAGSample]:
    """按配置生成挑战集（三类各 100 例，污染版与对照版成对）。

    状态: 待实现 —— T3.1：需先通过 G3-a 原型门禁（每类 20 例人工核对 100% 正确）
    """
    raise NotImplementedError("T3.1：需先通过 G3-a 原型门禁（每类 20 例人工核对 100% 正确）")


