#!/usr/bin/env python3
"""概率校准（Platt / 温度缩放）。

负责人: C
对应文档: docs/实验与评估规范.md §3
状态: 已实现（T4.1）

`calibrate(scores, y_true, method)` 返回一个**可调用映射** `f(scores) -> 概率`，便于在评估阶段把
判别器原始分数校准后再算阈值指标，并绘制可靠性图（校准曲线，必备图之一）。

- `platt`：对原始分数做一维逻辑回归（等价于 Platt scaling，含斜率与截距）；
- `temperature`：把分数视作 logit，温度 T > 0 最小化 NLL（在网格上搜索，确定性、无需额外依赖）；
- `none`：恒等映射（作为对照项）。
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from ..common.logging_utils import get_logger

LOG = get_logger(__name__)

CALIBRATION_METHODS = ("none", "platt", "temperature")
#: 温度搜索网格（对数间隔），避免依赖 scipy.optimize 的收敛差异
TEMPERATURE_GRID = (0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0)
EPS = 1e-6


def _logit(score: float) -> float:
    clipped = min(max(float(score), EPS), 1.0 - EPS)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _nll(probs: Sequence[float], y_true: Sequence[int]) -> float:
    total = 0.0
    for prob, label in zip(probs, y_true):
        clipped = min(max(float(prob), EPS), 1.0 - EPS)
        total -= math.log(clipped) if int(label) == 1 else math.log(1.0 - clipped)
    return total / max(1, len(y_true))


def calibrate(scores: Sequence[float], y_true: Sequence[int], method: str = "platt") -> Callable[[Sequence[float]], list[float]]:
    """拟合校准器并返回可调用的映射函数 `f(scores) -> list[float]`。

    拟合数据应为**训练侧**（dev-fit）的分数与标签；在报告侧（dev-test）上应用该映射，避免用
    测试数据拟合校准器。
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"未知校准方法：{method!r}（可选 {CALIBRATION_METHODS}）")
    values = [float(s) for s in scores]
    labels = [int(v) for v in y_true]
    if len(values) != len(labels):
        raise ValueError("scores 与 y_true 长度不一致")

    if method == "none" or len(values) < 2 or len(set(labels)) < 2:
        if method != "none":
            LOG.warning("校准样本不足或标签单一，%s 退化为恒等映射", method)
        return lambda xs: [float(x) for x in xs]

    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=0)
        model.fit([[v] for v in values], labels)
        return lambda xs: [float(p) for p in model.predict_proba([[float(x)] for x in xs])[:, 1]]

    # 温度缩放：在网格上选 NLL 最小的 T
    best_temp, best_nll = 1.0, float("inf")
    for temp in TEMPERATURE_GRID:
        probs = [_sigmoid(_logit(v) / temp) for v in values]
        score = _nll(probs, labels)
        if score < best_nll:
            best_temp, best_nll = temp, score
    LOG.info("温度缩放：T=%.2f（NLL=%.4f）", best_temp, best_nll)
    return lambda xs: [_sigmoid(_logit(float(x)) / best_temp) for x in xs]


def best_threshold(scores: Sequence[float], y_true: Sequence[int],
                   grid: Sequence[float] | None = None) -> float:
    """在给定分数上搜索使 F1 最大的阈值（用于契约里的"校准后的最优阈值"）。"""
    candidates = list(grid) if grid is not None else [i / 100 for i in range(5, 96, 1)]
    labels = [int(v) for v in y_true]
    best_value, best_f1 = 0.5, -1.0
    for threshold in candidates:
        tp = sum(1 for s, y in zip(scores, labels) if y == 1 and float(s) >= threshold)
        fp = sum(1 for s, y in zip(scores, labels) if y == 0 and float(s) >= threshold)
        fn = sum(1 for s, y in zip(scores, labels) if y == 1 and float(s) < threshold)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_value, best_f1 = float(threshold), f1
    return best_value
