#!/usr/bin/env python3
"""判别器定义（LogReg / GBDT / 轻量 MLP）。

负责人: C
对应文档: docs/实验与评估规范.md §3
状态: 已实现（T4.1）

三种判别器都必须固定 `random_state`（多种子可比、可复现的前提）。特征可能存在缺失（如 NLI 不可用
时为 NaN），因此统一前置 `SimpleImputer(median)`；线性模型与 MLP 另加 `StandardScaler`。
超参默认值集中在本模块的 `DEFAULT_PARAMS`，可由配置（`detect.detector_params`）覆盖——代码内不写
散落的魔法数。
"""
from __future__ import annotations

from typing import Any, Mapping

from ..common.logging_utils import get_logger

LOG = get_logger(__name__)

CLASSIFIER_KINDS = ("logreg", "gbdt", "mlp")

#: 各判别器的默认超参（可由 `detect.detector_params.<kind>` 覆盖）
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "scale": True},
    "gbdt": {"learning_rate": 0.05, "max_iter": 300, "max_depth": 3, "min_samples_leaf": 5,
             "l2_regularization": 1.0, "class_weight": "balanced", "scale": False},
    "mlp": {"hidden_layer_sizes": (32,), "alpha": 1e-3, "max_iter": 500, "scale": True},
}


def _sklearn_pipeline(scale: bool):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline, steps


def make_classifier(kind: str = "logreg", seed: int = 13,
                    params: Mapping[str, Any] | None = None) -> Any:
    """按名称返回判别器实例（必须固定 random_state）。

    `kind` ∈ {`logreg`, `gbdt`, `mlp`}（Abl-3 需要三种可替换判别器）。GBDT 用
    `HistGradientBoostingClassifier`（直方图实现，训练快且支持 NaN/blob 特征）。
    """
    if kind not in CLASSIFIER_KINDS:
        raise ValueError(f"未知判别器：{kind!r}（可选 {CLASSIFIER_KINDS}）")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "判别器需要 scikit-learn，请先执行 python -m pip install -r requirements.txt"
        ) from exc

    merged = {**DEFAULT_PARAMS[kind], **dict(params or {})}
    scale = bool(merged.pop("scale", False))
    Pipeline, steps = _sklearn_pipeline(scale)

    if kind == "logreg":
        estimator: Any = LogisticRegression(random_state=seed, **merged)
    elif kind == "gbdt":
        estimator = HistGradientBoostingClassifier(random_state=seed, **merged)
    else:
        estimator = MLPClassifier(random_state=seed, early_stopping=False, **merged)

    steps.append(("classifier", estimator))
    LOG.debug("构造判别器 %s：%s", kind, merged)
    return Pipeline(steps)
