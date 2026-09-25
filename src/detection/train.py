#!/usr/bin/env python3
"""训练与交叉验证。

负责人: C
对应文档: docs/实验与评估规范.md §4
状态: 已实现（T4.1）

纪律:
- 训练集用 dev-fit、报告用 dev-test（由 `detection/cli.py` 按 `split_dev_fit.jsonl` 划分）；
- 交叉验证按**基样本**分组（同一样本的污染版与对照版不跨折），避免折内泄漏——分组键由
  `sample_id` 去掉末段得到（`hotpotqa-dev-00042-retrieval_failure` → `hotpotqa-dev-00042`）；
- 交叉验证指标用 out-of-fold 分数计算（不是训练分数），报告须写明口径。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import FeatureRow
from ..evaluation.metrics import score_metrics
from .model import make_classifier

LOG = get_logger(__name__)


def base_key(sample_id: str) -> str:
    """从 sample_id 派生基样本键（去掉末尾的 challenge_type 段）。"""
    parts = str(sample_id).rsplit("-", 1)
    return parts[0] if len(parts) == 2 else str(sample_id)


def matrix_and_labels(features: Sequence[FeatureRow],
                      feature_cols: Sequence[str]) -> tuple[list[list[float]], list[int]]:
    """按 `feature_cols` 取特征矩阵与标签（缺失值保留 NaN，由判别器内的 imputer 处理）。"""
    rows = [[float(getattr(row, col, float("nan"))) for col in feature_cols] for row in features]
    labels = [int(row.is_hallucination) for row in features]
    return rows, labels


def train_and_evaluate(features: Sequence[FeatureRow], feature_cols: Sequence[str], kind: str,
                       seed: int, cv_folds: int = 5,
                       params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """5 折交叉验证训练并返回指标与模型对象。

    返回 dict：`cv_metrics`（OOF 分数上的指标）、`oof_scores`（逐样本，顺序同 `features`）、
    `model`（全量拟合的判别器，供 dev-test 报告与落盘）。样本或类别不足以做 CV 时，
    `cv_metrics` 为空 dict 并记日志（不伪造指标）。
    """
    rows, labels = matrix_and_labels(features, feature_cols)
    result: dict[str, Any] = {
        "kind": kind, "seed": int(seed), "cv_folds": int(cv_folds), "feature_cols": list(feature_cols),
        "n_samples": len(rows), "label_counts": {str(v): labels.count(v) for v in sorted(set(labels))},
    }
    if not rows or len(set(labels)) < 2:
        LOG.warning("样本不足或标签单一（n=%d, 标签=%s），跳过交叉验证",
                    len(rows), result["label_counts"])
        result["cv_metrics"] = {}
        result["oof_scores"] = [float("nan")] * len(rows)
        result["model"] = make_classifier(kind, seed, params).fit(rows, labels) if rows else None
        return result

    try:
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("交叉验证需要 scikit-learn 与 numpy（见 requirements.txt）") from exc

    groups = [base_key(row.sample_id) for row in features]
    n_splits = max(2, min(int(cv_folds), len(set(groups)), labels.count(0), labels.count(1)))
    if n_splits < int(cv_folds):
        LOG.warning("折数由 %d 调整为 %d（受分组数或类别样本量限制）", cv_folds, n_splits)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(rows), np.nan)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(rows, labels, groups), start=1):
        estimator = make_classifier(kind, seed, params)
        estimator.fit([rows[i] for i in train_idx], [labels[i] for i in train_idx])
        for position, prob in zip(test_idx, estimator.predict_proba([rows[i] for i in test_idx])[:, 1]):
            oof[position] = float(prob)
        LOG.debug("第 %d/%d 折完成（train=%d, test=%d）", fold, n_splits, len(train_idx), len(test_idx))

    if np.isnan(oof).any():
        LOG.warning("OOF 分数存在缺失（%d 条），这些样本不计入 CV 指标",
                    int(np.isnan(oof).sum()))
    covered = [i for i in range(len(rows)) if not np.isnan(oof[i])]
    result["oof_scores"] = [float(v) for v in oof]
    result["cv_metrics"] = score_metrics([labels[i] for i in covered], [float(oof[i]) for i in covered]) \
        if covered else {}
    result["cv_n_splits"] = n_splits

    final_model = make_classifier(kind, seed, params)
    final_model.fit(rows, labels)
    result["model"] = final_model
    if result["cv_metrics"]:
        LOG.info("CV(%d 折) 指标：AUC=%.4f, PR-AUC=%.4f, F1(macro)=%.4f",
                 n_splits, result["cv_metrics"]["auc"], result["cv_metrics"]["pr_auc"],
                 result["cv_metrics"]["f1_macro"])
    return result
