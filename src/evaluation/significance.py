#!/usr/bin/env python3
"""配对显著性检验（配对 t / Wilcoxon）。

负责人: C
对应文档: docs/实验与评估规范.md §4
状态: 已实现（T5.1）

纪律:
- **配对单位必须是 `sample_id`**：调用方须保证 `scores_a[i]` 与 `scores_b[i]` 是同一 `sample_id`
  上的两个方法分数（不得跨样本配对），本模块不重排输入；
- 主判据是 `AUC(M) − AUC(B1) ≥ 0.10`（3 种子均值），并给配对检验的 p 值；
- 分组评估会放大假阳性：`family_size` > 1 时同时给出 Holm 校正后的 p 值，报告中须标注比较次数。
"""
from __future__ import annotations

from typing import Any, Sequence

from ..common.logging_utils import get_logger

LOG = get_logger(__name__)


def _auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    from .metrics import score_metrics

    return float(score_metrics(list(y_true), list(scores))["auc"])


def paired_tests(y_true: Sequence[int], scores_a: Sequence[float], scores_b: Sequence[float],
                 alpha: float = 0.05, family_size: int = 1,
                 label_a: str = "M", label_b: str = "B1") -> dict[str, Any]:
    """对同一样本上的两种方法做配对 t 检验与 Wilcoxon 符号秩检验，各报 p 值与 AUC 差。

    返回：`n_pairs`、`auc_a`/`auc_b`/`delta_auc`、`ttest`/`wilcoxon`（含统计量与 p 值）、
    两个检验在 α 下是否显著、以及 `family_size` > 1 时的 Holm 校正 p 值。
    """
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("y_true / scores_a / scores_b 长度必须一致（配对单位是 sample_id）")
    try:
        import numpy as np
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("显著性检验需要 scipy 与 numpy（见 requirements.txt）") from exc

    labels = np.asarray([int(v) for v in y_true])
    a = np.asarray([float(v) for v in scores_a])
    b = np.asarray([float(v) for v in scores_b])
    diff = a - b

    result: dict[str, Any] = {
        "label_a": label_a, "label_b": label_b, "alpha": float(alpha), "n_pairs": int(len(diff)),
        "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
        "auc_a": _auc(labels, a) if len(set(labels.tolist())) > 1 else float("nan"),
        "auc_b": _auc(labels, b) if len(set(labels.tolist())) > 1 else float("nan"),
    }
    result["delta_auc"] = result["auc_a"] - result["auc_b"]

    if len(diff) < 2 or float(np.abs(diff).sum()) == 0.0:
        LOG.warning("配对样本不足或分数差全为 0（n=%d），检验结果记为 NaN", len(diff))
        result.update({"ttest": {"statistic": float("nan"), "p_value": float("nan")},
                       "wilcoxon": {"statistic": float("nan"), "p_value": float("nan")},
                       "significant_t": None, "significant_wilcoxon": None})
        return result

    t_stat, p_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_w = stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError as exc:  # 全部差值为 0 等退化情况
        LOG.warning("Wilcoxon 无法计算（%r），记为 NaN", exc)
        w_stat, p_w = float("nan"), float("nan")

    result["ttest"] = {"statistic": float(t_stat), "p_value": float(p_t)}
    result["wilcoxon"] = {"statistic": float(w_stat), "p_value": float(p_w)}
    result["significant_t"] = bool(p_t < alpha) if p_t == p_t else None
    result["significant_wilcoxon"] = bool(p_w < alpha) if p_w == p_w else None
    if int(family_size) > 1:
        result["family_size"] = int(family_size)
        result["holm_ttest"] = float(min(1.0, p_t * int(family_size))) if p_t == p_t else float("nan")
        result["holm_wilcoxon"] = float(min(1.0, p_w * int(family_size))) if p_w == p_w else float("nan")
        result["holm_note"] = "单步 Holm 校正（比较次数见 family_size），报告中须标注"
    result["meets_target"] = bool(result["delta_auc"] >= 0.10) if result["delta_auc"] == result["delta_auc"] else None
    return result
