#!/usr/bin/env python3
"""指标口径测试（docs/实验与评估规范.md §1）。

运行: python -m unittest tests.test_metrics -v
依赖 scikit-learn（未安装时自动跳过）。
"""
from __future__ import annotations

import unittest

from src.evaluation.metrics import latency_stats, score_metrics

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


class TestLatencyStats(unittest.TestCase):
    def test_mean_and_p95(self) -> None:
        stats = latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertAlmostEqual(stats["latency_ms_mean"], 22.0)
        self.assertEqual(stats["latency_ms_p95"], 100.0)

    def test_empty(self) -> None:
        stats = latency_stats([])
        self.assertNotEqual(stats["latency_ms_mean"], stats["latency_ms_mean"])  # NaN


@unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
class TestScoreMetrics(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        result = score_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["pr_auc"], 1.0)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["confusion"], {"tp": 2, "fp": 0, "tn": 2, "fn": 0})
        self.assertEqual(result["n_samples"], 4)
        self.assertEqual((result["n_positive"], result["n_negative"]), (2, 2))

    def test_reversed_scores_give_below_half(self) -> None:
        result = score_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], threshold=0.5)
        self.assertEqual(result["auc"], 0.0)

    def test_single_class_auc_is_nan(self) -> None:
        result = score_metrics([1, 1, 1], [0.9, 0.8, 0.7])
        self.assertNotEqual(result["auc"], result["auc"])  # NaN，而不是抛异常
        self.assertEqual(result["n_positive"], 3)

    def test_threshold_effect(self) -> None:
        low = score_metrics([0, 1], [0.4, 0.6], threshold=0.5)
        high = score_metrics([0, 1], [0.4, 0.6], threshold=0.7)
        self.assertEqual(low["confusion"]["tp"], 1)
        self.assertEqual(high["confusion"]["tp"], 0)
        self.assertEqual(high["recall"], 0.0)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            score_metrics([0, 1], [0.5])


if __name__ == "__main__":
    unittest.main()
