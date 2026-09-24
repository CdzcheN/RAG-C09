#!/usr/bin/env python3
"""接口契约测试：schema 往返一致 + 产物校验规则（docs/接口契约.md §2、§6）。

运行: python -m unittest tests.test_contract -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.common import io, validate
from src.common.schema import (FEATURE_COLUMNS, METHOD_FEATURE_GROUPS, MetricRecord,
                               Prediction, RAGSample, RetrievedPassage)


class TestSchemaRoundtrip(unittest.TestCase):
    def test_sample_roundtrip(self) -> None:
        s = RAGSample(sample_id="hotpotqa-dev-00427-retrieval_failure", source="hotpotqa",
                      split="challenge", challenge_type="retrieval_failure", question="Q?",
                      gold_answer="A", gold_context=("句1", "句2"), is_hallucination=1,
                      construct_params={"n_replace": 2})
        self.assertEqual(RAGSample.from_dict(s.to_dict()).to_dict(), s.to_dict())

    def test_prediction_roundtrip(self) -> None:
        p = Prediction(sample_id="s-1", exp_id="w3-smoke-13", answer="A",
                       passages=(RetrievedPassage(rank=1, title="t", text="x", score=1.5),),
                       baseline_confidence=0.7, latency_ms=12.0)
        self.assertEqual(Prediction.from_dict(p.to_dict()).to_dict(), p.to_dict())

    def test_feature_columns_frozen(self) -> None:
        self.assertEqual(len(FEATURE_COLUMNS), 14)
        self.assertIn("baseline_confidence", FEATURE_COLUMNS)

    def test_baseline_not_in_method_groups(self) -> None:
        flat = {c for cols in METHOD_FEATURE_GROUPS.values() for c in cols}
        self.assertNotIn("baseline_confidence", flat)

    def test_metric_record_serialisable(self) -> None:
        m = MetricRecord(exp_id="w4-logreg-13", stage="W4", seed=13, model="m",
                         split="challenge", n_samples=300, auc=0.81)
        self.assertEqual(m.to_dict()["auc"], 0.81)


class TestIOContract(unittest.TestCase):
    def test_meta_first_line_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            io.write_jsonl(path, [{"sample_id": "s-1", "exp_id": "e", "answer": "x"}],
                           meta=io.build_meta("e", 13, {"seeds": [13]}))
            meta, rows = io.read_jsonl_meta(path)
            self.assertEqual(meta["exp_id"], "e")
            self.assertIn("config_hash", meta)
            self.assertEqual(len(rows), 1)
            self.assertEqual(list(Path(td).glob("*.tmp")), [])

    def test_config_hash_stable_and_key_order_independent(self) -> None:
        a = {"x": 1, "y": [1, 2]}
        b = {"y": [1, 2], "x": 1}
        self.assertEqual(io.config_hash(a), io.config_hash(b))


class TestValidateRules(unittest.TestCase):
    def test_predictions_valid(self) -> None:
        rows = [{"sample_id": "a", "exp_id": "e", "answer": "x", "baseline_confidence": 0.5, "latency_ms": 1.0}]
        problems = validate._problems_predictions(rows, {"exp_id": "e", "seed": 1, "config_hash": "h"})
        self.assertEqual(problems, [])

    def test_predictions_detects_violations(self) -> None:
        rows = [{"sample_id": "a", "exp_id": "e"},
                {"sample_id": "a", "exp_id": "e", "answer": "y", "baseline_confidence": 1.7, "latency_ms": -1}]
        problems = validate._problems_predictions(rows, {})
        joined = " | ".join(problems)
        self.assertIn("缺少首行 _meta", joined)
        self.assertIn("缺少必需字段 answer", joined)
        self.assertIn("sample_id 重复", joined)
        self.assertIn("超出", joined)

    def test_features_detects_unregistered_column(self) -> None:
        row = {c: ("s" if c in ("sample_id", "exp_id", "challenge_type") else 0.5) for c in FEATURE_COLUMNS}
        row.update({"challenge_type": "evidence_conflict", "is_hallucination": 1, "extra_col": 1.0})
        problems = validate._problems_features([row])
        self.assertTrue(any("未登记的列" in p for p in problems))

    def test_features_valid_row(self) -> None:
        row = {c: ("s" if c in ("sample_id", "exp_id", "challenge_type") else 0.5) for c in FEATURE_COLUMNS}
        row.update({"challenge_type": "outdated", "is_hallucination": 0})
        self.assertEqual(validate._problems_features([row]), [])


if __name__ == "__main__":
    unittest.main()
