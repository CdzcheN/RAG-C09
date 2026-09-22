#!/usr/bin/env python3
"""词面重叠特征的单元测试（固定已知值，防止实现漂移）。

运行: python -m unittest tests.test_overlap -v
"""
from __future__ import annotations

import unittest

from src.features.overlap import (citation_coverage, exact_match, normalize,
                                  overlap_features, token_f1, tokenize)


class TestTokenize(unittest.TestCase):
    def test_lowercase_and_split(self) -> None:
        self.assertEqual(tokenize("Hello, World-2024!"), ["hello", "world", "2024"])

    def test_empty(self) -> None:
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])  # type: ignore[arg-type]

    def test_normalize_strips_punctuation(self) -> None:
        self.assertEqual(normalize("The Cat, sat."), "the cat sat")


class TestExactMatch(unittest.TestCase):
    def test_identical_ignoring_case_and_punctuation(self) -> None:
        self.assertEqual(exact_match("Paris", "paris."), 1.0)

    def test_different(self) -> None:
        self.assertEqual(exact_match("Paris", "London"), 0.0)

    def test_empty_both(self) -> None:
        self.assertEqual(exact_match("", ""), 1.0)


class TestTokenF1(unittest.TestCase):
    def test_perfect(self) -> None:
        self.assertEqual(token_f1("the cat sat", "the cat sat"), 1.0)

    def test_known_partial(self) -> None:
        # 预测 3 词全命中，gold 6 词：P=1.0, R=0.5 → F1=2/3
        self.assertAlmostEqual(token_f1("the cat sat", "the cat sat on the mat"), 2 / 3, places=6)

    def test_disjoint(self) -> None:
        self.assertEqual(token_f1("alpha beta", "gamma delta"), 0.0)

    def test_empty_prediction(self) -> None:
        self.assertEqual(token_f1("", "gold answer"), 0.0)


class TestCitationCoverage(unittest.TestCase):
    def test_full_coverage(self) -> None:
        self.assertEqual(citation_coverage("paris france", ["Paris is in France."]), 1.0)

    def test_partial_coverage(self) -> None:
        # 答案 4 个 token（paris/in/france/today），证据覆盖 3 个（today 未出现）→ 0.75
        # 注意：当前实现不做停用词过滤，覆盖率偏乐观（见 overlap.citation_coverage 的说明）
        self.assertAlmostEqual(citation_coverage("paris in france today", ["Paris is in France."]), 0.75)

    def test_no_evidence(self) -> None:
        self.assertEqual(citation_coverage("paris", []), 0.0)

    def test_empty_answer(self) -> None:
        self.assertEqual(citation_coverage("", ["Paris is in France."]), 0.0)


class TestOverlapFeatures(unittest.TestCase):
    def test_contract_columns(self) -> None:
        feats = overlap_features("Paris", "Paris", ["Paris is the capital of France."])
        self.assertEqual(set(feats), {"overlap_em", "overlap_f1", "citation_cov"})
        self.assertEqual(feats["overlap_em"], 1.0)
        self.assertEqual(feats["overlap_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
