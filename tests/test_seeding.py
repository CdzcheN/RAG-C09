#!/usr/bin/env python3
"""随机种子与可复现性测试（docs/项目启动与实施指南.md §1.5）。

运行: python -m unittest tests.test_seeding -v
"""
from __future__ import annotations

import random
import unittest

from src.common.seeding import DATA_SEED_BASE, DEFAULT_SEEDS, data_seed, set_seed


class TestSetSeed(unittest.TestCase):
    def test_same_seed_same_sequence(self) -> None:
        set_seed(13)
        a = [random.random() for _ in range(5)]
        set_seed(13)
        b = [random.random() for _ in range(5)]
        self.assertEqual(a, b)

    def test_different_seed_different_sequence(self) -> None:
        set_seed(13)
        a = [random.random() for _ in range(5)]
        set_seed(14)
        b = [random.random() for _ in range(5)]
        self.assertNotEqual(a, b)

    def test_numpy_reproducible(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy 未安装")
        set_seed(42)
        a = np.random.rand(4)
        set_seed(42)
        b = np.random.rand(4)
        self.assertTrue((a == b).all())


class TestSeedProtocol(unittest.TestCase):
    def test_default_seeds(self) -> None:
        self.assertEqual(DEFAULT_SEEDS, (13, 42, 2024))

    def test_data_seed_independent_of_model_seeds(self) -> None:
        self.assertEqual(data_seed(3), DATA_SEED_BASE + 3)
        self.assertNotIn(data_seed(3), DEFAULT_SEEDS)

    def test_data_seed_stable(self) -> None:
        self.assertEqual(data_seed(17), data_seed(17))


if __name__ == "__main__":
    unittest.main()
