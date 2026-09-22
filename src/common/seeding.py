#!/usr/bin/env python3
"""随机种子唯一入口（docs/项目启动与实施指南.md §1.5）。

纪律：实验代码禁止散落 np.random.seed / torch.manual_seed，一律调用 set_seed。
torch 为可选依赖：未安装时不报错，只记录跳过。
"""
from __future__ import annotations

import logging
import os
import random

LOGGER = logging.getLogger(__name__)

#: 全组统一的模型/训练种子
DEFAULT_SEEDS = (13, 42, 2024)
#: 数据构造种子基（与模型种子解耦，保证换种子时挑战集不变）
DATA_SEED_BASE = 1000


def set_seed(seed: int, deterministic: bool = False) -> int:
    """设置 python/numpy/torch 全部随机源，返回该种子。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        LOGGER.warning("numpy 未安装，跳过 numpy 随机种子设置")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            LOGGER.info("已开启确定性算法（可能降低性能）")
    except ImportError:
        LOGGER.warning("torch 未安装，跳过 torch 随机种子设置")

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass

    return seed


def data_seed(index: int, base: int = DATA_SEED_BASE) -> int:
    """第 index 个基样本的构造种子（独立于模型种子）。"""
    return base + index
