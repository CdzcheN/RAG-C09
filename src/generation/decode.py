#!/usr/bin/env python3
"""提示构造与解码。

负责人: B
对应文档: docs/项目启动与实施指南.md §1.3
状态: 骨架（待实现）：实现要点见各函数 docstring 与对应 WBS 任务
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401

def build_prompt(question: str, passages: Sequence[RetrievedPassage], template: str) -> str:
    """按模板拼接证据与问题。

    状态: 待实现 —— T1.3：模板见 configs/default.yaml generation.prompt_template
    """
    raise NotImplementedError("T1.3：模板见 configs/default.yaml generation.prompt_template")


def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 128, do_sample: bool = False) -> str:
    """贪心解码生成答案（省显存且为多种子复现前提）。

    状态: 待实现 —— T1.3：禁止默认开启采样
    """
    raise NotImplementedError("T1.3：禁止默认开启采样")


