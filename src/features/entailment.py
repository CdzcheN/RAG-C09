#!/usr/bin/env python3
"""NLI 蕴含特征（答案 ← 证据）。

负责人: B
对应文档: docs/实验与评估规范.md §1
状态: 已实现（T3.2）

模型: `cross-encoder/nli-deberta-v3-xsmall`（`configs/models.yaml nli`）。标签顺序从
`model.config.id2label` 解析（不同 checkpoint 顺序不同，硬编码会算反蕴含/矛盾）。
显存: 与生成模型**分时复用**（约 0.37 GB，指南 §1.3）——先跑完生成、释放，再提特征。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from ..common.logging_utils import get_logger

LOG = get_logger(__name__)

DEFAULT_NLI_ID = "cross-encoder/nli-deberta-v3-xsmall"
MAX_LENGTH = 512


@dataclass
class NLIModel:
    """NLI 交叉编码器与分词器（支持解包：`model, tokenizer = nli`）。"""

    model: Any
    tokenizer: Any
    model_id: str = DEFAULT_NLI_ID
    device: str = "cuda"

    def __iter__(self) -> Iterator[Any]:
        return iter((self.model, self.tokenizer))


def load_nli(model_id: str = DEFAULT_NLI_ID, device: str = "cuda") -> NLIModel:
    """装载交叉编码器 NLI 模型（fp16；CUDA 不可用时回退 CPU 并记日志）。"""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "装载 NLI 模型需要 torch 与 transformers，请先执行 python -m pip install -r requirements.txt"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA 不可用，NLI 模型 %s 回退到 CPU", model_id)
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float16)
    except TypeError:  # transformers < 4.56 的旧参数名
        model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()
    model.to(device)
    LOG.info("已装载 NLI 模型 %s（device=%s）", model_id, device)
    return NLIModel(model=model, tokenizer=tokenizer, model_id=model_id, device=device)


def label_indices(model: Any) -> tuple[int, int]:
    """返回 (蕴含标签下标, 矛盾标签下标)，从 `config.id2label` 解析。"""
    id2label = dict(getattr(getattr(model, "config", None), "id2label", None) or {})
    entailment = contradiction = None
    for idx, name in id2label.items():
        label = str(name).lower()
        if "entail" in label:
            entailment = int(idx)
        elif "contra" in label:
            contradiction = int(idx)
    if entailment is None or contradiction is None:
        LOG.warning("未能从 id2label=%s 解析标签顺序，回退 MNLI 约定（0=contradiction, 2=entailment）",
                    id2label)
        entailment = 2 if entailment is None else entailment
        contradiction = 0 if contradiction is None else contradiction
    return entailment, contradiction


def entailment_scores(answer: str, evidence: Sequence[str], model: Any, tokenizer: Any) -> dict[str, float]:
    """计算蕴含/矛盾概率的最大值与均值（契约列 `entailment_max` / `entailment_mean` / `contradiction_max`）。

    前提 = 每条证据句，假设 = 模型答案（逐句计算后聚合）；答案或证据为空时三个指标均为 NaN，
    由下游按契约在报告中标注缺失。
    """
    nan = float("nan")
    sentences = [str(s) for s in evidence if str(s or "").strip()]
    if not sentences or not (answer or "").strip():
        return {"entailment_max": nan, "entailment_mean": nan, "contradiction_max": nan}

    import torch

    entailment_idx, contradiction_idx = label_indices(model)
    pairs = [[sentence, answer] for sentence in sentences]
    inputs = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu()

    entailments = [float(row[entailment_idx]) for row in probs]
    contradictions = [float(row[contradiction_idx]) for row in probs]
    return {
        "entailment_max": max(entailments),
        "entailment_mean": sum(entailments) / len(entailments),
        "contradiction_max": max(contradictions),
    }
