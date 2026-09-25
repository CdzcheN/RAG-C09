#!/usr/bin/env python3
"""提示构造与解码。

负责人: B
对应文档: docs/项目启动与实施指南.md §1.3
状态: 已实现（T1.3）

纪律: 解码统一 `do_sample=False`（贪心）——省显存，且是"同种子逐字节一致"（门禁 G1）的前提；
`max_new_tokens`、`batch_size`、模板都来自 `configs/default.yaml`，代码内不写魔法数。
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from ..common.schema import RetrievedPassage

#: 提示中的证据块格式：`[rank] title: text`（rank 便于人工对照检索结果）
EVIDENCE_LINE = "[{rank}] {title}: {text}"

#: 生成结果里可能出现的模板残留前缀
_ANSWER_PREFIX_RE = re.compile(r"^\s*(answer|答案)\s*[:：]\s*", re.IGNORECASE)


def build_prompt(question: str, passages: Sequence[RetrievedPassage], template: str) -> str:
    """按模板拼接证据与问题。

    `template` 见 `configs/default.yaml generation.prompt_template`，需含 `{evidence}` 与
    `{question}` 两个占位符；证据按检索顺序逐行给出，供模型引用与人工复核。
    """
    if "{evidence}" not in template or "{question}" not in template:
        raise ValueError("提示模板必须同时包含 {evidence} 与 {question} 占位符")
    evidence = "\n".join(
        EVIDENCE_LINE.format(rank=p.rank, title=p.title or "(无标题)", text=p.text)
        for p in passages
    )
    return template.format(evidence=evidence, question=question)


def postprocess(text: str) -> str:
    """清理生成结果：去掉模板残留前缀、只取第一段/首行（避免模型自问自答）。"""
    cleaned = (text or "").strip()
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned)
    for stop in ("\n\n", "\nQuestion:", "\nQ:", "\n问题："):
        idx = cleaned.find(stop)
        if idx > 0:
            cleaned = cleaned[:idx]
    return cleaned.split("\n")[0].strip() if cleaned else ""


def generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 128,
             do_sample: bool = False, temperature: float | None = None) -> str:
    """贪心解码生成答案（省显存且为多种子复现前提）。

    `do_sample=False` 时忽略 `temperature`（只记入产物元信息，不传给 generate，避免框架告警）。
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("生成需要 torch，请先执行 python -m pip install -r requirements.txt") from exc

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if do_sample and temperature is not None:
        kwargs["temperature"] = float(temperature)

    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    new_tokens = output[0][prompt_len:]
    return postprocess(tokenizer.decode(new_tokens, skip_special_tokens=True))
