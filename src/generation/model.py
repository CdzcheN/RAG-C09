#!/usr/bin/env python3
"""生成模型装载与显存管理。

负责人: B
对应文档: docs/项目启动与实施指南.md §1.3
状态: 已实现（T1.3）

约束: 可用显存约 3.68 GB（`configs/models.yaml vram_budget_gb`）→ fp16 装载、`batch=1`、贪心解码。
返回的 `Generator` 支持解包（`model, tokenizer = load_generator(...)`），也支持整体传递。

    from src.generation.model import load_generator
    generator = load_generator("Qwen/Qwen2.5-0.5B-Instruct")
    answer = generate(generator.model, generator.tokenizer, prompt)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ..common.logging_utils import get_logger

LOG = get_logger(__name__)

#: dtype 名称 → torch dtype（延迟求值，避免未装 torch 时导入失败）
DTYPE_NAMES = ("float16", "bfloat16", "float32")


@dataclass
class Generator:
    """生成模型与分词器的组合（便于整体传递与解包）。"""

    model: Any
    tokenizer: Any
    model_id: str = ""
    dtype: str = "float16"
    device: str = "cuda"

    def __iter__(self) -> Iterator[Any]:
        return iter((self.model, self.tokenizer))


def _torch_dtype(dtype: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("装载生成模型需要 torch，请先执行 python -m pip install -r requirements.txt") from exc
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype not in mapping:
        raise ValueError(f"未知 dtype：{dtype!r}（可选 {DTYPE_NAMES}）")
    return mapping[dtype]


def load_generator(model_id: str, dtype: str = "float16", device: str = "cuda") -> Generator:
    """装载因果语言模型与分词器（fp16、低显存策略）。batch 固定为 1，见 §1.3。

    CUDA 不可用时自动回退 CPU 并记日志（不静默改变精度：dtype 仍按配置）。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "装载生成模型需要 torch 与 transformers，请先执行 python -m pip install -r requirements.txt"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA 不可用，%s 回退到 CPU（速度会明显下降）", model_id)
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    torch_dtype = _torch_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
    except TypeError:  # transformers < 4.56 的旧参数名
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    model.eval()
    model.to(device)
    LOG.info("已装载生成模型 %s（dtype=%s, device=%s）", model_id, dtype, device)
    return Generator(model=model, tokenizer=tokenizer, model_id=model_id, dtype=dtype, device=device)


def vram_report() -> dict[str, float]:
    """报告已用/可用显存，便于提前判断 OOM 风险；CUDA 不可用时各值为 0.0。"""
    empty = {"total_gb": 0.0, "allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}
    try:
        import torch
    except ImportError:
        return empty
    if not torch.cuda.is_available():
        return empty
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "total_gb": round(total_bytes / 1024 ** 3, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
        "free_gb": round(free_bytes / 1024 ** 3, 3),
    }


def release(model: Any) -> None:
    """释放显存（生成模型与 NLI 模型分时复用的前提，见指南 §1.3）。"""
    try:
        import torch
    except ImportError:
        return
    import gc

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
