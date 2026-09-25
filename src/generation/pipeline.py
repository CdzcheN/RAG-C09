#!/usr/bin/env python3
"""端到端管道：问题 → 检索 → 提示 → 生成 → 结构化记录。

负责人: B
对应文档: docs/项目启动与实施指南.md §2.2
状态: 已实现（T1.3）

门禁 G1: 同一种子连续两次运行，产出文件逐字节一致。保证手段——贪心解码（`do_sample=False`）、
检索同分按 `passage_id` 稳定排序、`set_seed` 唯一入口、提示模板与证据顺序完全由配置与样本决定。

用法（逐样本在自己的候选段落上检索，见 `src/retrieval/index.py` 的口径说明）:

    predictions = run_pipeline(samples, cfg, seed=13, limit=20)
    io.write_jsonl(out_path, [p.to_dict() for p in predictions], meta=meta)
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ..common.cli_utils import make_exp_id
from ..common.logging_utils import get_logger
from ..common.schema import Prediction, RAGSample
from ..common.seeding import set_seed
from ..retrieval.bm25 import build_bm25, search
from ..retrieval.index import build_passage_store
from .baseline_confidence import self_confidence
from .decode import build_prompt, generate
from .model import Generator, load_generator, vram_report

LOG = get_logger(__name__)
COMPONENT = "pipeline"

DEFAULT_PROMPT_TEMPLATE = "Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"


def generation_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """从 default.yaml + models.yaml 合并结果中取出生成参数（代码内不写魔法数）。"""
    generation = dict(cfg.get("generation") or {})
    models = dict((cfg.get("models") or {}).get("generator") or {})
    return {
        "model_id": str(generation.get("model") or models.get("id") or "Qwen/Qwen2.5-0.5B-Instruct"),
        "dtype": str(models.get("dtype") or "float16"),
        "device": str(models.get("device") or "cuda"),
        "max_new_tokens": int(generation.get("max_new_tokens", 128)),
        "do_sample": bool(generation.get("do_sample", False)),
        "temperature": float(generation.get("temperature", 1.0)),
        "template": str(generation.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE),
        "with_baseline_confidence": bool(generation.get("with_baseline_confidence", True)),
        "deterministic": bool(generation.get("deterministic", False)),
    }


def run_pipeline(samples: Sequence[RAGSample], cfg: Mapping[str, Any], seed: int,
                 limit: int | None = None, generator: Generator | None = None) -> list[Prediction]:
    """对每条样本跑完整管道，返回 Prediction 列表（含检索证据、提示、解码参数、延迟）。

    `limit` 只处理前 N 条（冒烟用）；`generator` 可复用已装载的模型（CLI 与 demo 传同一个实例，
    避免重复占用显存）。
    """
    settings = generation_settings(cfg)
    retrieval_cfg = dict(cfg.get("retrieval") or {})
    top_k = int(retrieval_cfg.get("top_k", 5))
    granularity = str(retrieval_cfg.get("granularity", "sentence"))
    exp_id = make_exp_id(str(cfg.get("exp_prefix", "exp")), COMPONENT, seed)

    # 门禁 G1：贪心解码 + 固定种子；需要严格确定性算子时由 generation.deterministic 打开
    set_seed(seed, deterministic=settings["deterministic"])
    if generator is None:
        generator = load_generator(settings["model_id"], settings["dtype"], settings["device"])
    LOG.info("管道启动 exp_id=%s seed=%s top_k=%s granularity=%s vram=%s",
             exp_id, seed, top_k, granularity, vram_report())

    decode_params = {"do_sample": settings["do_sample"],
                     "max_new_tokens": settings["max_new_tokens"],
                     "temperature": settings["temperature"]}

    todo = list(samples)[:limit] if limit is not None else list(samples)
    predictions: list[Prediction] = []
    for index, sample in enumerate(todo, start=1):
        started = time.perf_counter()
        store = build_passage_store([sample], granularity=granularity)
        index_bm25 = build_bm25(store)
        passages = search(sample.question, index_bm25, store, top_k=top_k)
        prompt = build_prompt(sample.question, passages, settings["template"])
        answer = generate(generator.model, generator.tokenizer, prompt,
                          max_new_tokens=settings["max_new_tokens"],
                          do_sample=settings["do_sample"],
                          temperature=settings["temperature"])
        confidence = (self_confidence(generator.model, generator.tokenizer, sample.question, answer)
                      if settings["with_baseline_confidence"] else float("nan"))
        latency_ms = (time.perf_counter() - started) * 1000.0

        predictions.append(Prediction(
            sample_id=sample.sample_id,
            exp_id=exp_id,
            answer=answer,
            passages=tuple(passages),
            prompt=prompt,
            baseline_confidence=confidence,
            decode=decode_params,
            latency_ms=latency_ms,
        ))
        if index % 20 == 0 or index == len(todo):
            LOG.info("进度 %d/%d（最近一条 latency=%.0f ms, 证据 %d 条）",
                     index, len(todo), latency_ms, len(passages))
    LOG.info("管道完成：%d 条预测，exp_id=%s", len(predictions), exp_id)
    return predictions
