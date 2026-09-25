#!/usr/bin/env python3
"""构造日志与挑战集汇总（`data/processed/construct_log.json` 的内容来源）。

负责人: A
对应文档: docs/数据构造规范.md §4、§5、§7
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Mapping, Sequence

from ..common.schema import RAGSample
from ..common.seeding import DATA_SEED_BASE


def summarize_challenge_set(samples: Sequence[RAGSample]) -> dict[str, Any]:
    """汇总挑战集：按**算子类别**分组的计数、污染/对照配对完整性、种子范围。

    这里按 `construct_params.operator` 分组（对照版的 `challenge_type="none"`，若按 challenge_type
    分组会把三类对照版混成一类）；同时给出原始 `challenge_type` 的分布以便核对。
    供 `construct_log.json` 与门禁 G3-a/G3-b 使用；本函数不做任何标签判定。
    """
    per_operator: dict[str, dict[str, int]] = {}
    type_counts: dict[str, int] = {}
    seeds: list[int] = []
    base_keys: dict[str, set[str]] = {}
    for sample in samples:
        params = dict(sample.construct_params or {})
        operator = str(params.get("operator") or sample.challenge_type)
        is_control = bool(params.get("is_control", False))
        entry = per_operator.setdefault(operator, {"polluted": 0, "control": 0,
                                                   "positive": 0, "negative": 0})
        entry["control" if is_control else "polluted"] += 1
        entry["positive" if int(sample.is_hallucination) == 1 else "negative"] += 1
        type_counts[sample.challenge_type] = type_counts.get(sample.challenge_type, 0) + 1
        if "seed" in params:
            seeds.append(int(params["seed"]))
        base_keys.setdefault(operator, set()).add(str(sample.provenance.get("base_sample_key", "")))

    return {
        "n_samples": len(samples),
        "per_operator": per_operator,
        "challenge_type_counts": type_counts,
        "n_base_per_operator": {key: len(value) for key, value in base_keys.items()},
        "pairing_complete": all(v["polluted"] == v["control"] for v in per_operator.values()),
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "n_undetermined": 0,  # 构造阶段不判定；由 derive_label 在模型答案产出后回填并统计
        "label_source": "construct_prior",
    }


def build_construct_log(samples: Sequence[RAGSample], cfg: Mapping[str, Any],
                        dataset_info: Mapping[str, Any] | None = None,
                        outputs: Mapping[str, str] | None = None) -> dict[str, Any]:
    """组装 `construct_log.json`（参数、种子、数量、产物哈希由调用方补齐）。"""
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "script": "src/datasets/challenge_builder.py",
        "dataset": dict(dataset_info or {}),
        "n_base_per_type_target": int(cfg.get("n_base_per_type", 50)),
        "data_seed_base": int(cfg.get("data_seed_base", DATA_SEED_BASE)),
        "operators": dict(cfg.get("operators") or {}),
        "outputs": dict(outputs or cfg.get("outputs") or {}),
        "summary": summarize_challenge_set(samples),
        "label_rule": "数据构造规范 §3（is_hallucination = 1 − is_supported，规则化判定）",
        "artifacts_sha256": {},  # 由 CLI 落盘后回填
    }
