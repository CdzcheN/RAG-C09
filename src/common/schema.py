#!/usr/bin/env python3
"""课题 C09 中间产物 schema —— 接口契约的唯一代码来源（docs/接口契约.md §2）。

四类对象：RAGSample（样本，A）→ Prediction（模型输出，B）→ FeatureRow（特征，B）→
MetricRecord（评估，C）。字段名必须与文档逐字一致；变更走契约变更流程。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"
META_KEY = "_meta"

CHALLENGE_TYPES = ("none", "retrieval_failure", "evidence_conflict", "outdated")
SPLITS = ("normal", "challenge")

#: FeatureRow 的固定列（新增列必须走契约变更流程）
FEATURE_COLUMNS = (
    "sample_id",
    "exp_id",
    "challenge_type",
    "is_hallucination",
    "entailment_max",
    "entailment_mean",
    "contradiction_max",
    "overlap_em",
    "overlap_f1",
    "citation_cov",
    "conflict_count",
    "retrieval_top_score",
    "answer_len",
    "baseline_confidence",
)

#: 主方法使用的特征列（对照组 baseline_confidence 不参与，见实验与评估规范 §2）
METHOD_FEATURE_GROUPS = {
    "semantic": ("entailment_max", "entailment_mean", "contradiction_max"),
    "overlap": ("overlap_em", "overlap_f1", "citation_cov"),
    "conflict": ("conflict_count", "retrieval_top_score"),
}


@dataclass(frozen=True)
class RetrievedPassage:
    rank: int
    title: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RAGSample:
    """数据集样本（挑战集/常规集），由 A 产出。"""

    sample_id: str
    source: str
    split: str
    challenge_type: str
    question: str
    gold_answer: str
    gold_context: tuple[str, ...]
    passages_for_retrieval: tuple[Mapping[str, str], ...] = ()
    is_hallucination: int = 0
    construct_params: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gold_context"] = list(self.gold_context)
        d["passages_for_retrieval"] = [dict(p) for p in self.passages_for_retrieval]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RAGSample":
        return cls(
            sample_id=d["sample_id"],
            source=d["source"],
            split=d["split"],
            challenge_type=d["challenge_type"],
            question=d["question"],
            gold_answer=d["gold_answer"],
            gold_context=tuple(d.get("gold_context", ())),
            passages_for_retrieval=tuple(d.get("passages_for_retrieval", ())),
            is_hallucination=int(d.get("is_hallucination", 0)),
            construct_params=dict(d.get("construct_params", {})),
            provenance=dict(d.get("provenance", {})),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class Prediction:
    """模型输出，由 B 产出（每行一条样本）。"""

    sample_id: str
    exp_id: str
    answer: str
    passages: tuple[RetrievedPassage, ...] = ()
    prompt: str = ""
    baseline_confidence: float = float("nan")
    decode: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passages"] = [p.to_dict() if isinstance(p, RetrievedPassage) else dict(p) for p in self.passages]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Prediction":
        return cls(
            sample_id=d["sample_id"],
            exp_id=d["exp_id"],
            answer=d.get("answer", ""),
            passages=tuple(RetrievedPassage(**p) for p in d.get("passages", ())),
            prompt=d.get("prompt", ""),
            baseline_confidence=float(d.get("baseline_confidence", float("nan"))),
            decode=dict(d.get("decode", {})),
            latency_ms=float(d.get("latency_ms", float("nan"))),
        )


@dataclass(frozen=True)
class FeatureRow:
    """特征行（一行一样本），由 B 产出，落盘 parquet。"""

    sample_id: str
    exp_id: str
    challenge_type: str
    is_hallucination: int
    entailment_max: float = float("nan")
    entailment_mean: float = float("nan")
    contradiction_max: float = float("nan")
    overlap_em: float = float("nan")
    overlap_f1: float = float("nan")
    citation_cov: float = float("nan")
    conflict_count: float = float("nan")
    retrieval_top_score: float = float("nan")
    answer_len: float = float("nan")
    baseline_confidence: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in FEATURE_COLUMNS}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureRow":
        values = {k: d.get(k) for k in FEATURE_COLUMNS}
        values["is_hallucination"] = int(values["is_hallucination"])
        for k in FEATURE_COLUMNS:
            if k not in ("sample_id", "exp_id", "challenge_type", "is_hallucination") and values[k] is not None:
                values[k] = float(values[k])
        return cls(**{k: v for k, v in values.items() if v is not None})


@dataclass(frozen=True)
class MetricRecord:
    """评估结果，由 C 产出，落盘 results/metrics/<exp_id>.json。"""

    exp_id: str
    stage: str
    seed: int
    model: str
    split: str
    n_samples: int
    auc: float = float("nan")
    pr_auc: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1_macro: float = float("nan")
    f1_micro: float = float("nan")
    confusion: Mapping[str, int] = field(default_factory=dict)
    threshold: float = 0.5
    latency_ms_mean: float = float("nan")
    config_hash: str = ""
    env_report: str = ""
    timestamp: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataset_revision_stub() -> dict[str, Any]:
    """占位：数据集 revision 由 datasets 加载时填充（见 docs/接口契约.md §4）。"""
    return {"squad": None, "hotpotqa": None}
