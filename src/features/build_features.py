#!/usr/bin/env python3
"""特征表构建与落盘（features.parquet）。

负责人: B
对应文档: docs/接口契约.md §2.3、§6
状态: 已实现（T3.3）

串联三组特征（语义蕴含 / 词面重叠 / 证据冲突）→ 契约固定列 `schema.FEATURE_COLUMNS`。
`baseline_confidence` 仅为对照 B1 记录，**不参与主方法特征组**（`METHOD_FEATURE_GROUPS` 不含它）。

标签回填（已与全组确认的口径）: A 阶段写的是构造先验；这里用
`challenge_builder.derive_label(模型答案, ...)` 回填真实标签；返回 None（undetermined）的样本
默认从特征表剔除，数量通过 `stats` 上报并在报告中说明（数据构造规范 §3）。
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import FeatureRow, Prediction, RAGSample
from ..common.text import tokenize
from ..datasets.challenge_builder import derive_label
from .consistency import conflict_features
from .entailment import entailment_scores
from .overlap import overlap_features

LOG = get_logger(__name__)

UNDETERMINED = "undetermined"


def _label_for(prediction: Prediction, sample: RAGSample) -> int | None:
    """按规范 §3 的规则回填标签；无法判定返回 None（由调用方剔除并计数）。"""
    params = dict(sample.construct_params or {})
    injected = [str(v) for v in (params.get("injected_values") or [])]
    if not injected:
        injected = [str(v) for v in (params.get("injections") or []) if isinstance(v, str)]
    return derive_label(prediction.answer, sample.gold_answer, injected,
                        sample.challenge_type, is_control=bool(params.get("is_control", False)))


def build_feature_table(predictions: Sequence[Prediction], samples: Sequence[RAGSample], nli: Any,
                        cfg: Mapping[str, Any] | None = None,
                        stats: MutableMapping[str, Any] | None = None) -> list[FeatureRow]:
    """串联三组特征，产出契约固定列的 FeatureRow 列表。

    `nli` 为 `NLIModel` 或 None（None 时语义/冲突特征记 NaN，便于无 NLI 依赖时打通链路）。
    `stats` 若给出则写入 `n_predictions` / `n_rows` / `n_undetermined` / `n_missing_sample`。
    落盘后必须通过 `python -m src.common.validate features <path>`（契约 §6）。
    """
    settings = dict((cfg or {}).get("features") or {})
    drop_undetermined = bool(settings.get("drop_undetermined", True))
    sample_by_id = {sample.sample_id: sample for sample in samples}

    rows: list[FeatureRow] = []
    n_undetermined = n_missing = 0
    for prediction in predictions:
        sample = sample_by_id.get(prediction.sample_id)
        if sample is None:
            n_missing += 1
            LOG.warning("预测 %s 找不到对应样本，跳过", prediction.sample_id)
            continue

        label = _label_for(prediction, sample)
        if label is None:
            n_undetermined += 1
            if drop_undetermined:
                continue
            label = 1  # 仅在显式关闭剔除时才会走到这里；报告中须说明该处理

        evidence = [str(p.text) for p in prediction.passages]
        scores = entailment_scores(prediction.answer, evidence, nli.model, nli.tokenizer) if nli else \
            {"entailment_max": float("nan"), "entailment_mean": float("nan"),
             "contradiction_max": float("nan")}
        overlap = overlap_features(prediction.answer, sample.gold_answer, evidence)
        conflict = conflict_features(list(prediction.passages), nli)

        rows.append(FeatureRow(
            sample_id=sample.sample_id,
            exp_id=prediction.exp_id,
            challenge_type=sample.challenge_type,
            is_hallucination=label,
            entailment_max=scores["entailment_max"],
            entailment_mean=scores["entailment_mean"],
            contradiction_max=scores["contradiction_max"],
            overlap_em=overlap["overlap_em"],
            overlap_f1=overlap["overlap_f1"],
            citation_cov=overlap["citation_cov"],
            conflict_count=conflict["conflict_count"],
            retrieval_top_score=conflict["retrieval_top_score"],
            answer_len=float(len(tokenize(prediction.answer))),
            baseline_confidence=float(prediction.baseline_confidence),
        ))

    if stats is not None:
        stats.update({"n_predictions": len(predictions), "n_rows": len(rows),
                      "n_undetermined": n_undetermined, "n_missing_sample": n_missing,
                      "drop_undetermined": drop_undetermined})
    if n_undetermined:
        LOG.info("标签回填：%d 条 undetermined 样本%s（数据构造规范 §3）",
                 n_undetermined, "已剔除" if drop_undetermined else "按正类保留")
    return rows
