#!/usr/bin/env python3
"""演示入口（复用实验代码，不另写一套逻辑）。

负责人: B
对应文档: docs/编码与协作规范.md §2.4
状态: 已实现（T6.3）

用法:
  python -m src.demo.app --question "Who wrote the song Milhouse?" [--top-k 5] [--with-features]

设计: 检索 → 提示 → 生成 → 自评置信度**全部走 `src.generation.pipeline.run_pipeline`**（把外部语料
包成一条 `RAGSample`，语料就是它的 `passages_for_retrieval`），避免"演示与实验两套逻辑"。
幻觉分数需要 C 批训练出的判别器：给出 `--detector` 或配置 `demo.detector` 时用其 `predict_proba`，
否则返回 None 并在输出中说明（不伪造分数）。
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Mapping, Sequence

from ..common import io
from ..common.logging_utils import get_logger
from ..common.schema import RAGSample
from ..datasets.reader import HOTPOTQA_CONFIG, HOTPOTQA_ID, SQUAD_ID, load_dataset_split, normalize
from ..generation.pipeline import run_pipeline
from ..generation.model import load_generator

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
LOG = get_logger(__name__)


def merge_config(cfg_path: str) -> dict[str, Any]:
    cfg = io.load_config(DEFAULT_CONFIG)
    cfg["models"] = io.load_config(ROOT / "configs" / "models.yaml")
    path = pathlib.Path(cfg_path)
    if path.resolve() != DEFAULT_CONFIG.resolve():
        cfg.update(io.load_config(path))
    return cfg


def load_corpus(cfg: Mapping[str, Any], passages: int) -> tuple[tuple[dict[str, str], ...], dict[str, Any]]:
    """取一小段外部语料（句级检索的候选段落）作为演示用的证据池。"""
    demo_cfg = dict((cfg.get("demo") or {}).get("corpus") or {})
    source = str(demo_cfg.get("source", "hotpotqa"))
    dataset_id = str(demo_cfg.get("dataset_id", HOTPOTQA_ID if source == "hotpotqa" else SQUAD_ID))
    config = demo_cfg.get("config", HOTPOTQA_CONFIG if source == "hotpotqa" else None)
    view = load_dataset_split(dataset_id, config, split=str(demo_cfg.get("split", "validation")))

    collected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row_index, row in enumerate(view):
        example = normalize(row, source, row_index=row_index, hf_split=view.split)
        for para in example.get("context") or []:
            text = " ".join(str(s) for s in (para.get("sentences") or [])).strip()
            key = (str(para.get("title", "")), text)
            if text and key not in seen:
                seen.add(key)
                collected.append({"title": key[0], "text": text})
            if len(collected) >= passages:
                break
        if len(collected) >= passages:
            break
    if not collected:
        raise ValueError("未能取到演示语料（检查数据集副本或 demo.corpus 配置）")
    return tuple(collected), view.provenance()


def _load_detector(detector_path: str | None) -> Any:
    """按路径加载已训练判别器（joblib/pickle）；失败返回 None（不伪造分数）。"""
    if not detector_path:
        return None
    import joblib  # 延迟导入：仅在使用时才需要

    path = pathlib.Path(detector_path)
    if not path.exists():
        LOG.warning("判别器文件不存在：%s", path)
        return None
    return joblib.load(path)


def answer_with_evidence(question: str, cfg_path: str = "configs/default.yaml",
                         top_k: int | None = None, detector_path: str | None = None,
                         with_features: bool = False) -> dict[str, Any]:
    """跑一次完整管道并返回答案、证据与幻觉分数，供命令行/界面展示。"""
    cfg = merge_config(cfg_path)
    seed = int((cfg.get("seeds") or [13])[0])
    demo_cfg = dict(cfg.get("demo") or {})
    retrieval_cfg = dict(cfg.get("retrieval") or {})
    if top_k is not None:
        retrieval_cfg["top_k"] = int(top_k)
        cfg["retrieval"] = retrieval_cfg

    passages, corpus_info = load_corpus(cfg, int(demo_cfg.get("corpus_passages", 500)))
    corpus_sample = RAGSample(
        sample_id="demo-corpus-00000-none", source="demo", split="normal", challenge_type="none",
        question=question, gold_answer="", gold_context=(),
        passages_for_retrieval=passages,
        construct_params={"operator": "none", "seed": seed, "is_control": True},
        provenance={"corpus": corpus_info},
    )

    model_cfg = dict((cfg.get("models") or {}).get("generator") or {})
    generator = load_generator(str((cfg.get("generation") or {}).get("model", model_cfg.get("id", ""))),
                               str(model_cfg.get("dtype", "float16")),
                               str(model_cfg.get("device", "cuda")))
    prediction = run_pipeline([corpus_sample], cfg, seed, generator=generator)[0]

    result: dict[str, Any] = {
        "question": question,
        "answer": prediction.answer,
        "passages": [p.to_dict() for p in prediction.passages],
        "prompt": prediction.prompt,
        "baseline_confidence": prediction.baseline_confidence,
        "hallucination_score": None,
        "detector": "unavailable",
        "latency_ms": prediction.latency_ms,
        "exp_id": prediction.exp_id,
        "corpus": corpus_info,
    }

    detector = _load_detector(detector_path or demo_cfg.get("detector"))
    if detector is not None:
        result["detector"] = str(detector_path or demo_cfg.get("detector"))
        result["hallucination_score"] = _score_with_detector(detector, prediction, corpus_sample, cfg,
                                                            with_features)

    return result


def _score_with_detector(detector: Any, prediction: Any, sample: RAGSample, cfg: Mapping[str, Any],
                         with_features: bool) -> float | None:
    """用判别器给单条样本打分；需要特征表时顺带算出（NLI 不可用则给 None）。"""
    try:
        from ..features.build_features import build_feature_table
        from ..features.entailment import DEFAULT_NLI_ID, load_nli

        nli_cfg = dict((cfg.get("models") or {}).get("nli") or {})
        nli = load_nli(str(nli_cfg.get("id", DEFAULT_NLI_ID)), str(nli_cfg.get("device", "cuda"))) \
            if with_features else None
        rows = build_feature_table([prediction], [sample], nli, cfg)
        if not rows:
            return None
        from ..common.schema import METHOD_FEATURE_GROUPS

        columns = [c for group in METHOD_FEATURE_GROUPS.values() for c in group]
        record = rows[0].to_dict()
        vector = [[float(record.get(c, float("nan"))) for c in columns]]
        return float(detector.predict_proba(vector)[0][1])
    except Exception as exc:  # noqa: BLE001 - 演示不应因打分失败而中断
        LOG.warning("判别器打分失败（%r），hallucination_score 记为 None", exc)
        return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RAG 幻觉检测演示：输入问题 → 证据 → 幻觉分数")
    ap.add_argument("--question", required=True, help="要提问的问题")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径（YAML）")
    ap.add_argument("--top-k", type=int, default=None, help="检索条数，缺省取配置值")
    ap.add_argument("--detector", default=None, help="已训练判别器路径（joblib），缺省读配置 demo.detector")
    ap.add_argument("--with-features", action="store_true", help="同时计算一致性特征（需要 NLI 模型）")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = answer_with_evidence(args.question, args.config, top_k=args.top_k,
                                     detector_path=args.detector, with_features=args.with_features)
    except Exception as exc:  # noqa: BLE001
        LOG.exception("演示失败：%r", exc)
        return 2

    print(f"问题：{result['question']}")
    print(f"答案：{result['answer'] or '(空)'}")
    print("证据：")
    for passage in result["passages"]:
        print(f"  [{passage['rank']}] {passage['title']} (score={passage['score']:.3f}) {passage['text'][:120]}")
    print(f"自评置信度（对照 B1）：{result['baseline_confidence']}")
    print(f"幻觉分数：{result['hallucination_score']}（判别器：{result['detector']}）")
    print(f"单条延迟：{result['latency_ms']:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
