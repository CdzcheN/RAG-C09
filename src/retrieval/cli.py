#!/usr/bin/env python3
"""src.retrieval 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法:
  python -m src.retrieval.cli --config configs/default.yaml --seed 13 [--limit 200] [--dry-run]

产出: `results/metrics/recall_at_k-<exp_id>.json` —— Recall@1/3/5（默认在 SQuAD dev 的抽样问上
评估，逐样本在自己的候选段落上检索），含 seed、config_hash、切块粒度与样本量（可追溯要求）。

退出码: 0 成功 / 1 输入错误 / 2 运行失败
"""
from __future__ import annotations

import argparse
import pathlib
import random
from typing import Any

from ..common import io
from ..common.cli_utils import add_common_args, bootstrap
from ..common.schema import RAGSample
from ..common.text import normalize
from ..datasets.reader import SQUAD_ID, load_normalized
from ..datasets.split import passages_of
from .bm25 import build_bm25, search
from .evaluate import recall_at_k_by_cutoffs
from .index import build_passage_store

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPONENT = "retrieval"


def _rel(path: pathlib.Path) -> str:
    """相对仓库根的展示用路径（`--out` 指向仓库外时退化为绝对路径）。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="构建检索索引并评估 Recall@k")
    add_common_args(ap)
    ap.add_argument("--top-k", type=int, default=None, help="检索条数，缺省取配置值")
    ap.add_argument("--granularity", default="sentence", choices=["sentence", "paragraph"],
                    help="切块粒度（影响 Recall@k 量级，需在报告中说明）")
    return ap


def _to_samples(rows: list[dict[str, Any]], seed: int, n_eval: int) -> list[RAGSample]:
    answerable = [r for r in rows if r.get("answer") and r.get("gold_context")]
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(answerable)), min(n_eval, len(answerable))))
    samples: list[RAGSample] = []
    for idx in picked:
        example = answerable[idx]
        row_index = int(example.get("row_index", -1))
        samples.append(RAGSample(
            sample_id=f"squad-dev-{row_index:05d}-none",
            source="squad", split="normal", challenge_type="none",
            question=str(example.get("question", "")),
            gold_answer=str(example.get("answer", "")),
            gold_context=tuple(str(s) for s in (example.get("gold_context") or [])),
            passages_for_retrieval=passages_of(example),
            is_hallucination=0,
            construct_params={"operator": "none", "seed": seed, "is_control": True},
            provenance={"row_index": row_index, "hf_split": example.get("hf_split", "dev")},
        ))
    return samples


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log, cfg, exp_id = bootstrap(args, component=COMPONENT)

    retrieval_cfg = dict(cfg.get("retrieval") or {})
    top_k = int(args.top_k if args.top_k is not None else retrieval_cfg.get("top_k", 5))
    n_eval = int(args.limit if args.limit is not None else 200)
    if args.dry_run:
        log.info("dry-run：配置与参数检查通过（top_k=%s, granularity=%s），未执行检索",
                 top_k, args.granularity)
        return 0

    try:
        view, rows = load_normalized(SQUAD_ID, split="validation")
        samples = _to_samples(rows, seed=int(args.seed or 13), n_eval=n_eval)
        if not samples:
            log.error("没有可评估的可回答问题，请检查数据集副本")
            return 2

        retrieved_texts: list[list[str]] = []
        gold_texts: list[set[str]] = []
        for sample in samples:
            store = build_passage_store([sample], granularity=args.granularity)
            index = build_bm25(store)
            hits = search(sample.question, index, store, top_k=top_k)
            retrieved_texts.append([normalize(hit.text) for hit in hits])
            gold_texts.append({normalize(s) for s in sample.gold_context})

        cutoffs = sorted({1, 3, top_k})
        metrics = recall_at_k_by_cutoffs(retrieved_texts, gold_texts, cutoffs)
        report = {
            "exp_id": exp_id,
            "seed": int(args.seed or 13),
            "config_hash": io.config_hash(cfg),
            "dataset": view.provenance(),
            "backend": "bm25",
            "granularity": args.granularity,
            "top_k": top_k,
            "n_queries": len(samples),
            "metrics": metrics,
        }
        out = ROOT / "results" / "metrics" / f"recall_at_k-{exp_id}.json"
        io.write_json(out, report)
        log.info("Recall@k 报告：%s（n=%d，%s）", _rel(out), len(samples),
                 ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    except (ValueError, FileNotFoundError) as exc:
        log.error("输入错误：%r", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("运行失败：%r", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
