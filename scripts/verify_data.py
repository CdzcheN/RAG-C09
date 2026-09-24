#!/usr/bin/env python3
"""校验已下载数据集的完整性，并统计内容（行数、字段、标签分布）。

校验内容:
  1. 每个文件存在且非空，并记录 SHA-256 与字节数；
  2. SQuAD v2.0  : 能按 UTF-8 JSON 解析，统计段落/问题/不可回答样本数；
  3. HotpotQA    : 能作为 parquet 读取（footer 元数据 + 抽样），统计行数与分布。

用法: python scripts/verify_data.py   （在 conda 环境 rag-c09 中执行，见 docs/项目启动与实施指南.md §1.1）
输出: 汇总 JSON 写入 data/raw/VERIFY.json，明细打印到 stdout
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: pathlib.Path) -> dict:
    return {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_squad(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 结构: data[] (article) -> paragraphs[] -> qas[]
    articles = payload["data"]
    paragraphs = [p for a in articles for p in a["paragraphs"]]
    questions = [qa for p in paragraphs for qa in p["qas"]]
    impossible = [qa for qa in questions if qa.get("is_impossible")]
    answers = [a["text"].strip() for qa in questions if not qa.get("is_impossible") for a in qa["answers"]]
    return {
        **file_info(path),
        "format": "SQuAD v2.0 JSON",
        "squad_version": payload.get("version"),
        "n_articles": len(articles),
        "n_paragraphs": len(paragraphs),
        "n_questions": len(questions),
        "n_impossible_questions": len(impossible),
        "n_answerable_questions": len(questions) - len(impossible),
        "n_answer_spans": len(answers),
    }


def verify_hotpot(path: pathlib.Path, sample_rows: int = 3) -> dict:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    table = pf.read_row_group(0) if pf.num_row_groups else pf.read()
    types = Counter(table.column("type").to_pylist())
    levels = Counter(table.column("level").to_pylist())
    heads = [
        {
            "id": table.column("id")[i].as_py(),
            "question": table.column("question")[i].as_py(),
            "answer": table.column("answer")[i].as_py(),
            "n_context": len(table.column("context")[i].as_py()["title"]),
            "n_supporting_facts": len(table.column("supporting_facts")[i].as_py()["title"]),
        }
        for i in range(min(sample_rows, table.num_rows))
    ]
    return {
        **file_info(path),
        "format": "HotpotQA distractor parquet",
        "columns": pf.schema_arrow.names,
        "n_row_groups": pf.num_row_groups,
        # 该文件的行数（校验分片完整性时与全量行数区分开）
        "n_rows_in_file": pf.metadata.num_rows,
        "n_rows_in_row_group_0": table.num_rows,
        "type_distribution_rowgroup0": dict(types),
        "level_distribution_rowgroup0": dict(levels),
        "samples": heads,
    }


def main() -> int:
    report: dict = {"datasets": []}

    squad_files = sorted((RAW / "squad").glob("*.json"))
    if not squad_files:
        print("[error] 未找到 SQuAD 文件", file=sys.stderr)
        return 1
    for f in squad_files:
        report["datasets"].append(verify_squad(f))

    hotpot_files = sorted((RAW / "hotpotqa").glob("*.parquet"))
    if not hotpot_files:
        print("[error] 未找到 HotpotQA 文件", file=sys.stderr)
        return 1
    for f in hotpot_files:
        report["datasets"].append(verify_hotpot(f))

    # 跨分片汇总 HotpotQA 训练集行数
    train_rows = sum(
        d["n_rows_in_file"]
        for d in report["datasets"]
        if d["file"].endswith(".parquet") and "train-" in d["file"]
    )
    report["summary"] = {
        "squad_train_questions": next(
            d["n_questions"] for d in report["datasets"] if d["file"].endswith("squad/train-v2.0.json")
        ),
        "squad_dev_questions": next(
            d["n_questions"] for d in report["datasets"] if d["file"].endswith("squad/dev-v2.0.json")
        ),
        "hotpot_train_rows": train_rows,
        "hotpot_validation_rows": next(
            d["n_rows_in_file"]
            for d in report["datasets"]
            if d["file"].endswith("validation-00000-of-00001.parquet")
        ),
    }

    (RAW / "VERIFY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for d in report["datasets"]:
        print(f"--- {d['file']}  ({d['bytes']:,} B)  {d['sha256'][:16]}…")
        for k, v in d.items():
            if k in {"file", "bytes", "sha256", "samples"}:
                continue
            print(f"      {k}: {v}")
    print("\n[summary]", json.dumps(report["summary"], ensure_ascii=False))
    print(f"[done] 报告写入 {RAW / 'VERIFY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
