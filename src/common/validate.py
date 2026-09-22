#!/usr/bin/env python3
"""产物校验（docs/接口契约.md §6 的规则实现）。

用法:
  python -m src.common.validate predictions results/predictions/<exp_id>.jsonl
  python -m src.common.validate features   results/features/<exp_id>.parquet
退出码: 0 通过；3 校验不通过
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Iterable, Mapping

from .io import read_jsonl_meta, read_table
from .schema import CHALLENGE_TYPES, FEATURE_COLUMNS, SPLITS

REQUIRED_PREDICTION_FIELDS = ("sample_id", "exp_id", "answer")
PROB_FIELDS = ("baseline_confidence",)
MAX_MISSING_RATE = 0.20


def _problems_predictions(rows: Iterable[Mapping[str, Any]], meta: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not meta:
        problems.append("缺少首行 _meta（契约 §4）")
    else:
        for key in ("exp_id", "seed", "config_hash"):
            if key not in meta:
                problems.append(f"_meta 缺少字段 {key}")

    seen: set[str] = set()
    for i, row in enumerate(rows, start=1):
        for f in REQUIRED_PREDICTION_FIELDS:
            if f not in row:
                problems.append(f"第 {i} 行缺少必需字段 {f}")
        sid = row.get("sample_id", "")
        if sid in seen:
            problems.append(f"sample_id 重复：{sid}")
        seen.add(sid)
        for f in PROB_FIELDS:
            v = row.get(f)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                problems.append(f"第 {i} 行 {f} 不是数值：{v!r}")
                continue
            if fv == fv and not (0.0 <= fv <= 1.0):  # 排除 NaN
                problems.append(f"第 {i} 行 {f}={fv} 超出 [0,1]")
        lat = row.get("latency_ms")
        if lat is not None:
            try:
                if float(lat) <= 0:
                    problems.append(f"第 {i} 行 latency_ms 应 > 0")
            except (TypeError, ValueError):
                problems.append(f"第 {i} 行 latency_ms 不是数值")
    return problems


def _problems_features(rows: list[Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    if not rows:
        return ["特征表为空"]
    extra = set(rows[0].keys()) - set(FEATURE_COLUMNS)
    if extra:
        problems.append(f"出现未登记的列：{sorted(extra)}（新增列需走契约变更流程）")
    n = len(rows)
    for col in FEATURE_COLUMNS:
        missing = sum(1 for r in rows if r.get(col) is None or r.get(col) == "")
        if missing and col not in ("sample_id", "exp_id", "challenge_type") and missing / n > MAX_MISSING_RATE:
            problems.append(f"列 {col} 缺失率 {missing / n:.0%} 超过 {MAX_MISSING_RATE:.0%}")
    for r in rows:
        if r.get("challenge_type") not in CHALLENGE_TYPES:
            problems.append(f"非法 challenge_type：{r.get('challenge_type')!r}（合法值 {CHALLENGE_TYPES}）")
            break
        if int(r.get("is_hallucination", -1)) not in (0, 1):
            problems.append(f"is_hallucination 应为 0/1，出现 {r.get('is_hallucination')!r}")
            break
    return problems


def validate_file(kind: str, path: str | pathlib.Path) -> list[str]:
    p = pathlib.Path(path)
    if not p.exists():
        return [f"文件不存在：{p}"]
    if kind == "predictions":
        meta, rows = read_jsonl_meta(p)
        return _problems_predictions(rows, meta)
    if kind == "features":
        return _problems_features(read_table(p))
    if kind == "samples":
        meta, rows = read_jsonl_meta(p)
        problems = [] if meta else ["缺少首行 _meta"]
        for r in rows:
            if r.get("split") not in SPLITS:
                problems.append(f"非法 split：{r.get('split')!r}")
                break
        return problems
    return [f"未知的产物类型：{kind}"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="校验中间产物是否符合接口契约")
    ap.add_argument("kind", choices=["predictions", "features", "samples"])
    ap.add_argument("path")
    args = ap.parse_args(argv)

    problems = validate_file(args.kind, args.path)
    if problems:
        print(f"[FAIL] {args.path}：{len(problems)} 条问题")
        for x in problems[:50]:
            print(f"  - {x}")
        return 3
    print(f"[OK] {args.path} 通过校验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
