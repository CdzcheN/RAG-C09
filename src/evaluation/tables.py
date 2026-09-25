#!/usr/bin/env python3
"""结果表导出（markdown / CSV）。

负责人: C
对应文档: docs/实验与评估规范.md §5
状态: 已实现（T6.2）

与图表同源：表格与图都从 `results/metrics/` 的数据生成，避免手抄数字出错。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _format(value: Any) -> str:
    """数值统一格式化：浮点保留 4 位，mean±std 元组按 `a ± b` 呈现。"""
    if isinstance(value, tuple) and len(value) == 2:
        mean, std = value
        try:
            return f"{float(mean):.4f} ± {float(std):.4f}"
        except (TypeError, ValueError):
            return f"{mean} ± {std}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "NaN"
        return f"{value:.4f}"
    return "" if value is None else str(value)


def to_markdown_table(rows: Sequence[Mapping[str, Any]],
                      columns: Sequence[str] | None = None,
                      combine_std: bool = True) -> str:
    """把指标汇总渲染为 markdown 表，供技术报告直接粘贴。

    `rows` 为 dict 序列（例如 `results/metrics/summary.csv` 的解析结果）；`columns` 缺省时按首行
    的键顺序。`combine_std=True` 时把 `<metric>` 与 `<metric>_std` 合并渲染为 `mean ± std` 一列
    （报告口径，规范 §4），`_std` 列不再单列。
    """
    if not rows:
        return "(无数据)"
    header = list(columns) if columns else list(rows[0].keys())
    if combine_std:
        std_only = [c for c in header if c.endswith("_std") and c[:-4] in header]
        header = [c for c in header if c not in std_only]

    formatted: list[dict[str, str]] = []
    for row in rows:
        item: dict[str, str] = {}
        for key in header:
            std = row.get(f"{key}_std")
            if combine_std and std is not None and not key.endswith("_std"):
                item[key] = _format((row.get(key), std))
            else:
                item[key] = _format(row.get(key))
        formatted.append(item)
    try:
        from tabulate import tabulate

        return tabulate(formatted, headers="keys", tablefmt="github")
    except ImportError:  # pragma: no cover
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        lines += ["| " + " | ".join(str(row.get(key, "")) for key in header) + " |"
                  for row in formatted]
        return "\n".join(lines)


def aggregate(records: Sequence[Mapping[str, Any]], group_keys: Sequence[str],
              value_keys: Sequence[str]) -> list[dict[str, Any]]:
    """按 `group_keys` 聚合多个 run 的指标，给出 mean ± std（3 种子的报告口径）。

    返回每行含分组键、`<metric>`（均值）、`<metric>_std`、`n_runs`；缺失值（NaN）不计入均值。
    """
    buckets: dict[tuple, list[Mapping[str, Any]]] = {}
    for record in records:
        buckets.setdefault(tuple(record.get(key) for key in group_keys), []).append(record)

    rows: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        group = buckets[key]
        row: dict[str, Any] = dict(zip(group_keys, key))
        row["n_runs"] = len(group)
        for metric in value_keys:
            values = [float(r[metric]) for r in group if r.get(metric) is not None and float(r[metric]) == float(r[metric])]
            if not values:
                row[metric] = float("nan")
                row[f"{metric}_std"] = float("nan")
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
            row[metric] = mean
            row[f"{metric}_std"] = variance ** 0.5
        rows.append(row)
    return rows
