#!/usr/bin/env python3
"""IO 工具：原子写、JSON/JSONL、配置哈希、表格读写（parquet 可选）。

约定：先写 <name>.tmp 再 rename，避免留下半成品产物（docs/编码与协作规范.md §4）。
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .schema import META_KEY

try:  # pyarrow 可选
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:  # pragma: no cover
    HAS_PARQUET = False

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | pathlib.Path, text: str, encoding: str = "utf-8") -> pathlib.Path:
    p = pathlib.Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, p)
    return p


def write_json(path: str | pathlib.Path, obj: Any) -> pathlib.Path:
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def read_json(path: str | pathlib.Path) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | pathlib.Path, rows: Iterable[Mapping[str, Any]],
                meta: Mapping[str, Any] | None = None) -> pathlib.Path:
    """写 JSONL；若给出 meta 则作为首行 _meta（契约 §4 的强制要求）。"""
    p = pathlib.Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        if meta is not None:
            fh.write(json.dumps({META_KEY: dict(meta)}, ensure_ascii=False, default=str) + "\n")
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, p)
    return p


def iter_jsonl(path: str | pathlib.Path) -> Iterator[dict[str, Any]]:
    """逐行读取，跳过 _meta 行。"""
    with pathlib.Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if META_KEY in obj:
                continue
            yield obj


def read_jsonl_meta(path: str | pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """返回 (meta, rows)。缺失 _meta 时 meta 为空 dict。"""
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if META_KEY in obj:
                meta = dict(obj[META_KEY])
            else:
                rows.append(obj)
    return meta, rows


def write_table(path: str | pathlib.Path, rows: Sequence[Mapping[str, Any]],
                columns: Sequence[str] | None = None) -> pathlib.Path:
    """写表格：优先 parquet，未安装 pyarrow 时退化为 CSV。"""
    p = pathlib.Path(path)
    ensure_dir(p.parent)
    if HAS_PARQUET and p.suffix == ".parquet":
        table = pa.Table.from_pylist([dict(r) for r in rows])
        if columns:
            table = table.select([c for c in columns if c in table.column_names])
        tmp = p.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, p)
        return p
    header = list(columns) if columns else (list(rows[0].keys()) if rows else [])
    lines = [",".join(header)]
    lines += [",".join(str(r.get(c, "")) for c in header) for r in rows]
    return atomic_write_text(p.with_suffix(".csv"), "\n".join(lines) + "\n")


def read_table(path: str | pathlib.Path) -> list[dict[str, Any]]:
    p = pathlib.Path(path)
    if p.suffix == ".parquet":
        if not HAS_PARQUET:
            raise RuntimeError("读取 parquet 需要 pyarrow（见 requirements.txt）")
        return pq.read_table(p).to_pylist()
    import csv

    with p.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def config_hash(cfg: Mapping[str, Any], length: int = 12) -> str:
    """配置哈希（排序键、紧凑序列化），用于产物元信息。"""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("读取 YAML 需要 PyYAML（见 requirements.txt）")
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}


def build_meta(exp_id: str, seed: int, cfg: Mapping[str, Any], model: str = "",
               dataset_revision: Mapping[str, Any] | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """构造产物首行 _meta（契约 §4）。"""
    import datetime as _dt

    meta = {
        "exp_id": exp_id,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "model": model,
        "dataset_revision": dict(dataset_revision or {}),
        "env_report": "results/env_report.json",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    meta.update(extra or {})
    return meta
