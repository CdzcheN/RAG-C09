#!/usr/bin/env python3
"""src.datasets 命令行入口（统一参数见 docs/编码与协作规范.md §2）。

用法:
  python -m src.datasets.cli --config configs/challenge.yaml --seed 1000 [--limit 20] [--dry-run]

产出（契约 §3）:
  data/processed/challenge_set.jsonl    挑战集（三类各 100 例：50 基样本 × 污染/对照）
  data/processed/normal_set.jsonl       常规集（SQuAD dev + HotpotQA validation 各 300 条）
  data/processed/construct_log.json     构造日志（参数、种子、数量、产物 SHA-256）
  data/processed/split_dev_fit.jsonl    按基样本 7:3 划分的训练侧
  data/processed/split_dev_test.jsonl   报告侧
  results/metrics/construct_audit.csv   人工抽检表（G3-b 用，人工标签列留空）

退出码: 0 成功 / 1 输入错误 / 2 运行失败 / 3 产物校验不通过
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import random
import subprocess
import sys
from typing import Any

from ..common import io, validate
from ..common.cli_utils import add_common_args, make_exp_id
from ..common.logging_utils import get_logger
from ..common.schema import SCHEMA_VERSION, RAGSample
from .challenge_builder import build_challenge_set
from .construct_log import build_construct_log
from .split import sample_normal_set, split_by_base_sample

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
COMPONENT = "challenge"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="构造挑战集与常规集")
    add_common_args(ap)
    ap.add_argument("--audit-out", default=None, help="抽检表输出路径")
    ap.add_argument("--audit-size", type=int, default=100, help="抽检样本条数（G3-b）")
    return ap


def merge_config(config_path: str) -> dict[str, Any]:
    """顶层合并 `configs/default.yaml` 与指定配置（挑战参数文件提供算子与输出路径）。

    只做顶层键覆盖：`dataset`/`sampling` 等公共段来自 default.yaml，`operators`/`n_base_per_type`
    等来自 challenge.yaml。
    """
    default_path = pathlib.Path(config_path)
    cfg = io.load_config(DEFAULT_CONFIG)
    if default_path.resolve() != DEFAULT_CONFIG.resolve():
        cfg.update(io.load_config(default_path))
    return cfg


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _rel(path: pathlib.Path) -> str:
    """相对仓库根的展示用路径；`--out` 指向仓库外时退化为绝对路径（不报错）。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_info(samples: list[RAGSample]) -> dict[str, Any]:
    for sample in samples:
        info = dict(sample.provenance or {}).get("dataset")
        if info:
            return dict(info)
    return {}


def _audit_rows(samples: list[RAGSample], size: int, seed: int) -> list[dict[str, Any]]:
    """按 challenge_type 比例抽样生成人工抽检表（人工标签列留空，由非作者填写）。"""
    rng = random.Random(seed)
    buckets: dict[str, list[RAGSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.challenge_type, []).append(sample)
    rows: list[dict[str, Any]] = []
    for challenge_type in sorted(buckets):
        bucket = sorted(buckets[challenge_type], key=lambda s: s.sample_id)
        quota = max(1, round(size * len(bucket) / max(1, len(samples))))
        for sample in rng.sample(bucket, min(quota, len(bucket))):
            params = dict(sample.construct_params or {})
            rows.append({
                "sample_id": sample.sample_id,
                "challenge_type": sample.challenge_type,
                "is_control": bool(params.get("is_control", False)),
                "construct_seed": params.get("seed", ""),
                "auto_label_prior": sample.is_hallucination,
                "human_label": "",
                "agreement": "",
                "disagreement_reason": "",
            })
    rows.sort(key=lambda r: str(r["sample_id"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = merge_config(args.config)
    except Exception as exc:  # noqa: BLE001 - 配置问题属输入错误
        print(f"[FAIL] 配置读取失败：{exc!r}", file=sys.stderr)
        return 1

    seed = int(args.seed if args.seed is not None else (cfg.get("seeds") or [13])[0])
    exp_id = args.exp_id or make_exp_id(str(cfg.get("exp_prefix", "exp")), COMPONENT, seed)
    log = get_logger(f"src.{COMPONENT}", exp_id=exp_id)
    log.info("component=%s exp_id=%s seed=%s limit=%s dry_run=%s",
             COMPONENT, exp_id, seed, args.limit, args.dry_run)

    outputs = dict(cfg.get("outputs") or {})
    if args.limit is not None:  # 冒烟：每类样本条数上限 → 基样本数 = 条数 // 2
        cfg["n_base_per_type"] = max(1, int(args.limit) // 2)
        sampling = dict(cfg.get("sampling") or {})
        sampling["normal_per_dataset"] = max(1, int(args.limit))
        cfg["sampling"] = sampling
        log.info("--limit=%s → n_base_per_type=%s，normal_per_dataset=%s",
                 args.limit, cfg["n_base_per_type"], sampling["normal_per_dataset"])

    if args.dry_run:
        log.info("dry-run：配置与参数检查通过，未执行计算（n_base_per_type=%s）",
                 cfg.get("n_base_per_type"))
        return 0

    out_dir = pathlib.Path(args.out) if args.out else ROOT / "data" / "processed"
    challenge_path = out_dir / pathlib.Path(outputs.get("challenge", "data/processed/challenge_set.jsonl")).name
    normal_path = out_dir / pathlib.Path(outputs.get("normal", "data/processed/normal_set.jsonl")).name
    log_path = out_dir / pathlib.Path(outputs.get("construct_log", "data/processed/construct_log.json")).name
    audit_path = pathlib.Path(args.audit_out) if args.audit_out else ROOT / str(
        outputs.get("audit", "results/metrics/construct_audit.csv"))

    try:
        challenge = build_challenge_set(cfg)
        if not challenge:
            log.error("挑战集为空：请检查数据集副本与算子参数")
            return 2
        dataset_info = _dataset_info(challenge)
        meta = io.build_meta(exp_id, seed, cfg, model="n/a", dataset_revision=dataset_info,
                             extra={"schema_version": SCHEMA_VERSION, "git_commit": _git_commit(),
                                    "n_samples": len(challenge)})
        io.write_jsonl(challenge_path, [s.to_dict() for s in challenge], meta=meta)

        normal = sample_normal_set(cfg, seed)
        normal_meta = dict(meta)
        normal_meta["n_samples"] = len(normal)
        io.write_jsonl(normal_path, [s.to_dict() for s in normal], meta=normal_meta)

        ratio = dict((cfg.get("sampling") or {}).get("split_ratio") or {"dev_fit": 0.7, "dev_test": 0.3})
        dev_fit, dev_test = split_by_base_sample(challenge, ratio, seed)
        fit_path = out_dir / "split_dev_fit.jsonl"
        test_path = out_dir / "split_dev_test.jsonl"
        io.write_jsonl(fit_path, [s.to_dict() for s in dev_fit], meta={**meta, "n_samples": len(dev_fit),
                                                                       "side": "dev_fit"})
        io.write_jsonl(test_path, [s.to_dict() for s in dev_test], meta={**meta, "n_samples": len(dev_test),
                                                                         "side": "dev_test"})

        audit_rows = _audit_rows(challenge, args.audit_size, seed)
        io.write_table(audit_path, audit_rows,
                       columns=["sample_id", "challenge_type", "is_control", "construct_seed",
                                "auto_label_prior", "human_label", "agreement", "disagreement_reason"])

        construct_log = build_construct_log(challenge, cfg, dataset_info=dataset_info)
        construct_log["artifacts_sha256"] = {
            _rel(p): _sha256(p)
            for p in (challenge_path, normal_path, fit_path, test_path, audit_path)
        }
        construct_log["outputs"] = {**construct_log.get("outputs", {}),
                                   "split_dev_fit": _rel(fit_path),
                                   "split_dev_test": _rel(test_path)}
        construct_log["env_report"] = "results/env_report.json"
        construct_log["config_hash"] = meta["config_hash"]
        construct_log["exp_id"] = exp_id
        construct_log["seed"] = seed
        io.write_json(log_path, construct_log)

        log.info("挑战集 %d 条 → %s；常规集 %d 条 → %s；划分 %d/%d；构造日志 → %s",
                 len(challenge), _rel(challenge_path), len(normal),
                 _rel(normal_path), len(dev_fit), len(dev_test),
                 _rel(log_path))
        log.info("抽检表 %d 行 → %s（人工标签列留空，交 B/C 复核）",
                 len(audit_rows), _rel(audit_path))
    except (ValueError, FileNotFoundError) as exc:
        log.error("输入错误：%r", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("运行失败：%r", exc)
        return 2

    problems: list[str] = []
    for kind, path in (("samples", challenge_path), ("samples", normal_path)):
        problems += [f"{path.name}: {p}" for p in validate.validate_file(kind, path)]
    if problems:
        for problem in problems[:20]:
            log.error("产物校验：%s", problem)
        return 3
    log.info("产物校验通过（challenge_set / normal_set）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
