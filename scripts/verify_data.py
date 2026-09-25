#!/usr/bin/env python3
"""校验本地数据集副本的**完整性**与**内容准确性**，并统计内容（行数、字段、分布）。

完整性：文件存在非空并记 SHA-256；与 `data/raw/MANIFEST.sha256` 逐条比对哈希（检测损坏/截断/替换）；
与上游基线数量核对（EXPECTED_COUNTS，来源：HuggingFace 数据集卡片 + 首次校验记录）。

内容准确性：SQuAD 逐条校验 `answer_start` 与原文对齐（`context[start:start+len(text)] == text`，
答案跨度能否被信任的硬条件）；HotpotQA 读 parquet footer 并对**全量**行校验 question/answer 非空、
`context.title` 与 `context.sentences` 长度一致、句子非空、`supporting_facts.title` 必须出现在
`context.title`、`sent_id` 必须在对应段落范围内，同时统计全量 type/level 分布。

判定：结构性缺陷零容忍；上游数据本身已知的瑕疵（空句、极少数 sent_id 越界标注错误）按比例容忍
并在报告中列明（详见 TOLERATED_PROBLEMS / KNOWN_ISSUE_NOTES）。

用法:
  python scripts/verify_data.py                        # 全量校验全部文件（需 pyarrow）
  python scripts/verify_data.py --only squad            # 只校验 SQuAD（无需 pyarrow）
  python scripts/verify_data.py --sample-rows 5000      # 大文件按步长抽样校验（列级问题全覆盖，字段级抽样）
  python scripts/verify_data.py --skip-manifest         # 跳过 MANIFEST 哈希比对（首次下载、尚无清单时）

输出: 汇总 JSON 写入 `data/raw/VERIFY.json`（含 `checks` 逐项结论），明细打印到 stdout。
退出码: 0 全部通过 / 1 输入缺失（文件或清单找不到）/ 2 运行失败（如缺 pyarrow）/ 3 校验不通过
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.sha256"
REPORT = RAW / "VERIFY.json"
PROBLEM_SAMPLE_LIMIT = 5

#: 上游基线（HuggingFace 数据集卡片 + 首次校验记录）；用于断言"文件内容与上游一致"
EXPECTED_COUNTS = {
    "squad/dev-v2.0.json": {"n_questions": 11873, "n_impossible_questions": 5945},
    "squad/train-v2.0.json": {"n_questions": 130319, "n_impossible_questions": 43498},
    "hotpotqa/validation-00000-of-00001.parquet": {"n_rows_in_file": 7405, "level_all": "hard"},
}
#: HotpotQA 训练分片合计行数（两片之和，官方 distractor 配置）
EXPECTED_HOTPOT_TRAIN_ROWS = 90447

#: 上游数据**本身存在**的瑕疵：占比在阈值内视为"已知瑕疵"（不影响本项目使用，构造时会跳过），
#: 超过阈值才判 FAIL。其余问题（空问题/空答案/结构不一致/supporting_facts 指向缺失标题）一律零容忍。
#: 实测值：空句约 0.6%、sent_id 越界约 0.024%（23/97852，HotpotQA 官方标注错误）。
TOLERATED_PROBLEMS = {
    "empty_sentence": 0.02,
    "supporting_fact_sent_id_out_of_range": 0.001,
}
#: 已知瑕疵的处理方式（写进报告，便于报告/论文如实说明）
KNOWN_ISSUE_NOTES = {
    "empty_sentence": "空句仅占位，不参与句级检索；句级切块会自然忽略",
    "supporting_fact_sent_id_out_of_range": "构造挑战样本时该基样本会被跳过（build_challenge_set 已计数）",
}

def judge_hotpot_problems(problems: dict[str, int], rows_checked: int) -> tuple[bool, str]:
    """判定 HotpotQA 字段校验结果：结构性缺陷零容忍，已知上游瑕疵按比例容忍。"""
    structural = {key: value for key, value in problems.items() if key not in TOLERATED_PROBLEMS}
    over_tolerance = []
    for key, limit in TOLERATED_PROBLEMS.items():
        count = problems.get(key, 0)
        if count and rows_checked and count / rows_checked > limit:
            over_tolerance.append(f"{key}={count}（{count / rows_checked:.3%} > 上限 {limit:.3%}）")
    if structural or over_tolerance:
        detail = []
        if structural:
            detail.append(f"结构性缺陷：{structural}")
        if over_tolerance:
            detail.append(f"超出容忍上限：{over_tolerance}")
        return False, "；".join(detail)
    known = {key: problems[key] for key in TOLERATED_PROBLEMS if problems.get(key)}
    detail = f"全量校验 {rows_checked} 行，结构一致、supporting_facts 可定位"
    if known:
        detail += "；已知上游瑕疵（容忍，不影响使用）：" + ", ".join(
            f"{key}={value}（{KNOWN_ISSUE_NOTES[key]}）" for key, value in known.items())
    return True, detail

class Checks:
    """逐项校验结论收集器（同时打印明细，最终决定退出码）。"""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.items.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")

    def failed(self) -> list[dict]:
        return [item for item in self.items if not item["passed"]]

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def file_info(path: pathlib.Path) -> dict:
    return {"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}

def read_manifest(path: pathlib.Path = MANIFEST) -> dict[str, str]:
    """读 MANIFEST.sha256 -> {相对 data/raw 的路径: sha256}。"""
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        entries[rel.strip().lstrip("./")] = digest.strip()
    return entries

def summarize_problems(problems: Counter, samples: list[str]) -> str:
    if not problems:
        return "全部通过"
    parts = ", ".join(f"{key}={count}" for key, count in sorted(problems.items()))
    return f"{parts}；示例：{samples[:PROBLEM_SAMPLE_LIMIT]}"

def verify_squad(path: pathlib.Path, max_answer_checks: int = 0) -> dict:
    """结构统计 + `answer_start` 与原文对齐校验（`max_answer_checks=0` 表示全量）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    articles = payload["data"]
    paragraphs = [p for a in articles for p in a["paragraphs"]]
    questions = [qa for p in paragraphs for qa in p["qas"]]
    impossible = [qa for qa in questions if qa.get("is_impossible")]
    answers = [a for qa in questions if not qa.get("is_impossible") for a in qa["answers"]]

    problems: Counter = Counter()
    samples: list[str] = []
    checked = 0
    for para in paragraphs:
        context = para["context"]
        for qa in para["qas"]:
            if qa.get("is_impossible"):
                continue
            for answer in qa["answers"]:
                if max_answer_checks and checked >= max_answer_checks:
                    break
                checked += 1
                text, start = str(answer["text"]), int(answer["answer_start"])
                if not text.strip():
                    problems["empty_answer_text"] += 1
                elif start < 0 or context[start:start + len(text)] != text:
                    problems["answer_offset_mismatch"] += 1
                    if len(samples) < PROBLEM_SAMPLE_LIMIT:
                        samples.append(f"{qa.get('id')}@{start}")
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
        "answer_offsets_checked": checked,
        "problems": dict(problems),
        "problem_samples": samples,
    }

def verify_hotpot(path: pathlib.Path, sample_rows: int = 0, progress_every: int = 20000) -> dict:
    """parquet 元数据 + 全量（或按步长抽样）字段级校验。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    types: Counter = Counter()
    levels: Counter = Counter()
    problems: Counter = Counter()
    samples: list[str] = []
    step = max(1, total_rows // sample_rows) if sample_rows else 1
    seen = checked = 0

    columns = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    for batch in pf.iter_batches(batch_size=1000, columns=columns):
        for row in batch.to_pylist():
            index, seen = seen, seen + 1
            if (index % step) != 0:
                continue
            checked += 1
            types[row["type"]] += 1
            levels[row["level"]] += 1
            if not str(row["question"] or "").strip():
                problems["empty_question"] += 1
            if not str(row["answer"] or "").strip():
                problems["empty_answer"] += 1
            titles = list(row["context"]["title"])
            sentences = list(row["context"]["sentences"])
            if len(titles) != len(sentences):
                problems["context_title_sentences_mismatch"] += 1
            if any(not str(sentence).strip() for group in sentences for sentence in group):
                problems["empty_sentence"] += 1
            facts = row["supporting_facts"]
            fact_titles, fact_ids = list(facts["title"]), list(facts["sent_id"])
            if not fact_titles:
                problems["missing_supporting_facts"] += 1
            position = {title: i for i, title in enumerate(titles)}
            for title, sent_id in zip(fact_titles, fact_ids):
                if title not in position:
                    problems["supporting_fact_title_absent"] += 1
                elif not 0 <= int(sent_id) < len(sentences[position[title]]):
                    problems["supporting_fact_sent_id_out_of_range"] += 1
                    if len(samples) < PROBLEM_SAMPLE_LIMIT:
                        samples.append(f"{row['id']}:{title}#{sent_id}")
            if progress_every and checked % progress_every == 0:
                print(f"      …已校验 {checked}/{total_rows // step} 行（{path.name}）")

    return {
        **file_info(path),
        "format": "HotpotQA distractor parquet",
        "columns": pf.schema_arrow.names,
        "n_row_groups": pf.num_row_groups,
        "n_rows_in_file": total_rows,
        "rows_checked": checked,
        "sample_step": step,
        "type_distribution": dict(types),
        "level_distribution": dict(levels),
        "problems": dict(problems),
        "problem_samples": samples,
        "samples": [
            {"id": row["id"], "question": row["question"], "answer": row["answer"],
             "n_context": len(row["context"]["title"]),
             "n_supporting_facts": len(row["supporting_facts"]["title"])}
            for row in pf.read_row_group(0).slice(0, 3).to_pylist()
        ] if pf.num_row_groups else [],
    }

def run_checks(records: list[dict], checks: Checks, manifest: dict[str, str], skip_manifest: bool,
               full_run: bool, n_found: int) -> None:
    """把原始统计变成可判定的结论（哈希 / 结构 / 准确性 / 上游基线）。

    只对**本次所选文件**下结论：`--only` 子集校验不会因为"没校验其它文件"而判 FAIL
    （但会显式说明未覆盖范围），也不会覆盖全量报告。
    """
    selected = {pathlib.Path(r["file"]).relative_to("data/raw").as_posix() for r in records}

    if full_run:
        checks.add("文件齐全", len(records) == n_found,
                   f"校验全部 {len(records)} 个文件：{', '.join(sorted(selected))}")
    else:
        uncovered = sorted(set(manifest) - selected)
        checks.add("校验范围", True,
                   f"仅校验 {len(records)} 个文件（{'/'.join(sorted(selected))}）；"
                   f"未覆盖 {len(uncovered)} 个：{', '.join(uncovered) if uncovered else '无'}")

    check_name = "MANIFEST 哈希比对" if full_run else "MANIFEST 哈希比对（本次所选文件）"
    if skip_manifest:
        checks.add(check_name, True, "按 --skip-manifest 跳过")
    elif not manifest:
        checks.add(check_name, False,
                   f"未找到清单 {MANIFEST.relative_to(ROOT)}（首次下载可用 --skip-manifest）")
    else:
        mismatched = sorted(rel for rel in selected
                            if rel in manifest and manifest[rel] != next(
                                r["sha256"] for r in records
                                if pathlib.Path(r["file"]).relative_to("data/raw").as_posix() == rel))
        absent = sorted(rel for rel in selected if rel not in manifest)
        detail = f"{len(selected)} 个文件与清单一致"
        if mismatched:
            detail = f"哈希不一致：{mismatched}"
        if absent:
            detail += f"；清单未登记：{absent}"
        checks.add(check_name, not mismatched and not absent, detail)

    for record in records:
        rel = pathlib.Path(record["file"]).relative_to("data/raw").as_posix()
        if record["format"].startswith("SQuAD"):
            problems = record["problems"]
            checks.add(f"SQuAD {rel} · 结构统计", record["n_questions"] > 0,
                       f"{record['n_articles']} 篇文章 / {record['n_paragraphs']} 段 / "
                       f"{record['n_questions']} 问（不可回答 {record['n_impossible_questions']}）")
            checks.add(f"SQuAD {rel} · answer_start 与原文对齐",
                       not problems,
                       f"校验 {record['answer_offsets_checked']} 个答案跨度，{summarize_problems(Counter(problems), record['problem_samples'])}")
            expected = EXPECTED_COUNTS.get(rel)
            if expected:
                same = all(record.get(key) == value for key, value in expected.items())
                checks.add(f"SQuAD {rel} · 与上游基线一致", same,
                           f"期望 {expected}，实际 " + ", ".join(f"{k}={record.get(k)}" for k in expected))
        else:
            problems = dict(record["problems"])
            checks.add(f"HotpotQA {rel} · schema 与行数", bool(record["columns"]),
                       f"{record['n_rows_in_file']} 行 / {record['n_row_groups']} row groups / "
                       f"列 {record['columns']}")
            passed, detail = judge_hotpot_problems(problems, record["rows_checked"])
            checks.add(f"HotpotQA {rel} · 字段与 supporting_facts 完整性", passed,
                       f"type={record['type_distribution']}, level={record['level_distribution']}；{detail}")
            expected = EXPECTED_COUNTS.get(rel)
            if expected:
                same = all(record.get(key) == value for key, value in expected.items() if key != "level_all")
                if "level_all" in expected:
                    same = same and set(record["level_distribution"]) == {expected["level_all"]}
                checks.add(f"HotpotQA {rel} · 与上游基线一致", same,
                           f"期望 {expected}，实际行数 {record['n_rows_in_file']}，level 分布 "
                           f"{record['level_distribution']}")

    # 仅当本次确实校验了 HotpotQA **训练分片**时才核对两片合计（--only validation 等子集不适用）
    if any(r["format"].startswith("HotpotQA") and "train-" in r["file"] for r in records):
        hotpot_train = sum(r["n_rows_in_file"] for r in records
                           if r["format"].startswith("HotpotQA") and "train-" in r["file"])
        checks.add("HotpotQA 训练分片合计行数", hotpot_train == EXPECTED_HOTPOT_TRAIN_ROWS,
                   f"两片合计 {hotpot_train} 行（期望 {EXPECTED_HOTPOT_TRAIN_ROWS}）")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验本地数据集副本的完整性与内容准确性")
    parser.add_argument("--only", default=None, help="只校验路径包含该子串的文件，如 squad / hotpotqa / validation")
    parser.add_argument("--sample-rows", type=int, default=0, help="大文件按步长抽样校验的行数（0 = 全量）")
    parser.add_argument("--skip-manifest", action="store_true", help="跳过 MANIFEST.sha256 哈希比对")
    parser.add_argument("--out", default=None,
                        help="汇总 JSON 输出路径（默认：全量校验写 data/raw/VERIFY.json，"
                             "--only 子集写 data/raw/VERIFY.<only>.json，避免覆盖完整报告）")
    args = parser.parse_args(argv)

    squad_files = sorted((RAW / "squad").glob("*.json"))
    hotpot_files = sorted((RAW / "hotpotqa").glob("*.parquet"))
    all_files = [*squad_files, *hotpot_files]
    selected = [p for p in all_files if args.only is None or args.only in p.as_posix()]
    if not all_files:
        print(f"[error] 未找到任何数据集文件（{RAW}）", file=sys.stderr)
        return 1
    if not selected:
        print(f"[error] 没有匹配的文件（--only={args.only!r}，目录 {RAW}）", file=sys.stderr)
        return 1
    full_run = args.only is None

    needs_pyarrow = any(p.suffix == ".parquet" for p in selected)
    if needs_pyarrow:
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError:
            print("[error] 校验 HotpotQA 需要 pyarrow；请安装 requirements.txt 或用 --only squad",
                  file=sys.stderr)
            return 2

    checks = Checks()
    records: list[dict] = []
    for path in selected:
        print(f"--- {path.relative_to(ROOT)}  ({path.stat().st_size:,} B)")
        if path.suffix == ".json":
            records.append(verify_squad(path))
        else:
            records.append(verify_hotpot(path, sample_rows=args.sample_rows))
    for record in records:
        for key, value in record.items():
            if key in {"file", "bytes", "sha256", "samples"}:
                continue
            print(f"      {key}: {value}")

    run_checks(records, checks, read_manifest(), args.skip_manifest,
               full_run=full_run, n_found=len(all_files))

    report = {
        "datasets": records,
        "checks": checks.items,
        "tolerated_problems": TOLERATED_PROBLEMS,
        "known_issue_notes": KNOWN_ISSUE_NOTES,
        "expected_counts": EXPECTED_COUNTS,
        "expected_hotpot_train_rows": EXPECTED_HOTPOT_TRAIN_ROWS,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "summary": {
            "squad_train_questions": next((r["n_questions"] for r in records
                                           if r["file"].endswith("squad/train-v2.0.json")), None),
            "squad_dev_questions": next((r["n_questions"] for r in records
                                         if r["file"].endswith("squad/dev-v2.0.json")), None),
            "hotpot_train_rows": sum(r["n_rows_in_file"] for r in records
                                     if r["format"].startswith("HotpotQA") and "train-" in r["file"]) or None,
            "hotpot_validation_rows": next((r["n_rows_in_file"] for r in records
                                            if r["file"].endswith("validation-00000-of-00001.parquet")), None),
        },
    }
    out_path = pathlib.Path(args.out) if args.out else (
        REPORT if full_run else RAW / f"VERIFY.{args.only}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failed = checks.failed()
    print(f"\n[summary] {json.dumps(report['summary'], ensure_ascii=False)}")
    print(f"[checks] {len(checks.items) - len(failed)}/{len(checks.items)} 通过")
    for item in failed:
        print(f"[FAIL] {item['name']}  {item['detail']}")
    print(f"[done] 报告写入 {out_path}")
    return 3 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
