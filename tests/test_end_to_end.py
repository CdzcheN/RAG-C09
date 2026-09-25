#!/usr/bin/env python3
"""端到端集成测试：A（数据构造）→ B（特征）→ C（判别与评估）链路，不加载真实生成模型。

覆盖:
1. 三类退化算子 + 对照版构造，契约字段与命名规则；
2. 标签规则化推导（数据构造规范 §3 全部分支）；
3. 按基样本划分：污染版与对照版成对同侧；
4. 特征表：标签回填、undetermined 剔除计数、列名等于 FEATURE_COLUMNS，落盘后过 validate；
5. 判别链路（需 scikit-learn）：5 折分组 CV → 预测 → 校准 → 分组指标 → 配对检验 → 汇总表；
6. 关键图（需 matplotlib）；
7. 全部 CLI 的 --dry-run 可跑通。

运行: python -m unittest tests.test_end_to_end -v
说明: 需要重依赖（sklearn / scipy / matplotlib）的用例在缺依赖时自动 skip，不视为失败。
"""
from __future__ import annotations

import importlib.util
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import io, validate  # noqa: E402
from src.common.schema import (FEATURE_COLUMNS, METHOD_FEATURE_GROUPS, FeatureRow,  # noqa: E402
                               Prediction, RAGSample)
from src.datasets.challenge_builder import (apply_evidence_conflict, apply_outdated,  # noqa: E402
                                            apply_retrieval_failure, derive_label, make_control)
from src.datasets.reader import normalize_hotpotqa  # noqa: E402
from src.datasets.split import base_key, split_by_base_sample  # noqa: E402
from src.features.build_features import build_feature_table  # noqa: E402

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
LOG_DIR = ROOT / "results" / "logs"


def hotpotqa_example(index: int) -> dict:
    """HotpotQA 原始字段格式的合成样本（6 个段落，支撑句含年份，便于三类算子都可用）。"""
    titles = [f"Topic{index}Part{j}" for j in range(6)]
    return {
        "id": f"row{index}",
        "question": f"In which year was {titles[0]} founded? (q{index})",
        "answer": "1990",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": [titles[0]], "sent_id": [0]},
        "context": {"title": titles,
                    "sentences": [[f"The {titles[j]} article started in {1990 + j}.",
                                   f"Its record was {100 + j} units."] for j in range(6)]},
    }


def build_pipeline_data(n_base: int = 4) -> tuple[list[RAGSample], list[Prediction]]:
    """构造「样本 + 对应预测」：覆盖三类污染版、对照版，以及一条 undetermined。"""
    samples: list[RAGSample] = []
    predictions: list[Prediction] = []
    for index in range(n_base):
        example = normalize_hotpotqa(hotpotqa_example(index))
        seed = 1000 + index
        pool = [{"title": f"Pool{index}-{j}", "text": f"Pool{index} passage {j} about Topic{index}."}
                for j in range(3)]
        attacks = {
            "retrieval_failure": apply_retrieval_failure(
                example, {"n_replace": 2, "distractor_pool": "native", "min_lexical_overlap": 0.0,
                          "_pool": pool}, seed=seed),
            "evidence_conflict": apply_evidence_conflict(
                example, {"slot_type": ["date", "number", "person", "place"], "n_inject": 1,
                          "inject_position": "middle"}, seed=seed),
            "outdated": apply_outdated(example, {"time_gap_years": 2}, seed=seed),
        }
        for challenge_type, result in attacks.items():
            if result is None:
                continue
            params = dict(result["construct_params"])
            injected = list(result["injected_values"])
            sample_id = f"hotpotqa-dev-{index:05d}-{challenge_type}"
            samples.append(RAGSample(
                sample_id=sample_id, source="hotpotqa", split="challenge",
                challenge_type=challenge_type, question=example["question"],
                gold_answer=example["answer"], gold_context=tuple(example["gold_context"]),
                passages_for_retrieval=tuple(result["passages_for_retrieval"]),
                is_hallucination=int(result["is_hallucination"]),
                construct_params={**params, "is_control": False, "injected_values": injected},
                provenance={**example, "base_sample_key": f"hotpotqa-dev-{index:05d}"}))
            answer = injected[0] if injected else "I don't know"
            predictions.append(Prediction(sample_id=sample_id, exp_id="e2e-13", answer=answer,
                                          baseline_confidence=0.6, latency_ms=8.0, prompt="p"))
        control = make_control(example, seed=seed)
        control_id = f"hotpotqa-dev-{index:05d}-none"
        samples.append(RAGSample(
            sample_id=control_id, source="hotpotqa", split="challenge", challenge_type="none",
            question=example["question"], gold_answer=example["answer"],
            gold_context=tuple(example["gold_context"]),
            passages_for_retrieval=tuple(control["passages_for_retrieval"]), is_hallucination=0,
            construct_params={**control["construct_params"], "injected_values": []},
            provenance={**example, "base_sample_key": f"hotpotqa-dev-{index:05d}"}))
        predictions.append(Prediction(sample_id=control_id, exp_id="e2e-13",
                                      answer=example["answer"], baseline_confidence=0.9,
                                      latency_ms=6.0, prompt="p"))
    # 再补一条独立样本「答案既不匹配 gold 也不匹配注入值」→ 应为 undetermined 并被剔除
    extra_index = 99
    extra_example = normalize_hotpotqa(hotpotqa_example(extra_index))
    extra_conflict = apply_evidence_conflict(
        extra_example, {"slot_type": ["date", "number", "person", "place"], "n_inject": 1,
                        "inject_position": "middle"}, seed=1000 + extra_index)
    if extra_conflict is not None:
        extra_id = f"hotpotqa-dev-{extra_index:05d}-evidence_conflict"
        samples.append(RAGSample(
            sample_id=extra_id, source="hotpotqa", split="challenge",
            challenge_type="evidence_conflict", question=extra_example["question"],
            gold_answer=extra_example["answer"],
            gold_context=tuple(extra_example["gold_context"]),
            passages_for_retrieval=tuple(extra_conflict["passages_for_retrieval"]),
            is_hallucination=1,
            construct_params={**dict(extra_conflict["construct_params"]), "is_control": False,
                              "injected_values": list(extra_conflict["injected_values"])},
            provenance={**extra_example, "base_sample_key": f"hotpotqa-dev-{extra_index:05d}"}))
        predictions.append(Prediction(sample_id=extra_id, exp_id="e2e-13", answer="banana republic",
                                      baseline_confidence=0.2, latency_ms=5.0, prompt="p"))
    return samples, predictions


class TestConstructionAndLabels(unittest.TestCase):
    """A 段：算子构造、标签规则、按基样本划分。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, _ = build_pipeline_data(n_base=3)

    def test_three_operators_and_controls(self) -> None:
        types = {s.challenge_type for s in self.samples}
        self.assertLessEqual({"retrieval_failure", "evidence_conflict", "outdated", "none"}, types)
        self.assertTrue(all(s.split == "challenge" and s.gold_answer for s in self.samples))
        self.assertTrue(all(s.gold_context for s in self.samples))
        self.assertTrue(all(s.passages_for_retrieval for s in self.samples))

    def test_sample_id_naming_and_uniqueness(self) -> None:
        ids = [s.sample_id for s in self.samples]
        self.assertEqual(len(ids), len(set(ids)))
        for sample in self.samples:
            parts = sample.sample_id.split("-")
            self.assertEqual(len(parts), 4, sample.sample_id)
            self.assertEqual(parts[1], "dev")
            self.assertTrue(sample.sample_id.endswith(f"-{sample.challenge_type}"))

    def test_polluted_and_control_are_paired(self) -> None:
        keys: dict[str, set[str]] = {}
        for sample in self.samples:
            keys.setdefault(sample.provenance["base_sample_key"], set()).add(sample.challenge_type)
        paired = {key: types for key, types in keys.items() if "none" in types}
        self.assertTrue(paired, "应至少有一组「污染版 + 对照版」配对")
        for key, types in paired.items():
            self.assertTrue(types - {"none"}, key)
        unpaired = {key for key, types in keys.items() if "none" not in types}
        # 唯一例外：专为 undetermined 计数追加的单侧样本（有意为之，见 build_pipeline_data）
        self.assertEqual(unpaired, {"hotpotqa-dev-00099"})

    def test_retrieval_failure_removes_gold_sentences(self) -> None:
        sample = next(s for s in self.samples if s.challenge_type == "retrieval_failure")
        corpus = " ".join(p["text"] for p in sample.passages_for_retrieval)
        self.assertFalse(any(gold in corpus for gold in sample.gold_context))
        self.assertTrue(sample.construct_params["removed_from_corpus"])

    def test_outdated_rewrites_year(self) -> None:
        sample = next(s for s in self.samples if s.challenge_type == "outdated")
        params = sample.construct_params
        corpus = " ".join(p["text"] for p in sample.passages_for_retrieval)
        self.assertIn(params["outdated_version"], corpus)
        self.assertNotIn(params["gold_version"], corpus)

    def test_derive_label_rules(self) -> None:
        cases = [
            (("I don't know", "1990", (), "retrieval_failure", False), 0),
            (("France", "France", (), "retrieval_failure", False), 1),
            (("in 1985", "1990", ("1985",), "evidence_conflict", False), 1),
            (("1990", "1990", ("1985",), "evidence_conflict", False), 0),
            (("nonsense", "1990", ("1985",), "evidence_conflict", False), None),
            (("1985", "1990", ("1985",), "outdated", False), 1),
            (("1990", "1990", (), "outdated", False), 0),
            (("1990", "1990", (), "retrieval_failure", True), 0),
            (("anything", "1990", (), "retrieval_failure", True), 1),
            (("无法回答", "1990", (), "retrieval_failure", False), 0),
        ]
        for (answer, gold, injected, challenge_type, is_control), expected in cases:
            with self.subTest(answer=answer, challenge_type=challenge_type, control=is_control):
                self.assertEqual(derive_label(answer, gold, injected, challenge_type, is_control),
                                 expected)

    def test_split_keeps_pairs_on_same_side(self) -> None:
        fit, test = split_by_base_sample(self.samples, {"dev_fit": 0.7, "dev_test": 0.3}, seed=13)
        fit_keys, test_keys = {base_key(s) for s in fit}, {base_key(s) for s in test}
        self.assertFalse(fit_keys & test_keys)
        self.assertEqual(len(fit) + len(test), len(self.samples))
        again, _ = split_by_base_sample(self.samples, {"dev_fit": 0.7}, seed=13)
        self.assertEqual([s.sample_id for s in again], [s.sample_id for s in fit])


class TestFeatureTable(unittest.TestCase):
    """B 段：特征表构建、标签回填、契约校验。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, cls.predictions = build_pipeline_data(n_base=3)
        cls.stats: dict = {}
        cls.rows = build_feature_table(cls.predictions, cls.samples, None, {"features": {}},
                                       stats=cls.stats)

    def test_rows_match_feature_columns(self) -> None:
        self.assertTrue(self.rows)
        for row in self.rows:
            self.assertEqual(set(row.to_dict()), set(FEATURE_COLUMNS))

    def test_undetermined_dropped_and_counted(self) -> None:
        unique_ids = len({p.sample_id for p in self.predictions})
        self.assertEqual(self.stats["n_undetermined"], 1)
        self.assertEqual(len(self.rows), unique_ids - 1)

    def test_labels_backfilled_by_rules(self) -> None:
        by_id = {row.sample_id: row for row in self.rows}
        for sample in self.samples:
            row = by_id.get(sample.sample_id)
            if row is None:
                continue
            prediction = next(p for p in self.predictions if p.sample_id == sample.sample_id)
            expected = derive_label(prediction.answer, sample.gold_answer,
                                    sample.construct_params.get("injected_values", []),
                                    sample.challenge_type,
                                    bool(sample.construct_params.get("is_control")))
            self.assertEqual(row.is_hallucination, expected, sample.sample_id)

    def test_baseline_confidence_kept_out_of_method_groups(self) -> None:
        flat = {name for group in METHOD_FEATURE_GROUPS.values() for name in group}
        self.assertNotIn("baseline_confidence", flat)
        self.assertTrue(all(row.baseline_confidence == row.baseline_confidence for row in self.rows))

    def test_features_file_passes_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = io.write_table(pathlib.Path(tmp) / "e2e.parquet",
                                  [row.to_dict() for row in self.rows],
                                  columns=list(FEATURE_COLUMNS))
            self.assertTrue(path.exists())
            self.assertEqual(validate.validate_file("features", path), [])


class TestDetectionAndEvaluation(unittest.TestCase):
    """C 段：判别器 → 分组指标 → 配对检验 → 汇总表 → 图表（需 sklearn / scipy / matplotlib）。"""

    rows: list[FeatureRow] = []
    columns: list[str] = []

    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_SKLEARN:
            return
        rng = random.Random(13)
        for index in range(30):
            label = index % 2
            for suffix, challenge_type in (("evidence_conflict", "evidence_conflict"), ("none", "none")):
                cls.rows.append(FeatureRow.from_dict({
                    "sample_id": f"hotpotqa-dev-{index:05d}-{suffix}", "exp_id": "e2e-logreg-13",
                    "challenge_type": challenge_type, "is_hallucination": label,
                    "entailment_max": 0.85 - 0.6 * label + rng.uniform(-0.05, 0.05),
                    "entailment_mean": 0.8 - 0.55 * label + rng.uniform(-0.05, 0.05),
                    "contradiction_max": 0.1 + 0.5 * label + rng.uniform(-0.05, 0.05),
                    "overlap_em": float(1 - label), "overlap_f1": 0.9 - 0.6 * label,
                    "citation_cov": 0.9 - 0.3 * label, "conflict_count": float(rng.randint(0, 2)),
                    "retrieval_top_score": rng.uniform(2, 9), "answer_len": float(rng.randint(1, 12)),
                    "baseline_confidence": 0.6 - 0.10 * label + rng.uniform(-0.4, 0.4),
                }))
        cls.columns = ["entailment_max", "entailment_mean", "contradiction_max", "overlap_em",
                       "overlap_f1", "citation_cov", "conflict_count", "retrieval_top_score"]

    @unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
    def test_train_cv_predict_and_calibrate(self) -> None:
        from src.detection.calibrate import best_threshold, calibrate
        from src.detection.model import make_classifier
        from src.detection.predict import predict
        from src.detection.train import train_and_evaluate
        from src.evaluation.metrics import score_metrics

        matrix = [[getattr(r, c) for c in self.columns] for r in self.rows]
        for kind in ("logreg", "gbdt", "mlp"):
            with self.subTest(kind=kind):
                trained = train_and_evaluate(self.rows, self.columns, kind, seed=13, cv_folds=5)
                self.assertGreater(trained["cv_metrics"]["auc"], 0.7)
                self.assertEqual(len(trained["oof_scores"]), len(self.rows))
                probs = make_classifier(kind, seed=13).fit(matrix, [r.is_hallucination for r in self.rows]) \
                    .predict_proba(matrix)[:, 1]
                self.assertTrue(all(0.0 <= float(p) <= 1.0 for p in probs))

        trained = train_and_evaluate(self.rows, self.columns, "logreg", seed=13)
        labels = [r.is_hallucination for r in self.rows]
        records = predict("e2e-logreg-13", trained["model"], self.rows, self.columns,
                          config_hash="e2e", model_name="qwen2.5-0.5b+logreg")
        self.assertTrue(all(r["config_hash"] == "e2e" and r["env_report"] for r in records))
        scores = [r["hallucination_prob"] for r in records]
        self.assertGreater(score_metrics(labels, scores)["auc"], 0.7)

        base_auc = score_metrics(labels, scores)["auc"]
        for method in ("none", "platt", "temperature"):
            with self.subTest(method=method):
                calibrated = calibrate(trained["oof_scores"], labels, method=method)(scores)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in calibrated))
                self.assertAlmostEqual(score_metrics(labels, calibrated)["auc"], base_auc, places=9)
        self.assertTrue(0.0 <= best_threshold(scores, labels) <= 1.0)

    @unittest.skipUnless(HAS_SKLEARN and HAS_SCIPY, "需要 scikit-learn 与 scipy")
    def test_grouped_significance_and_tables(self) -> None:
        from src.detection.train import train_and_evaluate
        from src.evaluation.grouped import group_sizes, grouped_metrics
        from src.evaluation.significance import paired_tests
        from src.evaluation.tables import aggregate, to_markdown_table

        trained = train_and_evaluate(self.rows, self.columns, "logreg", seed=13)
        labels = [r.is_hallucination for r in self.rows]
        model_scores = [float(s) for s in trained["oof_scores"]]
        baseline_scores = [1.0 - float(r.baseline_confidence) for r in self.rows]

        grouped = grouped_metrics(self.rows, model_scores)
        self.assertEqual(set(grouped), {"evidence_conflict", "none", "all"})
        self.assertEqual(grouped["all"]["n_samples"], len(self.rows))
        self.assertEqual(sum(group_sizes(self.rows).values()), len(self.rows))

        result = paired_tests(labels, model_scores, baseline_scores, alpha=0.05, family_size=3)
        self.assertEqual(result["n_pairs"], len(self.rows))
        self.assertGreater(result["delta_auc"], 0.0)
        for key in ("ttest", "wilcoxon"):
            self.assertTrue(0.0 <= result[key]["p_value"] <= 1.0)
        self.assertIn("holm_ttest", result)

        rows = aggregate([{"method": "M+logreg", "split": "challenge", "auc": 0.8, "pr_auc": 0.7},
                          {"method": "M+logreg", "split": "challenge", "auc": 0.9, "pr_auc": 0.8}],
                         ["method", "split"], ["auc"])
        self.assertEqual(rows[0]["n_runs"], 2)
        self.assertAlmostEqual(rows[0]["auc"], 0.85)
        self.assertIn("±", to_markdown_table(rows))

    @unittest.skipUnless(HAS_MATPLOTLIB, "需要 matplotlib")
    def test_key_figures_written(self) -> None:
        from src.evaluation.grouped import grouped_metrics
        from src.evaluation.metrics import score_metrics
        from src.evaluation.plots import (plot_confusion, plot_grouped_bars, plot_reliability,
                                          plot_roc_pr)

        labels = [r.is_hallucination for r in self.rows]
        scores = [0.9 if label == 1 else 0.1 for label in labels]
        baseline = [1.0 - float(r.baseline_confidence) for r in self.rows]
        summary = grouped_metrics(self.rows, scores)
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            saved = [plot_grouped_bars(summary, str(out / "grouped.png")),
                     plot_roc_pr(labels, {"M": scores, "B1": baseline}, str(out / "roc.png")),
                     plot_confusion(score_metrics(labels, scores)["confusion"], str(out / "cm.png")),
                     plot_reliability(scores, labels, str(out / "rel.png"))]
            for path in saved:
                self.assertTrue(pathlib.Path(path).exists())
                self.assertGreater(pathlib.Path(path).stat().st_size, 1000)


class TestCliDryRun(unittest.TestCase):
    """全部 CLI 入口的 --dry-run 可跑通（保证没有导入/参数错误）。"""

    MODULES = ["src.datasets.cli", "src.retrieval.cli", "src.generation.cli", "src.features.cli",
               "src.detection.cli", "src.evaluation.cli"]

    @classmethod
    def setUpClass(cls) -> None:
        cls.before = set(LOG_DIR.glob("*.log")) if LOG_DIR.exists() else set()

    @classmethod
    def tearDownClass(cls) -> None:
        for path in sorted(set(LOG_DIR.glob("*.log")) - cls.before):  # 清理本测试产生的运行日志
            path.unlink()

    def test_dry_run_exits_zero(self) -> None:
        for module in self.MODULES:
            with self.subTest(module=module):
                # 用测试专用 exp_id，避免把日志追加到已跟踪的 results/logs/*.log
                proc = subprocess.run([sys.executable, "-m", module, "--dry-run",
                                       "--exp-id", "unittest-dryrun-probe"], cwd=ROOT,
                                      capture_output=True, text=True, timeout=300, check=False)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_eval_requires_exp_ids(self) -> None:
        proc = subprocess.run([sys.executable, "-m", "src.evaluation.cli"], cwd=ROOT,
                              capture_output=True, text=True, timeout=300, check=False)
        self.assertEqual(proc.returncode, 1, "缺少 --exp-ids 时应返回输入错误码 1")


if __name__ == "__main__":
    unittest.main()
