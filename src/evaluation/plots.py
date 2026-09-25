#!/usr/bin/env python3
"""图表自绘（禁止截图，要求.md 2.3 条第 6 款）。

负责人: C
对应文档: docs/实验与评估规范.md §6
状态: 已实现（T6.2）

六张必备图（每张都由 `results/metrics/` 的数据重绘，图与数据版本一致）:
1. `plot_grouped_bars`    按幻觉类型的 AUC/查准/查全分组柱状图（含样本量标注）
2. `plot_roc_pr`          ROC 与 PR 曲线（M vs B1 vs B2 同图）
3. `plot_confusion`       混淆矩阵热图（最优阈值）
4. `plot_feature_distribution` 特征分布对比（幻觉 vs 非幻觉箱线图）
5. `plot_ablation`        消融结果对比图
6. `plot_reliability`     校准曲线（可靠性图）

字体：优先 Noto Sans CJK（`scripts/make_figures.py` 已验证可用），按可用性回退，保证中文不出方框。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common.logging_utils import get_logger

LOG = get_logger(__name__)

CJK_FONTS = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC", "WenQuanYi Zen Hei",
             "Source Han Sans SC", "Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans")


def setup_matplotlib() -> Any:
    """配置无界面后端与中文字体，返回 `matplotlib.pyplot`。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("绘图需要 matplotlib（见 requirements.txt）") from exc

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams.get("font.sans-serif", [])]
            break
    else:
        LOG.warning("未找到中文字体（%s 均不可用），图内中文可能显示为方框", CJK_FONTS)
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _finish(plt: Any, fig: Any, out: str, note: str = "") -> str:
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    LOG.info("图已保存：%s", out)
    return str(out)


def plot_grouped_bars(summary: Mapping[str, Mapping[str, Any]], out: str,
                      metrics: Sequence[str] = ("auc", "precision", "recall"),
                      title: str = "按幻觉类型分组指标") -> str:
    """按幻觉类型绘制 AUC / 查准 / 查全分组柱状图（含误差棒与样本量 n）。

    `summary` 形如 `{类型: {指标: 值, "指标_std": 标准差, "n_samples": n}}`（`grouped_metrics` 的输出
    可直接使用；`all` 组会一并绘出）。
    """
    plt = setup_matplotlib()
    groups = list(summary.keys())
    if not groups:
        raise ValueError("summary 为空，无可绘制内容")

    n_metrics = len(metrics)
    width = 0.8 / max(1, n_metrics)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(groups)), 4.2))
    for index, metric in enumerate(metrics):
        values, errors = [], []
        for group in groups:
            value = float(summary[group].get(metric, float("nan")))
            values.append(0.0 if value != value else value)
            std = summary[group].get(f"{metric}_std")
            errors.append(0.0 if std is None else float(std))
        offset = (index - (n_metrics - 1) / 2) * width
        ax.bar([i + offset for i in range(len(groups))], values, width=width, yerr=errors,
               capsize=3, label=metric)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={summary[g].get('n_samples', '?')})" for g in groups], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("指标值")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _finish(plt, fig, out, "误差棒 = 标准差（多 seed）；n = 组内样本量")


def plot_roc_pr(y_true: Sequence[int], curves: Mapping[str, Sequence[float]], out: str,
                title: str = "ROC 与 PR 曲线") -> str:
    """绘制 M / B1 / B2 的 ROC 与 PR 曲线（`curves` = {方法名: 分数序列}）。"""
    plt = setup_matplotlib()
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("绘制 ROC/PR 需要 scikit-learn（见 requirements.txt）") from exc

    labels = [int(v) for v in y_true]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, scores in curves.items():
        fpr, tpr, _ = roc_curve(labels, [float(s) for s in scores])
        axes[0].plot(fpr, tpr, label=name, linewidth=1.5)
        precision, recall, _ = precision_recall_curve(labels, [float(s) for s in scores])
        axes[1].plot(recall, precision, label=name, linewidth=1.5)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_xlabel("假正例率 FPR")
    axes[0].set_ylabel("真正例率 TPR")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("查全率 Recall")
    axes[1].set_ylabel("查准率 Precision")
    axes[1].set_title("PR")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    return _finish(plt, fig, out, f"正类 = 幻觉；n = {len(labels)}（+{sum(labels)} / -{len(labels) - sum(labels)}）")


def plot_confusion(confusion: Mapping[str, int], out: str, title: str = "混淆矩阵（最优阈值）") -> str:
    """混淆矩阵热图（四格标注 tp/fp/tn/fn 与数值）。"""
    plt = setup_matplotlib()
    matrix = [[int(confusion.get("tn", 0)), int(confusion.get("fp", 0))],
              [int(confusion.get("fn", 0)), int(confusion.get("tp", 0))]]
    fig, ax = plt.subplots(figsize=(4.4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i][j]}", ha="center", va="center",
                    color="black" if matrix[i][j] < max(max(row) for row in matrix) else "white")
    ax.set_xticks([0, 1], labels=["预测非幻觉", "预测幻觉"], fontsize=8)
    ax.set_yticks([0, 1], labels=["真实非幻觉", "真实幻觉"], fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _finish(plt, fig, out, "正类 = 幻觉")


def plot_feature_distribution(rows: Sequence[Mapping[str, Any]], out: str,
                              feature: str = "entailment_max",
                              title: str = "特征分布：幻觉 vs 非幻觉") -> str:
    """特征分布箱线图（`rows` 为 FeatureRow 字典序列，按 is_hallucination 分组）。"""
    plt = setup_matplotlib()
    groups: dict[int, list[float]] = {0: [], 1: []}
    for row in rows:
        value = row.get(feature)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric == numeric:
            groups[int(row.get("is_hallucination", 0))].append(numeric)
    if not groups[0] and not groups[1]:
        raise ValueError(f"特征 {feature} 无有效数值，无法绘图")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    labels = [f"非幻觉 (n={len(groups[0])})", f"幻觉 (n={len(groups[1])})"]
    try:  # matplotlib >= 3.9 用 tick_labels，旧版用 labels
        ax.boxplot([groups[0], groups[1]], tick_labels=labels)
    except TypeError:  # pragma: no cover
        ax.boxplot([groups[0], groups[1]], labels=labels)
    ax.set_ylabel(feature)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return _finish(plt, fig, out, "箱线图：中位数/四分位/离群点")


def plot_ablation(summary: Mapping[str, Mapping[str, Any]], out: str, metric: str = "auc",
                  title: str = "消融结果对比") -> str:
    """消融结果对比柱状图（`summary` = {变体名: {metric: 值, "metric_std": 标准差}}）。"""
    plt = setup_matplotlib()
    variants = list(summary.keys())
    if not variants:
        raise ValueError("summary 为空，无可绘制内容")
    values = [float(summary[v].get(metric, float("nan"))) for v in variants]
    errors = [float(summary[v].get(f"{metric}_std", 0.0) or 0.0) for v in variants]
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(variants)), 4.2))
    ax.bar(range(len(variants)), [0.0 if v != v else v for v in values], yerr=errors, capsize=3)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return _finish(plt, fig, out, "误差棒 = 标准差（多 seed）")


def plot_reliability(scores: Sequence[float], y_true: Sequence[int], out: str, n_bins: int = 10,
                     title: str = "校准曲线（可靠性图）") -> str:
    """校准曲线：每个概率分箱里的平均预测概率 vs 实际正类率（含对角参考线）。"""
    plt = setup_matplotlib()
    pairs = sorted(((float(s), int(y)) for s, y in zip(scores, y_true)), key=lambda item: item[0])
    if not pairs:
        raise ValueError("scores 为空，无法绘制校准曲线")
    bin_size = max(1, len(pairs) // max(1, n_bins))
    xs, ys, sizes = [], [], []
    for start in range(0, len(pairs), bin_size):
        chunk = pairs[start:start + bin_size]
        if not chunk:
            continue
        xs.append(sum(s for s, _ in chunk) / len(chunk))
        ys.append(sum(y for _, y in chunk) / len(chunk))
        sizes.append(len(chunk))

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="理想校准")
    ax.plot(xs, ys, marker="o", linewidth=1.5, label="实际")
    ax.set_xlabel("预测幻觉概率")
    ax.set_ylabel("实际幻觉比例")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(plt, fig, out, f"分箱数 ≈ {len(sizes)}，每箱样本量 {min(sizes)}–{max(sizes)}")
