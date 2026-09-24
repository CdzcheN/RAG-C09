#!/usr/bin/env python3
"""生成开题报告插图：系统架构图、技术路线图（matplotlib，中文用 Noto Sans CJK）。

用法: python scripts/make_figures.py
输出: docs/figures/开题_架构图.png、docs/figures/开题_技术路线图.png
依赖: matplotlib（见 requirements.txt），中文字体 Noto Sans CJK（Ubuntu: fonts-noto-cjk）
说明: 图由数据/结构描述生成，可随时重绘；报告中的图不得使用截图（要求.md 2.3 条第 6 款）。
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"

LAYERS = [
    ("L0 数据层", "SQuAD v2.0 · HotpotQA（datasets 在线加载，revision 固定）", "#eef4fb"),
    ("L1 构造层", "三类退化算子：检索失败 / 证据冲突 / 答案过时（≥300 例）", "#eaf6ee"),
    ("L2 检索层", "BM25 稀疏检索（主线）· 稠密检索（可选支线）", "#fdf6e3"),
    ("L3 生成层", "Qwen2.5-0.5B-Instruct（fp16、贪心解码）+ 自评置信度基线", "#fdeeee"),
    ("L4 特征层", "蕴含概率（NLI）· 检索-生成重叠度 · 证据冲突计数", "#f2eef9"),
    ("L5 判别层", "LogReg / GBDT / 轻量 MLP + 概率校准", "#eef7f9"),
    ("L6 评估层", "AUC / PR-AUC · 分组评估 · 3 种子 + 配对检验 · 图表重绘", "#f4f4f4"),
]


def fig_architecture() -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(LAYERS) * 1.28 + 0.4)
    ax.axis("off")

    h, gap = 0.92, 0.36
    for i, (name, desc, color) in enumerate(reversed(LAYERS)):
        y = 0.2 + i * (h + gap)
        ax.add_patch(FancyBboxPatch((0.25, y), 9.5, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.1, edgecolor="#4a6b8a", facecolor=color))
        ax.text(0.55, y + h / 2, name, fontsize=11, fontweight="bold", va="center", ha="left", color="#22384d")
        ax.text(2.55, y + h / 2, desc, fontsize=9.2, va="center", ha="left", color="#33475b")
        if i < len(LAYERS) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y + h + 0.02), (5.0, y + h + gap - 0.02),
                                         arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#4a6b8a"))
    ax.text(5.0, len(LAYERS) * 1.28 + 0.12, "课题 C09 系统架构与数据流", fontsize=12.5,
            fontweight="bold", ha="center", color="#1c2b3a")
    fig.tight_layout()
    p = OUT / "开题_架构图.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_roadmap() -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    stages = [
        ("阶段一\n准备", "环境与依赖\n模型/数据预热\n种子协议", "#eef4fb"),
        ("阶段二\n管道打通", "检索 + 生成\n字段契约冻结", "#eef4fb"),
        ("阶段三\n基线与挑战集", "常规集基线\n三类算子 20 例原型\n→ 放量 ≥300 例", "#eaf6ee"),
        ("阶段四\n特征与判别器", "三组一致性特征\n判别器 + 校准\n误差分析迭代", "#f2eef9"),
        ("阶段五\n固化与验证", "3 种子 + 检验\n分组评估 + 消融\n人工抽检 100 例", "#f4f4f4"),
    ]
    gates = ["G0 冒烟通过", "G1 输出逐字节一致", "G2/G3 标签可自动判定", "G4 AUC 增益 ≥0.10", "G5/G6 消融・可追溯"]

    w, h = 2.05, 1.5
    for i, ((title, desc, color), gate) in enumerate(zip(stages, gates)):
        x = 0.25 + i * 2.5
        ax.add_patch(FancyBboxPatch((x, 2.55), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                                    linewidth=1.1, edgecolor="#4a6b8a", facecolor=color))
        ax.text(x + w / 2, 2.55 + h - 0.28, title, fontsize=9.6, fontweight="bold",
                ha="center", va="center", color="#22384d")
        ax.text(x + w / 2, 2.55 + h / 2 - 0.28, desc, fontsize=8.1, ha="center", va="center", color="#33475b")
        ax.add_patch(FancyBboxPatch((x + 0.12, 1.55), w - 0.24, 0.62, boxstyle="round,pad=0.03,rounding_size=0.3",
                                    linewidth=1.0, edgecolor="#b07a2a", facecolor="#fdf3e0"))
        ax.text(x + w / 2, 1.86, gate, fontsize=8.0, ha="center", va="center", color="#6b4a12")
        ax.add_patch(FancyArrowPatch((x + w / 2, 2.53), (x + w / 2, 2.19), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.1, color="#4a6b8a"))
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 3.3), (x + 2.5 - 0.02, 3.3), arrowstyle="-|>",
                                         mutation_scale=13, linewidth=1.3, color="#4a6b8a"))

    ax.add_patch(FancyArrowPatch((9.55, 1.45), (8.35, 1.45), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color="#a33", connectionstyle="arc3,rad=0.35"))
    ax.text(8.95, 1.02, "未达标：误差分析 → 补特征 / 换判别器 / 校准 → 重跑",
            fontsize=8.0, ha="center", color="#a33")
    ax.text(6.6, 4.72, "课题 C09 技术路线与阶段门禁", fontsize=12.5, fontweight="bold",
            ha="center", color="#1c2b3a")
    fig.tight_layout()
    p = OUT / "开题_技术路线图.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (fig_architecture(), fig_roadmap()):
        print(f"[ok] {p.relative_to(ROOT)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
