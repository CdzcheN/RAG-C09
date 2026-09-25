# 课题 C09：检索增强生成的幻觉检测与量化评估方法研究

开放域问答 RAG 管道 + “检索失败 / 证据冲突 / 答案过时”三类挑战子集 + 基于证据一致性的幻觉自动判别器。
课题完整要求见 [`C09_检索增强生成的幻觉检测与量化评估方法研究.md`](C09_检索增强生成的幻觉检测与量化评估方法研究.md)，
考核与交付要求见 [`要求.md`](要求.md)。

## 目录结构

```
RAG-C09/
├── C09_...md                课题任务书
├── 要求.md                  课程统一要求
├── requirements.txt        依赖清单（Ubuntu / Windows 通用；全队统一环境的唯一来源）
├── docs/                    项目文档（入口：开发文档索引.md）
│   ├── 开发文档索引.md       各文档职责、上手顺序、与要求.md 交付物的映射
│   ├── 项目启动与实施指南.md  环境与依赖、系统框架图、开发流程图（门禁）、三人分工、验收对照表
│   ├── 接口契约.md          中间产物 schema 与字段字典、目录命名、变更流程、产物校验
│   ├── 数据构造规范.md       三类退化算子、标签规则化推导、种子管理与质量门禁
│   ├── 实验与评估规范.md      指标口径、对照与消融设置、统计检验、图表规范、负结果纪律
│   ├── 编码与协作规范.md      目录命名、CLI 运行入口、配置与日志、跨平台约定、代码风格
│   ├── 实验记录.md          周会/结果/消融/失败/决策记录模板（周会围绕它开）
│   ├── 开题报告.md/.docx/.pdf 九要素开题报告（docx 为交付件，md 为源文件，pdf 为预览）
│   └── figures/             开题报告插图（由 scripts/make_figures.py 生成，可重绘）
├── data/
│   ├── README.md            数据集说明：来源、校验结果、许可
│   ├── raw/                 已下载的离线副本（备选，非主路径）
│   │   ├── squad/           SQuAD v2.0: train/dev JSON
│   │   ├── hotpotqa/        HotpotQA distractor parquet
│   │   ├── MANIFEST.sha256  SHA-256 校验清单
│   │   └── VERIFY.json      内容统计与校验报告
│   ├── interim/             构造中间产物（不入库）
│   └── processed/           冻结产物：challenge_set.jsonl / normal_set.jsonl / construct_log.json
├── refs/
│   ├── arxiv_ids.txt        参考文献候选清单（人工选定 + 分组）
│   ├── references.json      结构化元数据（含匹配证据与 ACL BibTeX 原文）
│   ├── references.bib       可直接引用的 BibTeX
│   ├── 参考文献清单.md       人类可读清单（含 DOI/arXiv 链接与本地 PDF 对应）
│   ├── 文献综述.md          课题文献综述（引用 29 篇，按 GB/T 7714—2025 著录）
│   └── pdf/                 29 篇开放获取论文 PDF + MANIFEST.sha256
├── configs/                 超参配置（代码内不写魔法数）
│   ├── default.yaml         种子、路径、数据集标识、检索与生成参数
│   ├── models.yaml          模型选型与显存约束
│   ├── challenge.yaml       三类退化算子参数与输出路径
│   └── detect.yaml          特征分组、对照、判别器、检验与目标增益
├── src/                     源码（分层见 docs/项目启动与实施指南.md §2.1）
│   ├── common/              schema · seeding · io · logging_utils · cli_utils · validate · text
│   ├── datasets/            在线加载 reader · 三类算子 challenge_builder · 抽样划分 split
│   ├── retrieval/           passage store · BM25（主管线）· dense（可选）· Recall@k
│   ├── generation/          模型装载 · 解码 · 端到端管道 · 自评置信度基线
│   ├── features/            entailment · overlap · consistency · build_features
│   ├── detection/           判别器 · 训练 · 概率校准 · 预测
│   ├── evaluation/          metrics · grouped · significance · tables · plots
│   └── demo/                演示入口（复用管道实现）
├── tests/                   unittest 单元测试（overlap / seeding / metrics / contract / end_to_end，共 56 用例）
├── results/                 实验产物（predictions · features · metrics · figures · logs）
└── scripts/
    ├── check_env.py              环境自检：conda 环境 + requirements.txt（依赖/GPU/镜像）
    ├── prefetch_assets.py        预热模型与数据集（Ubuntu/Windows 通用）
    ├── smoke_test.py             冒烟测试：结构 / 配置 / 契约 / 产物 / 管道前置
    ├── final_test.py             最终验收测试：L0 静态 → L1 单元 → L2 数据 → L3 管道 → L4 判别评估
    ├── make_figures.py           生成开题报告插图（matplotlib，可重绘）
    ├── md2docx.py                Markdown → docx 转换（需可选依赖 python-docx）
    ├── download_data.sh          数据集下载（离线备选；bash，Windows 需 WSL/Git Bash）
    ├── verify_data.py            数据集本地副本校验（行数/字段/分布）
    ├── download_refs.sh          下载参考文献 PDF
    ├── fetch_reference_metadata.py  抓取文献元数据（arXiv + OpenAlex + ACL）
    └── build_reference_docs.py   生成 BibTeX 与文献清单
```

## 环境准备（全队统一：conda 环境 `rag-c09`，Ubuntu / Windows 通用）

```bash
conda create -n rag-c09 python=3.10 -y
conda activate rag-c09
python -m pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com        # Windows: $env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/check_env.py                     # 环境自检，退出码须为 0
python scripts/prefetch_assets.py               # 预热模型与数据集（首次需联网）
```

- 依赖只有一处定义：[`requirements.txt`](requirements.txt)（3 个锚点包用 `==` 固定；跨平台差异见文件内注释）；
- 环境配置、跨平台激活与环境变量对照表、显存约束、数据在线加载：见 [`docs/项目启动与实施指南.md`](docs/项目启动与实施指南.md) §1；
- 数据（SQuAD v2 / HotpotQA）**由代码在线加载**，无需手动下载，见文档 §1.4；
- 工作区遗留的 `.venv/` 与其他旧环境**不再使用**。

## 运行入口

统一约定：每个模块一个 CLI，共享 `--config / --seed / --out / --limit / --dry-run / --exp-id`；
退出码 `0` 成功 / `1` 输入错误 / `2` 运行失败 / `3` 产物校验不通过（见 [`docs/编码与协作规范.md`](docs/编码与协作规范.md) §2）。

```bash
# 0) 自检与冒烟
python scripts/check_env.py --report results/env_report.json    # 环境自检（退出码须为 0）
python scripts/smoke_test.py --with-tests                       # 结构/配置/契约冒烟 + 单元测试
python scripts/final_test.py                                    # 最终验收（L0–L4，默认全跑；--fast 只跑 L0–L2）

# 1) 数据构造（A）
python -m src.datasets.challenge_builder --config configs/challenge.yaml --seed 1000
python -m src.datasets.split             --config configs/challenge.yaml

# 2) 管道与基线（B）
python -m src.generation.pipeline            --config configs/default.yaml --seed 13 --limit 20   # 20 条冒烟
python -m src.generation.baseline_confidence --config configs/default.yaml --seed 13

# 3) 特征提取（B）
python -m src.features.build_features --config configs/default.yaml --seed 13

# 4) 判别与评估（C）
python -m src.detection.train   --config configs/detect.yaml --seed 13 --classifier logreg
python -m src.evaluation.eval   --config configs/detect.yaml --exp-ids w4-consistency-logreg-13

# 5) 产物契约校验
python -m src.common.validate predictions results/predictions/w3-pipeline-13.jsonl
python -m src.common.validate features    results/features/w3-pipeline-13.parquet
```

**当前实现状态**：`src/` 下 28 个业务模块（A 数据与检索、B 生成与特征、C 判别与评估）已全部实现，
每个模块一个 CLI，均支持 `--dry-run`；契约与协作规范见 [`docs/接口契约.md`](docs/接口契约.md) 与
[`docs/编码与协作规范.md`](docs/编码与协作规范.md)。交付前的端到端验收跑
`python scripts/final_test.py`（缺重依赖的层级会记 SKIP 并说明安装方式，不会静默通过）。

## 随机种子

| 项 | 值 | 说明 |
|---|---|---|
| 模型 / 训练种子 | `13, 42, 2024` | 核心结论必须报告三者的 mean ± std 与配对检验 |
| 数据构造种子 | `1000 + 基样本序号` | **与模型种子解耦**，保证更换模型种子时挑战集逐字节不变（消融才可比） |
| 抽样 / 抽检种子 | 显式传入并记录 | 常规集抽样、人工抽检 100 例 |
| 统一入口 | `src/common/seeding.py::set_seed(seed)` | 实验代码禁止散落 `np.random.seed` / `torch.manual_seed` |

产物元信息（首行 `_meta`）含 `seed`、`config_hash`、数据集 `revision` 与环境指纹路径，见 [`docs/接口契约.md`](docs/接口契约.md) §4。

## 已获取的数据与文献

| 项目 | 内容 | 校验 |
|---|---|---|
| SQuAD v2.0 | train 130,319 问 / dev 11,873 问 | 与官方发布统计一致（见 `data/VERIFY.json`） |
| HotpotQA distractor | train 90,447 / validation 7,405 | 行数与列结构完整（同上） |
| 参考文献 | 29 篇开放获取论文 PDF + BibTeX | 标题/作者取自 arXiv API，出版信息取自 OpenAlex 与 ACL Anthology 官方条目 |

> `data/raw/` 下的本地副本与校验记录仅作**离线备选**；数据主路径为代码在线加载（`load_dataset`），见 [`docs/项目启动与实施指南.md`](docs/项目启动与实施指南.md) §1.4。

元数据采集做了双重校验（标题完全一致 + 作者姓氏有交集），并由 `build_reference_docs.py` 再做一道
“ACL 官方条目标题 == arXiv 标题”的交叉校验（当前全部通过）；未匹配到正式出版记录的条目一律
按预印本著录，不做人工补写，保证全部引用真实、可检索。

其中 7 篇（RAG 原始论文、RAG 综述、干扰上下文、模型自我认知、归因问答、幻觉雪球、Self-RAG）
未匹配到正式出版记录：它们多为 NeurIPS / ICML / ICLR 或纯 arXiv 论文，这些出版方不注册 DOI，
在 OpenAlex / Crossref 中没有可用的出版条目。另有客观原因：本机 IP 在批量查询中被 OpenAlex 限流
（HTTP 429），脚本已自动切换到 Crossref 兜底。OpenAlex 限流恢复后重新执行下面两条命令即可自动补全
（脚本带增量缓存，仅重试未匹配的条目）：

```bash
conda activate rag-c09 && python scripts/fetch_reference_metadata.py && python scripts/build_reference_docs.py
```

## 复现命令

```bash
conda activate rag-c09                                 # 环境配置见 docs/项目启动与实施指南.md §1
python scripts/check_env.py                            # 0) 环境自检（退出码须为 0）

python scripts/prefetch_assets.py                      # 1) 预热模型与数据集（首次联网，之后离线）
                                                       #    数据集由 load_dataset 在线加载，无需手动下载（文档 §1.4）

bash scripts/download_refs.sh                          # 2) 下载文献 PDF
python scripts/fetch_reference_metadata.py             #    抓取文献元数据
python scripts/build_reference_docs.py                 #    生成 BibTeX 与清单
```

所有下载脚本幂等，可重复执行；网络中断后重跑即可续传。

## 环境实测说明

- `huggingface.co` 在本机不可达（连接被重置），数据与模型请走 `hf-mirror.com` 镜像；
- HotpotQA 官方站点 `curtis.ml.cmu.edu` 连接超时，故改用 HuggingFace 数据集镜像；
- `arxiv.org`、`aclanthology.org`、`api.openalex.org` 可正常访问；
- 可用算力：RTX 3050 Laptop（4 GB 显存）、15 GB 内存、约 70 GB 可用磁盘，
  满足“单张 ≤12 GB 消费级 GPU、48 小时内完成全部实验”的约束（需按 4 GB 显存选型生成模型）。

## 后续待办

- [ ] RAG 管道：检索器（BM25 / 稠密检索）+ 开源小生成模型（如 Qwen2.5-0.5B/1.5B-Instruct），全部离线可复现
- [ ] 常规测试集基线与生成模型自评置信度基线
- [ ] 挑战子集构造（≥300 例，覆盖检索失败 / 证据冲突 / 答案过时，随机种子管理）
- [ ] 一致性特征提取（蕴含概率、检索—生成重叠度）与判别器训练
- [ ] 按幻觉类型分组评估、消融实验、≥3 随机种子与显著性检验
- [ ] 版本管理：**不在本工作区进行**（团队实际协作仓库位于别处），本目录不使用 Git
