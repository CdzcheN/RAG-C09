# 数据集说明（课题 C09）

> **数据获取主路径为代码在线加载**：SQuAD v2.0 与 HotpotQA 通过 HuggingFace `datasets` 按标识直接加载
> （`load_dataset("rajpurkar/squad_v2")`、`load_dataset("hotpotqa/hotpot_qa", "distractor")`），不需要手动下载，
> 见 `docs/项目启动与实施指南.md` §1.4。本目录下的文件是此前用脚本下载的**离线副本**，作为无网络时的备选与完整性校验记录保留。

本目录保存课题 C09 所需的公开数据集。所有文件均由脚本下载，未做任何人工修改。

## 1. 文件清单

| 文件 | 大小 | 内容 |
|---|---|---|
| `squad/train-v2.0.json` | 42,123,633 B | SQuAD v2.0 训练集 |
| `squad/dev-v2.0.json` | 4,370,528 B | SQuAD v2.0 开发集 |
| `hotpotqa/train-00000-of-00002.parquet` | 165,624,177 B | HotpotQA distractor 训练集（分片 1/2） |
| `hotpotqa/train-00001-of-00002.parquet` | 166,162,479 B | HotpotQA distractor 训练集（分片 2/2） |
| `hotpotqa/validation-00000-of-00001.parquet` | 27,452,575 B | HotpotQA distractor 验证集 |

完整性校验值见 [`raw/MANIFEST.sha256`](raw/MANIFEST.sha256)，内容统计见 [`raw/VERIFY.json`](raw/VERIFY.json)。

## 2. 获取渠道

- **SQuAD v2.0**：官方发布页 <https://rajpurkar.github.io/SQuAD-explorer/>，直接下载 `dataset/train-v2.0.json` 与 `dataset/dev-v2.0.json`。
- **HotpotQA (distractor)**：官方发布页 <https://hotpotqa.github.io/>。
  本机实测官方下载地址 `curtis.ml.cmu.edu` 不可达（连接超时），因此改从
  HuggingFace 数据集 [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
  的 `distractor` 配置获取（经镜像 `hf-mirror.com`）。该配置即原始 distractor 设定的等价转换，
  字段与官方 JSON 版一一对应（见下节列结构）。

## 3. 校验结果

由 `scripts/verify_data.py` 实际读取文件统计（完整结果见 `VERIFY.json`）：

| 数据集 | 统计量 | 数值 |
|---|---|---|
| SQuAD v2.0 train | 问题数 | 130,319（其中不可回答 43,498） |
| SQuAD v2.0 train | 篇章数 / 文章数 | 19,035 / 442 |
| SQuAD v2.0 dev | 问题数 | 11,873（其中不可回答 5,945） |
| SQuAD v2.0 dev | 篇章数 / 文章数 | 1,204 / 35 |
| HotpotQA distractor train | 行数 | 45,224 + 45,223 = **90,447** |
| HotpotQA distractor validation | 行数 | 7,405（问句难度全部为 `hard`，符合 distractor 设定） |

上述数字与两个数据集的官方发布统计一致，可作为下载完整性的证据。
HotpotQA 的列结构为 `id, question, answer, type, level, supporting_facts, context`，
其中 `context` 为逐篇的 `title` + `sentences` 列表，`supporting_facts` 标注支撑句，可直接用于
构造“检索失败 / 证据冲突”类挑战样本。

## 4. 重新下载与校验

```bash
bash scripts/download_data.sh          # 幂等；bash 脚本，Windows 需 WSL/Git Bash
conda activate rag-c09 && python scripts/verify_data.py
```

`download_data.sh` 使用 `curl -C -` 断点续传并重试，网络中断后重复执行即可。

## 5. 使用许可

两个数据集均供学术研究免费使用，具体条款以各自官方发布页声明为准；
引用时请分别引用 SQuAD 与 HotpotQA 的原始论文（见 `refs/参考文献清单.md`）。
本目录文件不再对外分发。
