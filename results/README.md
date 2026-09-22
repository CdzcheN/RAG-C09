# 实验产物目录（results/）

本目录保存**全部实验产物**，是技术报告中每个数字的追溯终点（可追溯性门禁 G6，见 `docs/实验与评估规范.md` §9）。
本目录**不入库大文件**，图片与指标文件入库或随代码提交。

## 子目录与命名

| 子目录 | 内容 | 产者 | 命名 |
|---|---|---|---|
| `predictions/` | 模型预测（每行一条样本，首行为 `_meta`） | B | `<exp_id>.jsonl` |
| `features/` | 特征表（一行一样本） | B | `<exp_id>.parquet` |
| `metrics/` | 指标记录与汇总 | C | `<exp_id>.json`、`summary.csv` |
| `figures/` | 图表（由 `metrics/` 重绘，禁止截图） | C | 语义化名称，如 `grouped_auc.png` |
| `logs/` | 运行日志（含命令、config_hash、种子、异常） | 全组 | `<exp_id>.log` |

`exp_id` 命名：`{stage}-{method}-{seed}`，全小写连字符分隔，例如 `w4-consistency-logreg-13`（见 `docs/接口契约.md` §3）。

## 环境指纹

`env_report.json` 由各人自行生成，随实验记录留档：

```bash
python scripts/check_env.py --report results/env_report.json
```

## 纪律

1. **不覆盖历史结果**：重跑生成新的 `exp_id`（可加 `-r2` 后缀）；
2. **报告中的每个数字**都要能追到本目录中的一条记录；
3. 单个产物超过 50 MB 时改为压缩存储（`.jsonl.gz`）或只保留必要列；
4. 失败运行也要留日志（便于复盘），但不必保留其半成品产物。
