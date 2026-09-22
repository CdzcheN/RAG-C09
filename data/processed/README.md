# data/processed/ —— 冻结的数据产物

本目录存放**挑战集与常规集**，是数据再构造的交付物（`要求.md` 2.2 条"数据再构造"、1.4 条"数据构造脚本"）。

| 文件 | 内容 | 规模 | 产者 |
|---|---|---|---|
| `challenge_set.jsonl` | 挑战集：检索失败 / 证据冲突 / 答案过时，各 100 例（50 基样本 × 污染版 + 对照版） | ≥300 例 | A |
| `normal_set.jsonl` | 常规集：SQuAD v2 dev 与 HotpotQA validation 各抽样 300 条 | 600 条 | A |
| `construct_log.json` | 构造日志：算子参数、种子、各类数量、产物 SHA-256、数据集 revision | — | A |

要求：

1. **必须入库**（小体积、可复现），并在 `construct_log.json` 中记录全部构造参数与种子；
2. 字段定义见 `docs/接口契约.md` §2.1，构造规则见 `docs/数据构造规范.md`；
3. 冻结后只增不改：任何调整都新起版本（`v2`）并说明原因；
4. 人工抽检结果写到 `results/metrics/construct_audit.csv`，一致率需 ≥90% 才允许冻结（G3 门禁）。
