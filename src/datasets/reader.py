#!/usr/bin/env python3
"""数据读取：按数据集标识在线加载并做字段规范化。

负责人: A
对应文档: docs/项目启动与实施指南.md §1.4、docs/数据构造规范.md §1
状态: 已实现（T1.1）

主路径：`datasets.load_dataset` 在线加载并固定 `revision`（契约 §4）；
离线降级：`datasets` 未安装或联网失败时，改读工作区 `data/raw/` 下的本地副本，并在日志中
如实记录 `origin="local"`（对应 §1.4 的约定：本地副本仅作离线备选，不是取数主路径）。

网络不稳时的三个行为（避免把时间浪费在 huggingface_hub 的指数退避重试上）：
1. **进程内失败记忆**：一旦确认在线不可用，同一进程内后续加载直接走本地副本/缓存，不再重复等重试
   （SQuAD + HotpotQA 两次加载只忍一次网络超时）；
2. **缓存反解 revision**：联网取不到 `sha` 时，从 `HF_HOME/datasets/{owner}___{name}/{config}/{version}/{sha}`
   缓存目录反解出上游提交哈希，使契约 §4 的 `dataset_revision` 不为空（并标 `revision_source="cache"`）；
3. **强制离线**：设 `HF_HUB_OFFLINE=1` 或 `C09_OFFLINE=1` 时直接读缓存，完全不发网络请求。

规范化后的统一结构（供 challenge_builder / split / retrieval 消费）：

    {
      "id": str,                       # 原始样本 id
      "source": "squad" | "hotpotqa",
      "hf_split": str,                 # 上游划分名（squad: train/validation，hotpotqa: validation）
      "split_tag": str,                # 对外命名用的划分标签，统一为 "dev"（见契约 §2.1 示例）
      "row_index": int,                # 在所属划分中的行索引（provenance 用）
      "question": str,
      "answer": str,                   # gold 答案；SQuAD 不可回答题为 ""
      "is_impossible": bool,
      "context": [{"title": str, "sentences": [str, ...]}, ...],
      "supporting_facts": {"title": [str, ...], "sent_id": [int, ...]},
      "gold_context": [str, ...],      # 支撑句文本（不可回答题为空）
    }
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from typing import Any, Iterator, Mapping, Sequence

from ..common.logging_utils import get_logger
from ..common.schema import FeatureRow, MetricRecord, Prediction, RAGSample  # noqa: F401
from ..common.text import split_sentences

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG = get_logger(__name__)

#: SQuAD v2 在线标识
SQUAD_ID = "rajpurkar/squad_v2"
#: HotpotQA 在线标识与配置
HOTPOTQA_ID = "hotpotqa/hotpot_qa"
HOTPOTQA_CONFIG = "distractor"

#: 本地离线副本（仅在 datasets 不可用/联网失败时使用）
LOCAL_SQUAD_FILES = {"train": "data/raw/squad/train-v2.0.json",
                     "validation": "data/raw/squad/dev-v2.0.json"}
LOCAL_HOTPOTQA_FILES = {"validation": "data/raw/hotpotqa/validation-*.parquet",
                        "train": "data/raw/hotpotqa/train-*.parquet"}

#: 句切分见 `src/common/text.py::split_sentences`（近似切分，原型阶段需人工确认）


class SplitView(Sequence[Mapping[str, Any]]):
    """一个数据集划分的统一视图：既包住在线的 HF Dataset，也包住本地副本读出的行。

    属性:
        dataset_id / config / split / origin / revision / local_file / local_sha256
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, dataset_id: str, config: str | None,
                 split: str, origin: str, revision: str | None = None,
                 revision_source: str | None = None,
                 local_file: str | None = None, local_sha256: str | None = None) -> None:
        self._rows: list[Mapping[str, Any]] = list(rows)
        self.dataset_id = dataset_id
        self.config = config
        self.split = split
        self.origin = origin
        self.revision = revision
        self.revision_source = revision_source
        self.local_file = local_file
        self.local_sha256 = local_sha256

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._rows)

    @property
    def rows(self) -> list[Mapping[str, Any]]:
        return self._rows

    def provenance(self) -> dict[str, Any]:
        """写入产物 _meta / construct_log 的溯源信息（契约 §4）。"""
        info: dict[str, Any] = {"dataset_id": self.dataset_id, "config": self.config,
                                "split": self.split, "origin": self.origin,
                                "n_rows": len(self._rows)}
        if self.revision:
            info["revision"] = self.revision
            if self.revision_source:
                info["revision_source"] = self.revision_source
        if self.local_file:
            info["local_file"] = self.local_file
            info["local_sha256"] = self.local_sha256
        return info


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: 本进程内在线加载失败的原因（非 None 时不再重复联网，避免每次都等 huggingface_hub 重试）
_ONLINE_FAILURE: dict[str, str | None] = {"reason": None}

#: huggingface_hub 在网络抖动时会刷大量 "Retrying in Ns" 行，降级到 ERROR（不影响我们自己的告警）
for _noisy in ("huggingface_hub", "huggingface_hub.file_download", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


def reset_online_state() -> None:
    """清空"本进程内在线不可用"的记忆（供测试与交互式排查使用）。"""
    _ONLINE_FAILURE["reason"] = None


def online_available() -> bool:
    """本进程内是否仍会尝试在线加载。"""
    return _ONLINE_FAILURE["reason"] is None


def offline_requested() -> bool:
    """是否被要求强制离线（`HF_HUB_OFFLINE=1` 或项目自己的 `C09_OFFLINE=1`）。"""
    return any(os.environ.get(name, "").strip() in ("1", "true", "True", "yes")
               for name in ("C09_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"))


def _hf_home() -> pathlib.Path:
    home = os.environ.get("HF_HOME")
    return pathlib.Path(home) if home else pathlib.Path.home() / ".cache" / "huggingface"


def cached_revision(dataset_id: str, config: str | None = None) -> str | None:
    """从 HF 缓存目录反解数据集 commit 哈希（`datasets/{owner}___{name}/{config}/{version}/{sha}`）。

    取最近修改的那个 40 位十六进制目录名；找不到时返回 None（不联网）。
    """
    root = _hf_home() / "datasets" / dataset_id.replace("/", "___")
    if not root.exists():
        return None
    candidates: list[tuple[float, str]] = []
    for config_dir in root.iterdir():
        if not config_dir.is_dir() or (config and config_dir.name != config):
            continue
        for version_dir in config_dir.iterdir():
            if not version_dir.is_dir():
                continue
            for sha_dir in version_dir.iterdir():
                name = sha_dir.name
                if sha_dir.is_dir() and len(name) == 40 and all(c in "0123456789abcdef" for c in name):
                    candidates.append((sha_dir.stat().st_mtime, name))
    if not candidates:
        return None
    return max(candidates)[1]


def _hf_revision(dataset_id: str, revision: str | None, config: str | None = None) -> tuple[str | None, str]:
    """尽力取上游提交哈希，返回 (revision, 来源)。

    来源：`hub`（联网取到）/ `cache`（从 HF 缓存目录反解）/ `config`（配置里显式给定）/ `missing`。
    """
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        cached = cached_revision(dataset_id, config)
        return (cached, "cache") if cached else (revision, "config" if revision else "missing")
    try:
        info = HfApi().dataset_info(dataset_id, revision=revision)
        sha = getattr(info, "sha", None)
        if sha:
            return str(sha), "hub"
    except Exception as exc:  # noqa: BLE001 - 网络问题不应让取数失败
        LOG.warning("联网获取 %s 的 revision 失败（%r），尝试从 HF 缓存反解", dataset_id, exc)
    cached = cached_revision(dataset_id, config)
    if cached:
        LOG.info("从 HF 缓存反解 %s 的 revision：%s", dataset_id, cached[:12])
        return cached, "cache"
    if revision:
        return revision, "config"
    LOG.warning("%s 的 revision 无法确定（无网络且无缓存），元信息中记为 None", dataset_id)
    return None, "missing"


def _iter_squad_local(path: pathlib.Path) -> list[Mapping[str, Any]]:
    """读 SQuAD v2 JSON 并摊平为与 HF 同构的行（id/title/context/question/answers）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[Mapping[str, Any]] = []
    for article in payload.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                answers = qa.get("answers", []) or []
                rows.append({
                    "id": qa.get("id", ""),
                    "title": title,
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": {"text": [a.get("text", "") for a in answers],
                                "answer_start": [int(a.get("answer_start", -1)) for a in answers]},
                })
    return rows


def _iter_hotpotqa_local(path: pathlib.Path) -> list[Mapping[str, Any]]:
    """读 HotpotQA distractor parquet（列名与 HF 一致）。"""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"读取 {path.name} 需要 pyarrow（pip install -r requirements.txt）；"
            "或安装 datasets 并联网走在线加载"
        ) from exc
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _load_local(dataset_id: str, config: str | None, split: str) -> SplitView:
    hf_split = "validation" if split in ("dev", "validation") else split
    if dataset_id == SQUAD_ID:
        rel = LOCAL_SQUAD_FILES.get(hf_split)
        if rel is None:
            raise ValueError(f"SQuAD 本地副本不支持划分 {split!r}（见 data/raw/）")
        path = ROOT / rel
        rows = _iter_squad_local(path)
    elif dataset_id == HOTPOTQA_ID:
        pattern = LOCAL_HOTPOTQA_FILES.get(hf_split)
        if pattern is None:
            raise ValueError(f"HotpotQA 本地副本不支持划分 {split!r}（见 data/raw/）")
        matches = sorted((ROOT / pattern).parent.glob(pathlib.Path(pattern).name))
        if not matches:
            raise FileNotFoundError(
                f"未找到本地副本 {pattern}；请安装 datasets 联网加载，或先执行 scripts/download_data.sh"
            )
        rows = []
        for p in matches:
            rows.extend(_iter_hotpotqa_local(p))
        path = matches[0] if len(matches) == 1 else None
    else:
        raise FileNotFoundError(
            f"未安装 datasets 且无 {dataset_id!r} 的本地副本；请先 pip install datasets 联网加载"
        )

    sha = _sha256(path) if path is not None else None
    LOG.info("本地副本加载：%s split=%s 行数=%d", dataset_id, hf_split, len(rows))
    return SplitView(rows, dataset_id=dataset_id, config=config, split=hf_split, origin="local",
                     local_file=str(path.relative_to(ROOT)) if path is not None else str(pattern),
                     local_sha256=sha)


def load_dataset_split(dataset_id: str, config: str | None = None, split: str = "validation",
                       revision: str | None = None, offline: bool = False) -> SplitView:
    """加载指定数据集的某个划分；revision 固定后写入产物元信息（契约 §4）。

    在线优先（`datasets.load_dataset`），失败时降级到 `data/raw/` 本地副本并记录 origin。
    `offline=True` 或环境变量 `HF_HUB_OFFLINE` / `C09_OFFLINE` 为真时，直接使用 HF 缓存、不发网络请求；
    一旦本进程内在线加载失败，后续加载也不再重复联网（见 `reset_online_state`）。
    """
    if offline or offline_requested():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        LOG.info("离线模式：直接使用 HF 缓存，不发网络请求（%s）", dataset_id)
    elif not online_available():
        LOG.info("本进程内在线加载此前已失败（%s），直接使用本地副本：%s",
                 _ONLINE_FAILURE["reason"], dataset_id)
        return _load_local(dataset_id, config, split)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        LOG.warning("未安装 datasets，改用本地副本（data/raw/）：%s", dataset_id)
        return _load_local(dataset_id, config, split)

    try:
        kwargs: dict[str, Any] = {"split": split}
        if revision:
            kwargs["revision"] = revision
        ds = load_dataset(dataset_id, config, **kwargs)
        rows = [dict(row) for row in ds]
        rev, rev_source = _hf_revision(dataset_id, revision, config)
        LOG.info("在线加载：%s split=%s 行数=%d revision=%s（来源 %s）",
                 dataset_id, split, len(rows), rev, rev_source)
        return SplitView(rows, dataset_id=dataset_id, config=config, split=split, origin="hf",
                         revision=rev, revision_source=rev_source)
    except Exception as exc:  # noqa: BLE001 - 网络/上游问题一律降级，不中断实验
        _ONLINE_FAILURE["reason"] = repr(exc)
        LOG.warning("在线加载 %s 失败（%r）；本进程内后续加载不再重试联网，改用本地副本 data/raw/",
                    dataset_id, exc)
        return _load_local(dataset_id, config, split)


def _answerable_support_sentence(context: str, answer_start: int) -> tuple[int, str]:
    """由答案起始偏移定位所在句，返回 (句号, 句文本)。"""
    sentences = split_sentences(context)
    offset = 0
    for idx, sentence in enumerate(sentences):
        start = context.find(sentence, offset)
        if start < 0:
            start = offset
        end = start + len(sentence)
        if answer_start < 0 or start <= answer_start < end:
            return idx, sentence
        offset = end
    if sentences:
        return len(sentences) - 1, sentences[-1]
    return -1, ""


def normalize_squad(example: Mapping[str, Any]) -> dict[str, Any]:
    """把 SQuAD v2 样本规范化为统一结构（保留不可回答标记）。

    备注: SQuAD 无句级标注，`supporting_facts` 由答案起始偏移推导（见 `split_sentences` 的
    近似说明）；不可回答样本的 `gold_context` 为空列表，`is_impossible=True`。
    """
    answers = example.get("answers") or {}
    texts = [t for t in (answers.get("text") or []) if t]
    starts = [int(s) for s in (answers.get("answer_start") or [])]
    context = example.get("context", "") or ""
    title = example.get("title", "") or ""
    impossible = not texts

    if impossible:
        gold_context: list[str] = []
        support = {"title": [], "sent_id": []}
    else:
        sent_idx, sentence = _answerable_support_sentence(context, starts[0] if starts else -1)
        gold_context = [sentence] if sentence else []
        support = {"title": [title] if sentence else [], "sent_id": [sent_idx] if sentence else []}

    return {
        "id": example.get("id", ""),
        "source": "squad",
        "hf_split": example.get("hf_split", "dev"),
        "split_tag": "dev",
        "row_index": int(example.get("row_index", -1)),
        "question": example.get("question", ""),
        "answer": texts[0] if texts else "",
        "is_impossible": impossible,
        "context": [{"title": title, "sentences": split_sentences(context)}],
        "supporting_facts": support,
        "gold_context": gold_context,
    }


def normalize_hotpotqa(example: Mapping[str, Any]) -> dict[str, Any]:
    """把 HotpotQA distractor 样本规范化为统一结构。

    `context` 由 `{"title": [...], "sentences": [[句...], ...]}` 转为
    `[{"title": str, "sentences": [str, ...]}, ...]`；`gold_context` 取 `supporting_facts`
    指向的句子文本。
    """
    raw_context = example.get("context") or {}
    titles = list(raw_context.get("title") or [])
    sentences = list(raw_context.get("sentences") or [])
    context = [{"title": str(t), "sentences": [str(s) for s in (sentences[i] if i < len(sentences) else [])]}
               for i, t in enumerate(titles)]

    facts = example.get("supporting_facts") or {}
    fact_titles = [str(t) for t in (facts.get("title") or [])]
    fact_sents = [int(s) for s in (facts.get("sent_id") or [])]
    by_title = {p["title"]: p["sentences"] for p in context}
    gold_context: list[str] = []
    for title, sent_id in zip(fact_titles, fact_sents):
        sents = by_title.get(title, [])
        if 0 <= sent_id < len(sents):
            gold_context.append(sents[sent_id])

    return {
        "id": example.get("id", ""),
        "source": "hotpotqa",
        "hf_split": example.get("hf_split", "validation"),
        "split_tag": "dev",
        "row_index": int(example.get("row_index", -1)),
        "question": example.get("question", ""),
        "answer": example.get("answer", "") or "",
        "is_impossible": False,
        "context": context,
        "supporting_facts": {"title": fact_titles, "sent_id": fact_sents},
        "gold_context": gold_context,
    }


def normalize(example: Mapping[str, Any], source: str, row_index: int = -1,
              hf_split: str = "dev") -> dict[str, Any]:
    """按 source 分派到对应规范化函数，并补上溯源字段。"""
    enriched = dict(example)
    enriched.setdefault("row_index", row_index)
    enriched.setdefault("hf_split", hf_split)
    if source == "squad":
        return normalize_squad(enriched)
    if source == "hotpotqa":
        return normalize_hotpotqa(enriched)
    raise ValueError(f"未知数据来源：{source!r}（应为 'squad' 或 'hotpotqa'）")


def source_of(dataset_id: str) -> str:
    """数据集标识 → 契约中的 `source` 取值。"""
    if dataset_id == SQUAD_ID:
        return "squad"
    if dataset_id == HOTPOTQA_ID:
        return "hotpotqa"
    raise ValueError(f"未知数据集标识：{dataset_id!r}")


def load_normalized(dataset_id: str, config: str | None = None, split: str = "validation",
                    revision: str | None = None, limit: int | None = None) -> tuple[SplitView, list[dict[str, Any]]]:
    """加载并规范化一个划分，返回 (视图, 规范化样本列表)。

    `limit` 只影响返回的规范化列表（便于冒烟），不改变视图本身。
    """
    view = load_dataset_split(dataset_id, config, split=split, revision=revision)
    source = source_of(dataset_id)
    rows: list[dict[str, Any]] = []
    for i, example in enumerate(view):
        if limit is not None and len(rows) >= limit:
            break
        rows.append(normalize(example, source, row_index=i, hf_split=view.split))
    return view, rows
