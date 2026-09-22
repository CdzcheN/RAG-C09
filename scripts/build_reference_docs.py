#!/usr/bin/env python3
"""由 refs/references.json 生成 BibTeX（refs/references.bib）与文献清单（refs/参考文献清单.md）。

字段来源优先级：
  1. ACL Anthology 官方 BibTeX 原文（若该文有 ACL 版）—— booktitle/pages/publisher/doi 等最权威；
  2. OpenAlex（数据源自 Crossref）—— 期刊/会议名、卷期页码、DOI；
  3. arXiv API —— 标题、作者、年份；缺失出版信息时按预印本著录。

脚本只做结构化转换与格式整理，不引入任何外部知识，也不补写任何查不到的字段。
BibTeX key 沿用 refs/arxiv_ids.txt 中自定义的 key，便于与本地 PDF 文件名一一对应。

用法: python3 scripts/build_reference_docs.py
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFS = ROOT / "refs"
GROUP_NAMES = {
    "A": "A. RAG 基础与检索增强",
    "B": "B. 幻觉与事实性评估",
    "C": "C. 幻觉检测与不确定性度量",
    "D": "D. 一致性、自然语言推断与基准数据",
}
# ACL 官方字段的著录顺序
ACL_FIELD_ORDER = [
    "title", "author", "editor", "booktitle", "month", "year",
    "address", "publisher", "pages", "doi", "url",
]


# --------------------------------------------------------------------------- BibTeX 解析
def _extract_braced(s: str, i: int) -> tuple[str, int]:
    """返回 s[i] 处 '{' 所配对的内容（不含外层花括号）与结束位置。"""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    raise ValueError("unbalanced braces")


def _split_fields(body: str) -> list[str]:
    """按顶层逗号切分字段定义（忽略花括号与引号内的逗号）。"""
    parts, buf, depth, in_quote = [], [], 0, False
    for ch in body:
        if ch == '"' and depth == 0:
            in_quote = not in_quote
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0 and not in_quote:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def parse_bibtex(raw: str) -> tuple[str, dict[str, str]]:
    """解析单条 BibTeX，返回 (条目类型, {字段名: 值})。"""
    text = raw.strip()
    start = text.index("{")
    kind = text[1:start].strip().lower()
    body, _ = _extract_braced(text, start)
    fields: dict[str, str] = {}
    for idx, part in enumerate(_split_fields(body)):
        if idx == 0 or "=" not in part:  # 第 0 段是引用键
            continue
        name, _, value = part.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] + value[-1] in {"{}", '""'}:
            value = value[1:-1]
        fields[name.strip().lower()] = re.sub(r"\s+", " ", value).strip()
    return kind, fields


# --------------------------------------------------------------------------- 条目构造
def strip_ver(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def clean_doi(doi: str | None) -> str:
    return re.sub(r"^https?://doi\.org/", "", doi or "")


def arxiv_trace(item: dict) -> list[tuple[str, str]]:
    """所有条目都保留 arXiv 溯源字段，便于复核与预印本引用。"""
    out = [("eprint", strip_ver(item["arxiv_id"])), ("archivePrefix", "arXiv")]
    if item.get("arxiv_primary_category"):
        out.append(("primaryClass", item["arxiv_primary_category"]))
    return out


def bib_entry(item: dict) -> str:
    pub = item.get("published_venue") or {}
    if item.get("acl_bib"):
        kind, fields = parse_bibtex(item["acl_bib"])
        ordered = [(k, fields[k]) for k in ACL_FIELD_ORDER if k in fields]
        ordered += [(k, v) for k, v in fields.items() if k not in ACL_FIELD_ORDER]
    else:
        venue, doi = pub.get("venue"), clean_doi(pub.get("doi"))
        kind = pub.get("kind") or "misc"  # 抓取阶段已把各来源类型归一化为 BibTeX 类型
        ordered = [("title", "{" + item["title"] + "}"), ("author", " and ".join(item["authors"]))]
        ordered.append(("year", str(pub.get("publication_year") or item["published"][:4])))
        if kind == "inproceedings" and venue:
            ordered.append(("booktitle", venue))
        elif kind == "article" and venue:
            ordered.append(("journal", venue))
        elif kind == "misc":
            # 无正式出版来源：按预印本著录，不携带 volume/pages（避免与预印本状态矛盾）
            ordered.append(("howpublished", "arXiv preprint"))
        if kind != "misc":
            if pub.get("volume"):
                ordered.append(("volume", str(pub["volume"])))
            if pub.get("issue"):
                ordered.append(("number", str(pub["issue"])))
            if pub.get("first_page") and pub.get("last_page"):
                ordered.append(("pages", f"{pub['first_page']}--{pub['last_page']}"))
            elif pub.get("first_page"):
                ordered.append(("pages", str(pub["first_page"])))
        if doi:
            ordered.append(("doi", doi))
        if not venue:
            ordered.append(("note", "预印本；未匹配到正式出版记录"))

    ordered += arxiv_trace(item)
    width = max(len(k) for k, _ in ordered)
    body = ",\n".join(f"  {k:<{width}} = {{{v}}}" for k, v in ordered)
    return f"@{kind}{{{item['key']},\n{body}\n}}\n"


# --------------------------------------------------------------------------- 清单构造
def entry_meta(item: dict) -> dict:
    """汇总用于清单展示的字段（优先正式出版信息）。"""
    pub = item.get("published_venue") or {}
    acl = {}
    if item.get("acl_bib"):
        _, acl = parse_bibtex(item["acl_bib"])
    year = acl.get("year") or pub.get("publication_year") or item["published"][:4]
    venue = acl.get("booktitle") or pub.get("venue") or "arXiv 预印本"
    pages = acl.get("pages") or ""
    if not pages and pub.get("first_page") and pub.get("last_page"):
        pages = f"{pub['first_page']}–{pub['last_page']}"
    doi = clean_doi(acl.get("doi") or pub.get("doi"))
    return {"year": year, "venue": venue, "pages": pages, "doi": doi}


def md_row(idx: int, item: dict) -> str:
    meta = entry_meta(item)
    authors = item["authors"]
    shown = authors[0] + (" 等" if len(authors) > 1 else "")
    link = (
        f"[{meta['doi']}](https://doi.org/{meta['doi']})"
        if meta["doi"]
        else f"[arXiv:{strip_ver(item['arxiv_id'])}](https://arxiv.org/abs/{strip_ver(item['arxiv_id'])})"
    )
    flag = "★ " if "起步文献" in item.get("note", "") else ""
    return (
        f"| {idx} | {item['group']} | {flag}`{item['key']}` | {item['title']} | {shown} | {meta['year']} | "
        f"{meta['venue']} | {meta['pages'] or '—'} | {link} | [{item['key']}.pdf](pdf/{item['key']}.pdf) |"
    )


def main() -> int:
    items = json.loads((REFS / "references.json").read_text(encoding="utf-8"))
    items.sort(key=lambda it: (it["group"], it["key"].lower()))

    # 质量门禁：ACL 官方条目的标题必须与 arXiv 标题一致，否则视为错配
    mismatched: list[str] = []
    for it in items:
        if not it.get("acl_bib"):
            continue
        _, fields = parse_bibtex(it["acl_bib"])
        acl_title = re.sub(r"[{}]", "", fields.get("title", ""))
        if re.sub(r"[^a-z0-9]+", " ", acl_title.lower()).strip() != re.sub(
            r"[^a-z0-9]+", " ", it["title"].lower()
        ).strip():
            mismatched.append(f"{it['key']}: ACL 条目「{acl_title}」≠ arXiv「{it['title']}」")
    for m in mismatched:
        print(f"[warn] 标题不一致 {m}", file=sys.stderr)

    (REFS / "references.bib").write_text(
        "% 课题 C09 参考文献（自动生成，请勿手工编辑）\n"
        "% 生成: scripts/build_reference_docs.py；元数据: refs/references.json\n"
        "% 来源: arXiv API + OpenAlex(Crossref) + ACL Anthology 官方 BibTeX\n\n"
        + "\n".join(bib_entry(it) for it in items),
        encoding="utf-8",
    )

    n_acl = sum(1 for it in items if it["acl_bib"])
    n_pub = sum(1 for it in items if (it.get("published_venue") or {}).get("matched"))
    lines = [
        "# 课题 C09 参考文献清单",
        "",
        "> 自动生成，请勿手工编辑。元数据来源：**arXiv API**（标题/作者/年份）、"
        "**OpenAlex / Crossref**（期刊会议、卷期页码、DOI）、"
        "**ACL Anthology 官方 BibTeX**（ACL 系论文的完整出版信息）。",
        "> 匹配需同时满足“标题完全一致”与“作者姓氏有交集”；同一论文存在多个出版记录时，按“正式出版 > 主会 > 高被引”的顺序取值；未匹配到正式出版记录的条目一律按预印本著录。",
        "",
        f"共 **{len(items)}** 篇，全部开放获取，PDF 已下载至 `refs/pdf/`；"
        f"其中 {n_acl} 篇取自 ACL Anthology 官方条目，{n_pub} 篇匹配到正式出版记录。",
        "",
        "## 按主题分组",
        "",
    ]
    if mismatched:
        lines += ["## ⚠ 待复核：标题交叉校验未通过", ""] + [f"- {m}" for m in mismatched] + [""]
    lines += [f"- **{name}**：{sum(1 for it in items if it['group'] == g)} 篇" for g, name in GROUP_NAMES.items()]
    lines += [
        "",
        "## 明细",
        "",
        "| # | 组 | BibTeX key | 标题 | 第一作者 | 年份 | 发表处 | 页码 | DOI / arXiv | 本地 PDF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [md_row(i, it) for i, it in enumerate(items, 1)]
    lines += [
        "",
        "> `★` 标记为课题文档中列出的起步文献。",
        "",
        "## 课题起步文献对应关系",
        "",
        "| 课题文档所列文献 | 本清单条目 |",
        "|---|---|",
        "| Lewis et al., NeurIPS 2020（RAG） | `Lewis2020RAG` |",
        "| Ji et al., ACM Computing Surveys, 2023, 55(12): 1-38 | `Ji2023Survey` |",
        "| Rajpurkar et al., EMNLP 2016（SQuAD） | `Rajpurkar2016SQuAD` |",
        "",
        "## 使用说明",
        "",
        "- `refs/references.bib`：可直接被 LaTeX 使用，或导入 Zotero / EndNote；"
        "每条都带 `eprint` / `archivePrefix` 字段，便于溯源核对。",
        "- `refs/references.json`：结构化元数据，额外记录了各来源的匹配证据"
        "（`openalex_title`、`author_surname_overlap`）与 ACL BibTeX 原文（`acl_bib`）。",
        "- `refs/pdf/MANIFEST.sha256`：PDF 完整性校验清单。",
        "- 版权提示：这些论文按其各自许可协议发布（arXiv 多为非独占许可，ACL 系为 CC BY），"
        "本目录仅用于课程课题的文献调研，不再分发。",
        "",
        "## 复现方式",
        "",
        "```bash",
        "bash    scripts/download_refs.sh              # 下载 PDF -> refs/pdf/",
        "python3 scripts/fetch_reference_metadata.py   # 抓取元数据 -> refs/references.json",
        "python3 scripts/build_reference_docs.py       # 生成 -> refs/references.bib + 本清单",
        "```",
        "",
    ]
    (REFS / "参考文献清单.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] {len(items)} 条；ACL 官方条目 {n_acl} 条；匹配到正式出版记录 {n_pub} 条")
    print("[out ] refs/references.bib, refs/参考文献清单.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
