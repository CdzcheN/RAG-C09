#!/usr/bin/env python3
"""抓取课题 C09 参考文献的权威元数据，输出 refs/references.json。

数据来源（均在线公开、可复核）：
  1. arXiv API      —— 标题、作者全名列表、首次提交/更新日期、arXiv 分类、journal_ref
  2. OpenAlex       —— 出版记录（DOI、卷期页码、来源刊名、会议来源），数据源自 Crossref
  3. Crossref API   —— OpenAlex 的兜底来源（同一上游数据）；OpenAlex 限流时自动切换
  4. ACL Anthology  —— DOI 前缀为 10.18653/v1/ 的 ACL 系论文，直接取其官方 BibTeX 原文

匹配正确性保障：检索结果必须“标题归一化后完全一致”且“作者姓氏有交集”才被接受，
否则视为未匹配。所有字段均来自上述接口的真实返回，查不到就留空，绝不按记忆补写。

用法: python3 scripts/fetch_reference_metadata.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
IDS_FILE = ROOT / "refs" / "arxiv_ids.txt"
OUT_FILE = ROOT / "refs" / "references.json"
MAILTO = "c09-course-project@example.edu"  # OpenAlex / Crossref 礼貌池标识
UA = "RAG-C09-reference-collector/1.0 (mailto:%s)" % MAILTO
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_REPO_NAMES = {"arXiv (Cornell University)", "arXiv"}
# 元数据 schema 版本：变更抓取逻辑时递增，使旧缓存自动失效
SCHEMA_VERSION = 4

# 次级出版形态（Findings / Workshop / Demo 等）：多条记录都匹配时优先取主会版本
SECONDARY_VENUE = re.compile(
    r"findings|workshop|demonstration|system demo|student research|dialdoc|tutorial", re.I
)

# 各来源的文献类型 -> BibTeX 类型
OPENALEX_KIND = {
    "proceedings-article": "inproceedings",
    "conference-paper": "inproceedings",
    "book-chapter": "inproceedings",
    "article": "article",
    "review": "article",
}
CROSSREF_KIND = {
    "proceedings-article": "inproceedings",
    "book-chapter": "inproceedings",
    "journal-article": "article",
}

_openalex_disabled = False  # 一旦被限流，后续条目直接切换到 Crossref


# --------------------------------------------------------------------------- 网络
def http_get(url: str, tries: int = 3, timeout: int = 45) -> bytes:
    """用 curl 抓取（带重试）。

    - arXiv / OpenAlex 前置的 CDN 会以 406 拒绝 Python urllib 的请求（TLS/头部指纹），
      同一 UA 下 curl 正常，故统一走 curl；
    - 不使用 curl 自带的 --retry：`--retry-all-errors` 遇到 429 会遵循 Retry-After
      而长时间挂起，重试统一由本函数负责；
    - curl 的 --max-time 不覆盖 DNS 解析阶段，故再套一层 Python 侧硬超时。
    """
    cmd = [
        "curl", "-sS", "-fL",
        "--connect-timeout", "15", "--max-time", str(timeout),
        "-A", UA, "-H", "Accept: */*", url,
    ]
    proc = subprocess.CompletedProcess(cmd, 1, b"", b"")
    for attempt in range(tries):
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout + 15)
        except subprocess.TimeoutExpired:
            print(f"[warn] 硬超时 {timeout + 15}s: {url[:90]}", file=sys.stderr)
            continue
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        err = proc.stderr.decode("utf-8", "replace")
        if "429" in err:  # 被限流，重试无意义
            raise RuntimeError(f"HTTP 429 rate limited: {url}")
        time.sleep(1 + attempt)
    raise RuntimeError(f"GET failed: {url} (curl exit {proc.returncode}: {err[:200]})")


# --------------------------------------------------------------------------- 输入
def parse_id_file() -> list[dict]:
    items = []
    for raw in IDS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        group, arxiv_id, key, note = line.split(None, 3)
        items.append({"group": group, "arxiv_id": arxiv_id, "key": key, "note": note})
    return items


def fetch_arxiv(ids: list[str]) -> dict[str, dict]:
    """按 ID 批量取 arXiv 元数据（每批 20 条）。"""
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 20):
        chunk = ids[i : i + 20]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}
        )
        root = ET.fromstring(http_get(url, tries=3, timeout=60))
        for e in root.findall("a:entry", ARXIV_NS):
            aid = re.sub(r"v\d+$", "", e.find("a:id", ARXIV_NS).text.rsplit("/", 1)[-1])
            jr = e.find("arxiv:journal_ref", ARXIV_NS)
            out[aid] = {
                "title": " ".join(e.find("a:title", ARXIV_NS).text.split()),
                "authors": [
                    a.find("a:name", ARXIV_NS).text for a in e.findall("a:author", ARXIV_NS)
                ],
                "published": e.find("a:published", ARXIV_NS).text[:10],
                "updated": e.find("a:updated", ARXIV_NS).text[:10],
                "arxiv_primary_category": e.find("arxiv:primary_category", ARXIV_NS).get("term"),
                "arxiv_journal_ref": jr.text.strip() if jr is not None else "",
            }
        time.sleep(1)
    return out


# --------------------------------------------------------------------------- 匹配工具
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def surnames(names: list[str]) -> set[str]:
    """取作者姓氏用于交叉验证，兼容 "Lewis, Patrick" 与 "Patrick Lewis" 两种写法。"""
    out = set()
    for n in names:
        if not n:
            continue
        part = n.split(",")[0] if "," in n else n
        out.add(norm(part.split()[-1]))
    return out - {""}


def split_pages(pages: str | None) -> tuple[str | None, str | None]:
    """把 "9332-9346" / "2383–2392" 拆成 (首页, 末页)。"""
    if not pages:
        return None, None
    m = re.match(r"^\s*([A-Za-z]?\d+)\s*[-–—]+\s*([A-Za-z]?\d+)\s*$", pages)
    if not m:
        return pages.strip() or None, None
    return m.group(1), m.group(2)


def acl_id_from_doi(doi: str | None) -> str | None:
    m = re.match(r"^https?://doi\.org/10\.18653/v1/(.+)$", doi or "")
    return m.group(1).upper() if m else None


def record_rank(r: dict) -> tuple:
    """多条候选记录并存时的取舍顺序：正式出版 > 主会 > 高被引 > 有 DOI / 有年份。"""
    return (
        r["kind"] == "misc",
        r["venue"] is None,
        bool(r["venue"] and SECONDARY_VENUE.search(r["venue"])),
        -(r.get("citations") or 0),
        r["doi"] is None,
        r["publication_year"] is None,
    )


# --------------------------------------------------------------------------- OpenAlex
def pick_venue(w: dict) -> str | None:
    """从 locations 中挑一个“真期刊/会议”来源名，跳过机构库与 arXiv。"""
    good_types = {"journal", "conference", "book series", "proceedings", "ebook platform"}
    for loc in w.get("locations") or []:
        src = loc.get("source") or {}
        name = src.get("display_name")
        if name and src.get("type") in good_types and name not in ARXIV_REPO_NAMES:
            return name
    src = (w.get("primary_location") or {}).get("source") or {}
    name = src.get("display_name")
    if name and name not in ARXIV_REPO_NAMES and "Discovery" not in name and "Repository" not in name:
        return name
    return None


def fetch_openalex(title: str, authors: list[str]) -> dict:
    """按标题检索 OpenAlex，并用标题+作者双重条件验证；优先取正式出版记录。"""
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": title, "per-page": 6, "mailto": MAILTO}
    )
    data = json.loads(http_get(url, tries=2, timeout=40))
    want, mine = norm(title), surnames(authors)
    candidates: list[dict] = []
    for w in data.get("results", []):
        if norm(w.get("title")) != want:
            continue
        oa_authors = [a["author"]["display_name"] for a in w.get("authorships", [])]
        overlap = mine & surnames(oa_authors)
        if not overlap:
            continue  # 同名不同文的记录，丢弃
        bib = w.get("biblio") or {}
        doi = (w.get("ids") or {}).get("doi") or w.get("doi")
        otype = w.get("type")
        candidates.append(
            {
                "matched": True,
                "source": "openalex",
                "source_id": w.get("id"),
                "source_title": w.get("title"),
                "source_first_author": (oa_authors or [None])[0],
                "author_surname_overlap": sorted(overlap),
                "doi": doi,
                "source_type": otype,
                "kind": OPENALEX_KIND.get(otype or "", "misc"),
                "publication_year": w.get("publication_year"),
                "citations": w.get("cited_by_count"),
                "venue": pick_venue(w),
                "volume": bib.get("volume"),
                "issue": bib.get("issue"),
                "first_page": bib.get("first_page"),
                "last_page": bib.get("last_page"),
                "acl_anthology_id": acl_id_from_doi(doi),
            }
        )
    if not candidates:
        return {"matched": False, "source": "openalex"}

    candidates.sort(key=record_rank)
    return candidates[0]


# --------------------------------------------------------------------------- Crossref
def fetch_crossref(title: str, authors: list[str]) -> dict:
    """Crossref 兜底：字段与 OpenAlex 同源，用于 OpenAlex 限流或未收录的情况。

    先按 bibliographic 检索，召回不足时再按 title 字段检索一次。
    """
    want, mine = norm(title), surnames(authors)
    select = (
        "DOI,title,container-title,volume,issue,page,type,author,"
        "published,published-print,published-online,event,is-referenced-by-count"
    )
    candidates: list[dict] = []
    for query_field in ("query.bibliographic", "query.title"):
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {query_field: title, "rows": 20, "mailto": MAILTO, "select": select}
        )
        data = json.loads(http_get(url, tries=2, timeout=45))
        for it in (data.get("message") or {}).get("items", []):
            hit_title = (it.get("title") or [""])[0]
            if norm(hit_title) != want:
                continue
            cr_authors = [a.get("family") or "" for a in it.get("author", [])]
            overlap = mine & surnames(cr_authors)
            if not overlap:
                continue
            year = None
            for k in ("published-print", "published-online", "published", "issued"):
                parts = (it.get(k) or {}).get("date-parts") or []
                if parts and parts[0]:
                    year = parts[0][0]
                    break
            first, last = split_pages(it.get("page"))
            doi = it.get("DOI")
            ctype = it.get("type")
            venue = (it.get("container-title") or [None])[0]
            if not venue and it.get("event", {}).get("name"):
                venue = it["event"]["name"]
            candidates.append(
                {
                    "matched": True,
                    "source": "crossref",
                    "source_id": f"https://doi.org/{doi}",
                    "source_title": hit_title,
                    "source_first_author": (cr_authors or [None])[0],
                    "author_surname_overlap": sorted(overlap),
                    "doi": f"https://doi.org/{doi}" if doi else None,
                    "source_type": ctype,
                    "kind": CROSSREF_KIND.get(ctype or "", "misc"),
                    "publication_year": year,
                    "citations": it.get("is-referenced-by-count"),
                    "venue": venue,
                    "volume": it.get("volume"),
                    "issue": it.get("issue"),
                    "first_page": first,
                    "last_page": last,
                    "acl_anthology_id": acl_id_from_doi(f"https://doi.org/{doi}" if doi else None),
                }
            )
        if candidates:
            break
        time.sleep(0.3)
    if not candidates:
        return {"matched": False, "source": "crossref"}

    candidates.sort(key=record_rank)
    return candidates[0]


def fetch_metadata(title: str, authors: list[str]) -> dict:
    """OpenAlex 优先，失败（含限流）时退回 Crossref。"""
    global _openalex_disabled
    errors: list[str] = []
    if not _openalex_disabled:
        try:
            rec = fetch_openalex(title, authors)
            if rec.get("matched"):
                return rec
            errors.append("openalex: no title+author match")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)[:200]
            errors.append(f"openalex: {msg}")
            if "429" in msg:
                _openalex_disabled = True
                print("[warn] OpenAlex 被限流，后续条目改用 Crossref", file=sys.stderr)
    try:
        rec = fetch_crossref(title, authors)
        if rec.get("matched"):
            return rec
        errors.append("crossref: no title+author match")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"crossref: {str(exc)[:200]}")
    return {"matched": False, "errors": errors}


# --------------------------------------------------------------------------- ACL 官方 BibTeX
def fetch_acl_bib(acl_id: str) -> str:
    """取 ACL Anthology 官方 BibTeX 原文（权威的 booktitle/pages/publisher/address）。

    Anthology 的 ID 大小写不统一：老式 ID 形如 D16-1264（大写），
    合集式 ID 形如 2023.emnlp-main.397（小写），因此按序尝试两种写法。
    """
    for cand in dict.fromkeys([acl_id.lower(), acl_id.upper()]):
        try:
            raw = http_get(f"https://aclanthology.org/{cand}.bib", tries=1, timeout=30).decode(
                "utf-8", "replace"
            )
        except Exception:  # noqa: BLE001 - 404 属正常，换下一种写法
            continue
        if raw.lstrip().startswith("@"):
            return raw
    print(f"[warn] ACL bib 抓取失败 {acl_id}", file=sys.stderr)
    return ""


# --------------------------------------------------------------------------- 主流程
def load_cache() -> dict[str, dict]:
    """把已有的 references.json 当缓存：只复用 schema 版本一致且抓取完成的条目。"""
    if not OUT_FILE.exists():
        return {}
    try:
        old = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 缓存损坏时重新抓取
        return {}
    if not isinstance(old, list):
        return {}
    return {
        it["key"]: it
        for it in old
        if isinstance(it, dict)
        and it.get("_schema") == SCHEMA_VERSION
        and (it.get("published_venue") or {}).get("matched") is True  # 只复用抓取成功的条目
    }


def main() -> int:
    items = parse_id_file()
    print(f"[info] {len(items)} 条候选文献", file=sys.stderr)

    cache = load_cache()
    cached_keys = set(cache)
    todo = [it for it in items if it["key"] not in cache]

    arxiv = fetch_arxiv([it["arxiv_id"] for it in items])
    missing = [it["arxiv_id"] for it in items if it["arxiv_id"] not in arxiv]
    if missing:
        print(f"[error] arXiv 未返回: {missing}", file=sys.stderr)
        return 1

    for it in items:
        it.update(arxiv[it["arxiv_id"]])
        it["_schema"] = SCHEMA_VERSION
        if it["key"] in cache:
            old = cache[it["key"]]
            for k in ("published_venue", "acl_bib"):
                if k in old:
                    it[k] = old[k]

    def dump() -> None:
        OUT_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[info] 命中缓存 {len(cached_keys)} 条，需抓取 {len(todo)} 条", file=sys.stderr)
    for i, it in enumerate(todo, 1):
        it["published_venue"] = fetch_metadata(it["title"], it["authors"])
        acl = (it["published_venue"] or {}).get("acl_anthology_id")
        it["acl_bib"] = fetch_acl_bib(acl) if acl else ""
        print(
            f"[{i:2}/{len(todo)}] {it['key']:26} "
            f"match={it['published_venue'].get('matched')} "
            f"src={it['published_venue'].get('source', '-')} acl={acl or '-'}",
            file=sys.stderr,
        )
        dump()  # 每条完成即落盘，被中断后重跑会跳过已完成条目
        time.sleep(0.3)

    failed = [it["key"] for it in items if (it.get("published_venue") or {}).get("matched") is None]
    dump()

    matched = sum(1 for it in items if (it.get("published_venue") or {}).get("matched"))
    by_src: dict[str, int] = {}
    for it in items:
        src = (it.get("published_venue") or {}).get("source", "none")
        by_src[src] = by_src.get(src, 0) + 1
    print(f"[done] 写入 {OUT_FILE}；匹配 {matched}/{len(items)}，来源分布 {by_src}")
    if failed:
        print(f"[error] 以下条目未匹配到出版记录，可重跑本脚本重试: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
