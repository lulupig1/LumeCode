"""记忆召回的词法匹配信号：分词、BM25、实体抽取。

编码助手的记忆里最要紧的往往是精确字符串——文件名、命令、符号名
（``permissions/sandbox.py``、``ruff format``、``AsyncTaskManager``）。
这类 token 用 embedding 语义匹配容易糊掉，纯字面匹配又抓不住同义表述，
因此向量召回之外再叠加两路信号：

- **BM25**：字面关键词匹配（含 camelCase / snake_case / 路径拆分、中文 2-gram）；
- **实体重叠**：查询与记忆共享的文件路径、引号内容、标识符。

两路信号都是**加分项**：最终分数 = 余弦相似度 + BM25 加权 + 实体加权，
现有 ``min_score`` / 相对阈值仍作用在融合分上，排序行为保持向后兼容。
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------

# 只取"词"片段：字母数字、下划线、点、斜杠、反斜杠、连字符。
_SEGMENT_RE = re.compile(r"[0-9A-Za-z_./\\-]+")
# 按路径/命名分隔符拆段。
_SPLIT_RE = re.compile(r"[./\\_-]+")
# camelCase / PascalCase / HTTPServer 式边界。
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# 中文连续片段：整体不成词（无分词器），用 2-gram 近似。
_CJK_RE = re.compile(r"[一-鿿]+")

# 编码语境下几乎不携带信息的词。宁可留一点噪声，也不要把有意义的短词
# （如 `py`、`go`、`rs`）误删——它们靠 IDF 自然降权。
STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "you",
        "your",
        "have",
        "has",
        "not",
        "but",
        "can",
        "will",
        "when",
        "then",
        "than",
        "into",
        "out",
        "its",
        "it's",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "as",
        "is",
        "be",
        "or",
        "if",
        "do",
        "we",
        "i",
    }
)


def tokenize(text: str) -> list[str]:
    """把文本拆成小写词项，供 BM25 使用。

    会拆分路径（``a/b.py`` → ``a``, ``b``）、命名分隔符（``snake_case``）、
    camelCase（``AsyncTaskManager`` → ``async``, ``task``, ``manager``）；
    中文按 2-gram 处理，无需引入分词依赖。
    """
    if not text:
        return []

    tokens: list[str] = []
    for raw in _SEGMENT_RE.findall(text):
        for seg in _SPLIT_RE.split(raw):
            if not seg:
                continue
            for part in _CAMEL_RE.split(seg):
                piece = part.lower()
                if len(piece) >= 2 and piece not in STOPWORDS:
                    tokens.append(piece)

    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))

    return tokens


# ---------------------------------------------------------------------------
# 实体抽取
# ---------------------------------------------------------------------------

_ENTITY_EXTS = (
    "py|pyi|js|jsx|ts|tsx|go|rs|java|kt|swift|c|cc|cpp|h|hpp|cs|rb|php|sh|ps1|"
    "md|rst|txt|json|jsonl|yaml|yml|toml|ini|cfg|conf|lock|csv|sql|html|css"
)
# 带目录的路径（带已知扩展名），如 `lumecode/memory/recall.py`。
_PATH_ENTITY_RE = re.compile(rf"(?:[\w.~-]+[/\\])+[\w.~-]+\.(?:{_ENTITY_EXTS})\b", re.IGNORECASE)
# 裸文件名，如 `recall.py`、`pyproject.toml`。
_FILE_ENTITY_RE = re.compile(rf"\b[\w.-]+\.(?:{_ENTITY_EXTS})\b", re.IGNORECASE)
# 点分模块路径，如 `lumecode.memory.recall`。
_DOTTED_ENTITY_RE = re.compile(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){2,}\b")
# 反引号 / 引号内的内容——通常是命令、参数或专有名词。
_QUOTED_ENTITY_RE = re.compile(r"`([^`\n]{2,80})`|\"([^\"\n]{2,80})\"|“([^”\n]{2,80})”")
# snake_case 标识符。
_SNAKE_ENTITY_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
# camelCase / PascalCase 标识符。
_CAMEL_ENTITY_RE = re.compile(r"\b[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b")

_MIN_ENTITY_LEN = 2
_MAX_ENTITIES = 40


def _normalize_entity(raw: str) -> str:
    value = raw.strip().strip("`\"'“”").strip().lower()
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def extract_entities(text: str) -> set[str]:
    """抽取查询/记忆里的"实体"，作为第三路检索信号。

    覆盖文件路径、裸文件名、点分模块、引号内容、snake_case 与 camelCase
    标识符。纯正则实现，不调用 LLM；噪声由上限和实体重叠加权控制。
    """
    if not text:
        return set()

    found: list[str] = []
    for m in _PATH_ENTITY_RE.finditer(text):
        found.append(m.group(0))
    for m in _FILE_ENTITY_RE.finditer(text):
        found.append(m.group(0))
    for m in _DOTTED_ENTITY_RE.finditer(text):
        found.append(m.group(0))
    for m in _QUOTED_ENTITY_RE.finditer(text):
        found.append(next(g for g in m.groups() if g))
    for m in _SNAKE_ENTITY_RE.finditer(text):
        found.append(m.group(0))
    for m in _CAMEL_ENTITY_RE.finditer(text):
        found.append(m.group(0))

    entities: set[str] = set()
    for raw in found:
        value = _normalize_entity(raw)
        if len(value) >= _MIN_ENTITY_LEN:
            entities.add(value)
        if len(entities) >= _MAX_ENTITIES:
            break
    return entities


# ---------------------------------------------------------------------------
# 文档表示
# ---------------------------------------------------------------------------


def build_document(text: str) -> dict[str, object]:
    """为一段记忆文本构建可序列化（JSON）的词法索引载荷。"""
    tokens = tokenize(text)
    return {
        "tokens": tokens,
        "doc_len": len(tokens),
        "entities": sorted(extract_entities(text)),
    }


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

BM25_K1 = 1.5
BM25_B = 0.75
# 出现在超过这个比例文档里的词项大幅降权：它们在编码记忆里几乎无区分度
# （如 `file`、`code`），但 IDF 在文档数很少时降得不够狠。
GENERIC_DF_RATIO = 0.5
GENERIC_TERM_PENALTY = 0.4


def _idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_scores(query_text: str, docs: list[dict[str, object]]) -> list[float]:
    """对每篇文档计算 BM25 分，并归一化到约 0~1（除以查询词权重和）。

    ``docs`` 每项需含 ``tokens``（list[str]）与 ``doc_len``（int）；
    缺少字段的（旧索引）按空文档处理，得分 0。
    """
    scores = [0.0] * len(docs)
    if not docs:
        return scores

    query_terms = Counter(tokenize(query_text))
    if not query_terms:
        return scores

    n_docs = len(docs)
    doc_tfs: list[Counter[str]] = []
    doc_lens: list[int] = []
    for doc in docs:
        tokens = doc.get("tokens")
        tf = Counter(tokens) if isinstance(tokens, list) else Counter()
        doc_tfs.append(tf)
        doc_lens.append(len(tokens) if isinstance(tokens, list) else 0)

    avgdl = sum(doc_lens) / n_docs if n_docs else 0.0
    if avgdl <= 0:
        return scores

    df = {term: sum(1 for tf in doc_tfs if term in tf) for term in query_terms}

    query_weight = 0.0
    term_weights: dict[str, float] = {}
    for term in query_terms:
        if df[term] == 0:
            continue
        weight = _idf(n_docs, df[term])
        if df[term] / n_docs > GENERIC_DF_RATIO:
            weight *= GENERIC_TERM_PENALTY
        term_weights[term] = weight
        query_weight += weight

    if query_weight <= 0:
        return scores

    for i, tf in enumerate(doc_tfs):
        if not tf:
            continue
        dl = doc_lens[i]
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl)
        total = 0.0
        for term, weight in term_weights.items():
            freq = tf.get(term, 0)
            if not freq:
                continue
            total += weight * freq * (BM25_K1 + 1) / (freq + norm)
        scores[i] = total / query_weight

    return scores


# ---------------------------------------------------------------------------
# 融合
# ---------------------------------------------------------------------------

# 融合权重刻意保守：BM25/实体只做加分，不推翻余弦阈值判定。
# 语义分仍是主信号；字面/实体信号负责把"意思接近但 token 精确命中"的
# 记忆顶上来，并救回余弦分偏低但精确匹配的记忆。
BM25_FUSION_WEIGHT = 0.4
ENTITY_FUSION_WEIGHT = 0.085
MAX_ENTITY_HITS = 3


def fuse_scores(
    cosine: float,
    bm25: float,
    entity_hits: int,
) -> float:
    """把三路信号合成为一个分数（阈值仍按原有语义作用于该分数）。"""
    entity_bonus = min(entity_hits, MAX_ENTITY_HITS) * ENTITY_FUSION_WEIGHT
    return cosine + BM25_FUSION_WEIGHT * bm25 + entity_bonus
