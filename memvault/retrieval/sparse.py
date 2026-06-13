"""
BM25SparseRetriever — BM25 稀疏检索器（字符级 bigram）

模拟 bge-m3 的 learned sparse 输出，用统计方法做精确关键词匹配。
纯 Python 实现，零外部依赖。

中文友好策略：
  - 中文字符按单字切 bigram
  - 英文/数字保持单词边界
"""

import logging
import math
from collections import Counter
from typing import Dict, List, Tuple

from memvault.retrieval.base import AbstractRetriever

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# Bigram 分词
# ═══════════════════════════════════════════════════

def char_bigrams(text: str) -> List[str]:
    """中文友好的字符 bigram 分词。

    中文按单字切 bigram，英文/数字保持单词边界。

    Args:
        text: 输入文本

    Returns:
        bigram token 列表
    """
    tokens = []
    buf = ""
    for ch in text:
        # Unicode 范围：CJK 统一表意文字 + 扩展 A
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            if buf:
                tokens.append(buf.lower())
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():
            buf += ch
        else:
            if buf:
                tokens.append(buf.lower())
                buf = ""
    if buf:
        tokens.append(buf.lower())

    # 生成 bigrams
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append(tokens[i] + tokens[i + 1])
    # 也保留 unigrams
    bigrams.extend(tokens)
    return bigrams


# ═══════════════════════════════════════════════════
# BM25 检索器
# ═══════════════════════════════════════════════════

class BM25SparseRetriever(AbstractRetriever):
    """BM25 稀疏检索器。

    模拟 bge-m3 learned sparse 输出的统计方法。
    使用字符 bigram 索引做精确关键词匹配。

    Args:
        k1: BM25 词频饱和度参数（默认 1.5）
        b: BM25 长度归一化参数（默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._entries: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._built = False

    # ── AbstractRetriever ──

    @property
    def is_available(self) -> bool:
        """BM25 始终可用（纯 Python，零外部依赖）。"""
        return True

    @property
    def name(self) -> str:
        return "Sparse (BM25)"

    def index(self, entries: List[str]):
        """构建 BM25 索引。

        Args:
            entries: 待索引的文本列表
        """
        self._entries = [e.strip() for e in entries if e.strip()]
        self._doc_tokens = [char_bigrams(e) for e in self._entries]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_len))

        # 计算 document frequency
        self._df = {}
        for tokens in self._doc_tokens:
            for token in set(tokens):
                self._df[token] = self._df.get(token, 0) + 1

        # 预计算 IDF
        N = len(self._entries)
        self._idf = {}
        for token, df in self._df.items():
            self._idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        self._built = True
        logger.info("BM25 index built: %d docs, avg_len=%.1f, vocab=%d",
                     N, self._avgdl, len(self._df))

    def retrieve(self, query: str,
                 top_k: int = 20) -> List[Tuple[float, str]]:
        """BM25 检索。

        Args:
            query: 查询文本
            top_k: 返回条数上限

        Returns:
            [(bm25_score, text), ...] 按分数降序
        """
        if not self._entries:
            return []

        query_tokens = char_bigrams(query)
        if not query_tokens:
            return []

        scores = []
        for i, doc_tokens in enumerate(self._doc_tokens):
            doc_len = self._doc_len[i]
            score = 0.0
            tf = Counter(doc_tokens)

            for qt in set(query_tokens):
                if qt not in self._idf:
                    continue
                f = tf.get(qt, 0)
                if f == 0:
                    continue

                idf = self._idf[qt]
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (
                    1.0 - self.b + self.b * doc_len / max(1, self._avgdl)
                )
                score += idf * numerator / denominator

            if score > 0:
                scores.append((score, self._entries[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    # ── 工具 ──

    @property
    def doc_count(self) -> int:
        """已索引文档数。"""
        return len(self._entries)

    def brute_retrieve(self, query: str,
                       candidates: List[str],
                       top_k: int = 20) -> List[Tuple[float, str]]:
        """暴力检索：对所有候选做 bigram 重叠评分。

        当预处理索引不可用时使用（例如直接从 MemoryStore 获取条目）。

        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回条数

        Returns:
            [(jaccard_score, text), ...]
        """
        query_bigrams = set(char_bigrams(query))
        if not query_bigrams:
            return []

        results = []
        for text in candidates:
            doc_bigrams = set(char_bigrams(text))
            if not doc_bigrams:
                continue

            overlap = len(query_bigrams & doc_bigrams)
            if overlap > 0:
                # Jaccard 相似度
                score = overlap / len(query_bigrams | doc_bigrams)
                results.append((score, text))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
