"""
检索器精度测试

验证三路检索 (Dense + Sparse + ColBERT) 的基本正确性。
使用确定性输入验证检索输出。
"""

import pytest

from memvault.retrieval.sparse import BM25SparseRetriever, char_bigrams
from memvault.retrieval.colbert import ColbertMaxSimReranker
from memvault.retrieval.fusion import rrf_fuse


class TestCharBigrams:

    def test_chinese_bigrams(self):
        """中文文本应正确生成 bigram。"""
        tokens = char_bigrams("你好世界")
        assert "你好" in tokens or "好世" in tokens or "世界" in tokens

    def test_mixed_cn_en(self):
        """中英混合文本。"""
        tokens = char_bigrams("Python很优雅")
        assert "python" in tokens or "Python" in tokens  # unigrams
        # 应该有中文字符的 bigram
        assert len(tokens) > 0

    def test_empty_text(self):
        """空文本应返回空列表。"""
        assert char_bigrams("") == []


class TestBM25SparseRetriever:

    @pytest.fixture
    def retriever(self):
        r = BM25SparseRetriever()
        r.index([
            "Python 是一门优雅的编程语言",
            "Python 的异步编程模型 asyncio 很强大",
            "JavaScript 是前端开发的主要语言",
            "今天天气很好适合出门散步",
        ])
        return r

    def test_index(self, retriever):
        assert retriever.doc_count == 4

    def test_basic_retrieval(self, retriever):
        results = retriever.retrieve("Python", top_k=2)
        assert len(results) > 0
        # 第一条应包含 Python
        assert "Python" in results[0][1]

    def test_no_match(self, retriever):
        results = retriever.retrieve("量子力学", top_k=2)
        assert len(results) == 0

    def test_empty_index(self):
        r = BM25SparseRetriever()
        results = r.retrieve("test")
        assert results == []


class TestColbertMaxSimReranker:

    @pytest.fixture
    def reranker(self):
        return ColbertMaxSimReranker()

    def test_rerank_exact_match(self, reranker):
        query = "Python bug"
        candidates = [
            "修复了一个 Python 并发 bug",
            "今天天气不错",
            "JavaScript 异步编程",
        ]

        results = reranker.rerank(query, candidates)
        assert len(results) > 0
        # 第一条应是 Python bug 相关的
        assert "Python" in results[0][1]

    def test_rerank_empty(self, reranker):
        results = reranker.rerank("test", [])
        assert results == []

    def test_token_similarity(self, reranker):
        """精确匹配 = 1.0，无重叠 = 0.0。"""
        assert reranker._token_similarity("python", "python") == 1.0
        assert reranker._token_similarity("abc", "xyz") == 0.0


class TestRRFFusion:

    def test_basic_fusion(self):
        dense = [(0.9, "A"), (0.8, "B"), (0.7, "C")]
        sparse = [(0.6, "B"), (0.5, "C"), (0.4, "D")]

        fused = rrf_fuse(dense, sparse, k=60)
        assert len(fused) == 4  # A, B, C, D
        # B 在两路都出现，应该排第一
        assert fused[0][1] == "B"

    def test_single_list(self):
        results = rrf_fuse([(0.9, "A"), (0.8, "B")])
        assert len(results) == 2
        assert results[0][1] == "A"

    def test_empty_lists(self):
        results = rrf_fuse([], [])
        assert results == []
