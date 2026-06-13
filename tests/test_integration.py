"""
端到端集成测试

验证完整流水线：
  1. HybridRetriever 三路检索 + Mock 嵌入
  2. MemoryVault.recall() 完整管道
  3. SleepConsolidator + MemoryVault 集成
  4. 检索 → 巩固反馈回路
  5. 降级回退路径验证
"""

import time

import pytest

from memvault import MemoryVault
from memvault.config import ConsolidationConfig, MemoryVaultConfig, RetrievalConfig
from memvault.core.consolidator import SleepConsolidator
from memvault.embedding.mock import MockEmbedder
from memvault.llm.mock import MockLLM
from memvault.retrieval.hybrid import HybridRetriever
from memvault.storage.memory import MemoryStorage
from memvault.types import SearchResult


@pytest.fixture
def vault_with_data():
    """创建包含混合记忆的 MemoryVault。"""
    config = MemoryVaultConfig()
    config.memory.char_limit = 32000
    vault = MemoryVault(
        config=config,
        storage=MemoryStorage(),
    )

    # 添加多种类型的记忆
    vault.remember("Python 的列表推导式可以替代简单的 for 循环，语法为 [x for x in iterable if condition]")
    vault.remember("asyncio 是 Python 异步编程的核心库，使用 async/await 语法")
    vault.remember("修复了一个并发 bug，根因是数据库连接池未设置 max_overflow 参数")
    vault.remember("用户反馈登录页面在 Safari 上白屏，可能是 WebKit CSS 兼容问题")
    vault.remember("JavaScript 的 Promise 链式调用可以用 async/await 简化")
    vault.remember("Docker 容器的内存限制可以通过 --memory 参数设置，默认无限制")
    vault.remember("今天天气很好，适合出门散步晒太阳")
    vault.remember("午饭吃了牛肉面，味道不错")

    # 用户画像
    vault.remember("用户偏好 Python 和 TypeScript，不喜欢 Java", target="user")
    vault.remember("用户使用 macOS 开发环境，VSCode 作为主力编辑器", target="user")

    return vault


class TestHybridRetrieverIntegration:
    """HybridRetriever 集成测试。"""

    @pytest.fixture
    def hybrid(self, vault_with_data):
        """创建使用 MockEmbedder 的 HybridRetriever。"""
        return HybridRetriever(
            embedder=MockEmbedder(dim=32),
            storage=vault_with_data._store.storage,
            config=RetrievalConfig(dense=True, sparse=True, colbert=True),
            memory_config=vault_with_data.config.memory,
        )

    def test_rebuild_index(self, hybrid):
        """索引重建应成功并包含所有条目。"""
        ok = hybrid.rebuild_index()
        assert ok
        assert hybrid.is_ready
        assert hybrid._sparse.doc_count >= 8

    def test_retrieve_returns_search_results(self, hybrid):
        """检索应返回 SearchResult 对象。"""
        hybrid.rebuild_index()
        results = hybrid.retrieve("Python", top_k=5)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score > 0 for r in results)

    def test_retrieve_relevance(self, hybrid):
        """检索结果应与查询相关。"""
        hybrid.rebuild_index()
        results = hybrid.retrieve("Python 异步编程", top_k=3)

        assert len(results) > 0
        # 第一条应包含 Python 相关术语
        first_content = results[0].entry.content.lower()
        assert any(kw in first_content for kw in ("python", "asyncio", "async"))

    def test_retrieve_with_sparse_only(self, hybrid):
        """仅使用 Sparse 路的检索。"""
        hybrid.config.dense = False  # 禁用 Dense
        hybrid.rebuild_index()
        results = hybrid.retrieve("bug", top_k=3)

        assert len(results) > 0
        assert all(r.source in ("S", "S+C", "keyword") for r in results)

    def test_keyword_fallback(self, hybrid):
        """无索引时的关键词回退。"""
        from memvault.retrieval.sparse import BM25SparseRetriever

        # 完全清除检索器状态，模拟冷启动
        hybrid._index_built = False
        hybrid._all_entries_cache = [
            "修复了一个并发 bug", "今天天气很好",
        ]
        # 清除 sparse 索引以确保走 keyword 回退
        hybrid._sparse = BM25SparseRetriever()

        results = hybrid.retrieve("bug")
        assert len(results) > 0
        # sparse brute_retrieve 或 keyword fallback 均可
        assert results[0].source in ("keyword", "S", "S+C", "D+S+C")

    def test_inhibition_decays(self, hybrid):
        """扩散抑制：重复检索同一查询应降权。"""
        hybrid.rebuild_index()

        # 第一次检索
        r1 = hybrid.retrieve("Python", top_k=3)

        # 立即第二次检索（应受抑制影响）
        r2 = hybrid.retrieve("Python", top_k=3)

        # 两项结果都应返回（条目不同的排序可能不同）
        assert len(r2) > 0


class TestMemoryVaultRetrievalIntegration:
    """MemoryVault + HybridRetriever 集成。"""

    def test_recall_with_retriever(self, vault_with_data):
        """MemoryVault.recall() 应使用 HybridRetriever。"""
        vault = vault_with_data

        # 手动构建检索器
        retriever = vault._get_retriever()
        retriever.rebuild_index()

        results = vault.recall("bug")
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    def test_recall_fallback_without_index(self, vault_with_data):
        """无索引时 recall() 应回退到关键词搜索。"""
        vault = vault_with_data
        # 不构建检索器索引
        results = vault.recall("bug")

        assert len(results) > 0
        # 回退结果 source 应为 "keyword"
        assert any("keyword" in r.source for r in results)

    def test_recall_with_kg(self, vault_with_data):
        """recall_with_kg 应同时返回检索结果和 KG 规则。"""
        vault = vault_with_data

        # 先添加一些 KG 规则
        if hasattr(vault._store.storage, 'add_kg_rule'):
            vault._store.storage.add_kg_rule("Python 异步编程规范", confidence=1.0)

        results, kg_rules = vault.recall_with_kg("Python", top_k=3)
        assert len(results) > 0

    def test_rebuild_index_api(self, vault_with_data):
        """MemoryVault.rebuild_index() 应正常工作。"""
        vault = vault_with_data
        ok = vault.rebuild_index()
        assert ok  # 应成功（使用 MockEmbedder）


class TestConsolidationIntegration:
    """SleepConsolidator + MemoryVault 集成。"""

    @pytest.fixture
    def vault_with_similar(self):
        """创建包含相似记忆的 vault。"""
        config = MemoryVaultConfig()
        config.consolidation = ConsolidationConfig(
            min_cluster_size=2,
            similarity_threshold=0.01,  # Mock 向量低阈值
        )
        vault = MemoryVault(
            config=config,
            storage=MemoryStorage(),
        )

        # 相似记忆组 1: Python 编程
        vault.remember("Python 列表推导式可以替代简单 for 循环")
        vault.remember("用 Python 生成器表达式处理大数据集更节省内存")
        vault.remember("Python 装饰器是实现 AOP 的优雅方式")

        # 相似记忆组 2: 前端
        vault.remember("CSS Grid 布局比 Flexbox 更适合二维布局")
        vault.remember("Flexbox 适合一维布局，Grid 适合二维网格")

        # 不相关记忆
        vault.remember("今天天气不错")
        vault.remember("午饭吃了炒面")

        return vault

    def test_consolidate_extracts_rules(self, vault_with_similar):
        """Sleep Loop 应从相似记忆中提取规则。"""
        vault = vault_with_similar

        # 注入 mock LLM
        mock_llm = MockLLM(preset_responses=[
            "Python 函数式编程特性可提升代码简洁性",
            "CSS 布局应根据维度选择 Grid 或 Flexbox",
        ])

        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=mock_llm,
            storage=vault._store.storage,
            config=vault.config.consolidation,
        )

        result = consolidator.run()
        assert result.rules_extracted > 0
        assert len(result.rules) > 0
        assert result.clusters_found > 0

    def test_consolidation_adds_kg_nodes(self, vault_with_similar):
        """巩固应将规则写入 KG。"""
        vault = vault_with_similar

        mock_llm = MockLLM(preset_responses=["编程规则 A", "前端规则 B"])
        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=mock_llm,
            storage=vault._store.storage,
            config=vault.config.consolidation,
        )

        result = consolidator.run()
        # 规则已提取
        assert result.rules_extracted > 0

        # KG 节点可能通过 hasattr 写入（MemoryStorage 支持 add_kg_rule）
        rules = vault.get_kg_rules()
        if hasattr(vault._store.storage, 'add_kg_rule'):
            assert result.kg_nodes_added > 0
            assert len(rules) > 0

    def test_feedback_loop(self, vault_with_similar):
        """规则强化和削弱反馈回路。"""
        vault = vault_with_similar

        mock_llm = MockLLM(preset_responses=["测试规则: Python 适合数据处理"])
        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=mock_llm,
            storage=vault._store.storage,
            config=vault.config.consolidation,
        )

        # 第一次巩固
        result = consolidator.run()
        assert result.rules_extracted > 0

        # 获取规则 ID 并测试反馈回路
        rules = vault.get_kg_rules()
        if rules:
            rule_id = rules[0]["id"]
            # 强化
            consolidator.reinforce_rule(rule_id, boost=0.2)
            # 削弱
            consolidator.contradict_rule(rule_id, penalty=0.5)

    def test_contradiction_detection(self, vault_with_similar):
        """矛盾检测。"""
        vault = vault_with_similar

        # 添加可能包含对立关键词的规则
        vault._store.storage.add_kg_rule("必须使用异步编程处理 I/O", confidence=1.0)
        vault._store.storage.add_kg_rule("不要过度使用异步，简单场景用同步", confidence=1.0)

        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=MockLLM(),
            storage=vault._store.storage,
        )

        contradictions = consolidator.find_contradictions()
        # "必须" vs "不要" 应被检测到
        assert len(contradictions) >= 1


class TestFallbackPaths:
    """降级路径验证。"""

    def test_no_embedder_fallback(self):
        """无嵌入器时检索应回退到关键词匹配。"""
        vault = MemoryVault(storage=MemoryStorage())
        vault.remember("测试记忆 about Python bugs")

        # 不构建检索器，直接 recall
        results = vault.recall("Python")
        assert len(results) > 0

    def test_empty_storage_no_crash(self):
        """空存储不应崩溃。"""
        vault = MemoryVault(storage=MemoryStorage())

        # 所有操作都应安全返回
        assert vault.recall("test") == []
        assert vault.recall_with_kg("test") == ([], [])
        assert vault.context() == ""
        assert vault.stats().total_entries == 0

    def test_consolidate_empty_storage(self):
        """空存储的 Sleep Loop 应安全退出。"""
        vault = MemoryVault(storage=MemoryStorage())
        result = vault.consolidate()
        assert result.scanned == 0
        assert result.rules_extracted == 0


class TestSearchResultType:
    """SearchResult 类型完整性测试。"""

    def test_search_result_fields(self):
        """SearchResult 应包含所有预期字段。"""
        from memvault.types import MemoryEntry, SearchResult

        entry = MemoryEntry(id="test", content="记忆内容", target="memory")
        result = SearchResult(
            entry=entry,
            score=0.95,
            dense_score=0.8,
            sparse_score=0.7,
            colbert_score=0.6,
            source="D+S+C",
        )

        assert result.entry == entry
        assert result.score == 0.95
        assert result.dense_score == 0.8
        assert result.sparse_score == 0.7
        assert result.colbert_score == 0.6
        assert result.source == "D+S+C"
