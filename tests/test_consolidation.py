"""
Sleep Loop 巩固测试

使用 Mock LLM + Mock Embedder 验证完整的 Sleep Loop 通路。
这是测试期从未覆盖的关键测试路径。
"""

import pytest

from memvault.config import ConsolidationConfig
from memvault.core.consolidator import SleepConsolidator
from memvault.embedding.mock import MockEmbedder
from memvault.llm.mock import MockLLM
from memvault.storage.memory import MemoryStorage


@pytest.fixture
def storage_with_entries():
    """创建包含测试记忆的存储。"""
    storage = MemoryStorage()

    # 添加相似记忆（应被聚类）
    entries = [
        ("memory", "Python 的列表推导式可以替代简单的 for 循环", 0.5),
        ("memory", "用生成器表达式处理大数据集比列表更节省内存", 0.5),
        ("memory", "asyncio 是 Python 异步编程的核心库", 0.5),
        ("memory", "JavaScript 的 Promise 和 async/await 类似 Python 的协程", 0.5),
        # 孤立记忆（不应被聚类）
        ("memory", "今天天气很好", 0.3),
        ("memory", "午饭吃了牛肉面", 0.3),
    ]

    for target, content, importance in entries:
        storage.add_entry(target, content, importance)

    return storage


class TestSleepConsolidator:

    def test_run_with_mocks(self, storage_with_entries):
        """使用 Mock LLM + Mock Embedder 运行完整的 Sleep Loop。"""
        config = ConsolidationConfig(
            min_cluster_size=2,
            similarity_threshold=0.01,  # Mock 向量无语义，需要极低阈值
        )
        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=MockLLM(preset_responses=["Python 编程规则1", "异步编程规则2"]),
            storage=storage_with_entries,
            config=config,
        )

        result = consolidator.run()
        assert result.scanned > 0
        assert result.rules_extracted > 0
        assert len(result.rules) > 0

    def test_dry_run(self, storage_with_entries):
        """试运行不应修改存储。"""
        config = ConsolidationConfig(
            similarity_threshold=0.01,  # Mock 向量需要极低阈值
        )
        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=MockLLM(preset_responses=["测试规则"]),
            storage=storage_with_entries,
            config=config,
        )

        before_count = storage_with_entries.count_entries()
        result = consolidator.run_dry()
        after_count = storage_with_entries.count_entries()

        assert before_count == after_count, "dry_run 不应修改存储"
        assert result.rules_extracted > 0

    def test_empty_storage(self):
        """空存储应安全退出。"""
        consolidator = SleepConsolidator(
            embedder=MockEmbedder(),
            llm=MockLLM(),
            storage=MemoryStorage(),
        )

        result = consolidator.run()
        assert result.scanned == 0
        assert result.clusters_found == 0
        assert result.rules_extracted == 0

    def test_insufficient_entries(self):
        """条目不足 min_cluster_size 时应跳过聚类。"""
        storage = MemoryStorage()
        storage.add_entry("memory", "唯一的一条记忆", 0.5)

        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=MockLLM(),
            storage=storage,
        )

        result = consolidator.run()
        assert result.clusters_found == 0

    def test_consolidation_result_fields(self, storage_with_entries):
        """ConsolidationResult 应包含所有字段。"""
        consolidator = SleepConsolidator(
            embedder=MockEmbedder(dim=32),
            llm=MockLLM(preset_responses=["规则"]),
            storage=storage_with_entries,
        )

        result = consolidator.run()
        assert hasattr(result, 'scanned')
        assert hasattr(result, 'clusters_found')
        assert hasattr(result, 'rules_extracted')
        assert hasattr(result, 'rules')
        assert hasattr(result, 'confidence_scores')
        assert hasattr(result, 'elapsed_seconds')
        assert hasattr(result, 'log')
