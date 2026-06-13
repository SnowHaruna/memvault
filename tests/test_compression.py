"""
压缩功能测试

验证首因+近因压缩 + LLM 回退。
"""

import pytest

from memvault.core.compressor import MemoryCompressor, create_llm_summarizer
from memvault.llm.mock import MockLLM


class TestMemoryCompressor:

    @pytest.fixture
    def entries(self):
        return [f"记忆条目 {i}: 这是第 {i} 条测试记忆的内容。" for i in range(12)]

    def test_below_threshold(self, entries):
        """条目数未达阈值时不压缩。"""
        compressor = MemoryCompressor(threshold=15)
        result = compressor.compress(entries[:5])
        assert result["success"]
        assert result["after"] == result["before"]
        assert "无需压缩" in result.get("message", "")

    def test_primacy_recency_preservation(self, entries):
        """首因和近因条目应保留原文。"""
        compressor = MemoryCompressor(
            primacy_count=2,
            recency_count=2,
            group_size=3,
            threshold=5,
            backup_enabled=False,
        )

        result = compressor.compress(entries)
        assert result["success"]
        assert "entries" in result

        new_entries = result["entries"]
        # 首因保留
        assert entries[0] in new_entries
        assert entries[1] in new_entries
        # 近因保留
        assert entries[-1] in new_entries
        assert entries[-2] in new_entries

    def test_backup_and_undo(self, entries):
        """压缩备份应可撤销。"""
        compressor = MemoryCompressor(
            primacy_count=1,
            recency_count=1,
            group_size=3,
            threshold=5,
            backup_enabled=True,
        )

        # 第一次压缩
        result = compressor.compress(entries)
        assert result["success"]
        assert "entries" in result
        compressed = result["entries"]
        assert len(compressed) < len(entries)

        # 撤销
        undo_result = compressor.undo(compressed)
        assert undo_result["success"]
        assert undo_result["restored"] == len(entries)

    def test_no_backup_available(self):
        """无备份时撤销应失败。"""
        compressor = MemoryCompressor()
        result = compressor.undo([])
        assert not result["success"]
        assert "没有可用" in result.get("error", "")

    def test_with_mock_llm(self, entries):
        """使用 Mock LLM 的压缩。"""
        llm = MockLLM(preset_responses=["摘要1", "摘要2"])
        summarizer = create_llm_summarizer(llm)

        compressor = MemoryCompressor(
            summarizer=summarizer,
            primacy_count=2,
            recency_count=2,
            group_size=3,
            threshold=5,
            backup_enabled=False,
        )

        result = compressor.compress(entries)
        assert result["success"]
        assert result["llm_used"]

    def test_fallback_merge(self):
        """回退合并应从每条取首句。"""
        batch = [
            "今天天气很好。阳光明媚。",
            "Python是一门优雅的语言。适合初学者。",
            "机器学习改变了世界！深度学习是核心。",
        ]
        result = MemoryCompressor._fallback_merge(batch)
        assert result is not None
        assert "今天天气很好" in result
        assert "Python是一门优雅的语言" in result
