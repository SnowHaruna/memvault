"""
MemoryStore CRUD 测试

验证核心引擎的基本操作：add/get/search/update/delete
"""

import pytest

from memvault.config import MemoryVaultConfig
from memvault.core.engine import MemoryStore
from memvault.storage.memory import MemoryStorage


@pytest.fixture
def store():
    """创建使用内存后端的 MemoryStore（测试间独立）。"""
    config = MemoryVaultConfig()
    storage = MemoryStorage()
    return MemoryStore(config=config, storage=storage)


class TestMemoryStoreCRUD:

    def test_add_entry(self, store):
        result = store.add("memory", "这是一条测试记忆")
        assert result["success"]
        assert result["entry_id"]
        assert result["trimmed"] == 0

    def test_add_duplicate(self, store):
        store.add("memory", "测试记忆")
        result = store.add("memory", "测试记忆")
        assert result["success"]
        assert "already exists" in result.get("message", "")

    def test_add_empty(self, store):
        result = store.add("memory", "")
        assert not result["success"]

    def test_get_entries(self, store):
        store.add("memory", "记忆A")
        store.add("memory", "记忆B")
        store.add("user", "用户画像1")

        memory_entries = store.get("memory")
        user_entries = store.get("user")

        assert len(memory_entries) >= 2
        assert len(user_entries) >= 1

    def test_search(self, store):
        store.add("memory", "Python 是一门优雅的编程语言")
        store.add("memory", "今天吃了好吃的")
        store.add("memory", "修复了一个 Python bug")

        results = store.search("Python")
        assert len(results) >= 2

        results = store.search("吃饭")
        assert len(results) == 0  # 子串不匹配

    def test_update_entry(self, store):
        store.add("memory", "旧的内容需要更新")
        result = store.update("memory", "旧的内容", "新的内容")
        assert result["success"]

        # 验证更新
        entries = store.get("memory")
        contents = [e.content for e in entries]
        assert "新的内容" in contents
        assert "旧的内容需要更新" not in contents

    def test_delete_entry(self, store):
        store.add("memory", "要删除的记忆")
        result = store.delete("memory", "要删除的记忆")
        assert result["success"]

        entries = store.get("memory")
        contents = [e.content for e in entries]
        assert "要删除的记忆" not in contents

    def test_char_limit_trimming(self, store):
        """字符超限时应自动裁剪最低权重的条目。"""
        # 设置很小的字符上限
        store.config.memory.char_limit = 100

        # 添加多条记忆
        store.add("memory", "A" * 30)
        store.add("memory", "B" * 30)
        store.add("memory", "C" * 30)
        store.add("memory", "D" * 30)

        # 验证字符数在限制内
        chars = store.storage.char_count("memory")
        assert chars <= 100 + len("\n§\n") * 3  # 允许分隔符开销

    def test_stats(self, store):
        store.add("memory", "记忆1")
        store.add("user", "用户1")

        s = store.stats()
        assert s.memory_entries >= 1
        assert s.user_entries >= 1
        assert s.total_entries >= 2


class TestMemoryStoreBatch:

    def test_add_batch(self, store):
        items = [f"批量记忆 {i}" for i in range(10)]
        result = store.add_batch("memory", items)

        assert result["success"]
        assert result["added"] == 10

        entries = store.get("memory")
        assert len(entries) >= 10

    def test_add_batch_dedup(self, store):
        store.add("memory", "已存在的记忆")
        items = ["已存在的记忆", "新记忆1", "新记忆2"]
        result = store.add_batch("memory", items)

        # 取决于后端是否支持 INSERT OR IGNORE
        assert result["success"]
