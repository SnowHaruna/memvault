"""
存储后端一致性测试

验证所有存储后端实现相同的 AbstractStorage 接口。
"""

import pytest

from memvault.storage.memory import MemoryStorage
from memvault.storage.file import FileStorage
from memvault.storage.sqlite import SQLiteStorage


# 测试所有可用后端
BACKENDS = [
    ("memory", lambda: MemoryStorage()),
]

# SQLite 后端（如果可用）
try:
    import sqlite3
    BACKENDS.append(("sqlite", lambda: SQLiteStorage(":memory:")))
except ImportError:
    pass

# File 后端（使用临时目录）
@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path / "memdata")


@pytest.mark.parametrize("name,backend_factory", BACKENDS)
class TestStorageBackend:

    @pytest.fixture
    def storage(self, name, backend_factory):
        backend = backend_factory()
        yield backend
        backend.close()

    def test_add_and_get(self, storage):
        eid = storage.add_entry("memory", "测试记忆")
        assert eid

        entry = storage.get_entry(eid)
        assert entry is not None
        assert entry.content == "测试记忆"
        assert entry.target == "memory"

    def test_add_duplicate(self, storage):
        eid1 = storage.add_entry("memory", "相同记忆")
        eid2 = storage.add_entry("memory", "相同记忆")
        assert eid1 == eid2  # 相同的哈希

    def test_get_entries(self, storage):
        storage.add_entry("memory", "记忆A")
        storage.add_entry("memory", "记忆B")
        storage.add_entry("user", "用户A")

        mem = storage.get_entries("memory")
        usr = storage.get_entries("user")

        assert len(mem) == 2
        assert len(usr) == 1

    def test_update_entry(self, storage):
        eid = storage.add_entry("memory", "旧内容")
        ok = storage.update_entry(eid, "新内容")
        assert ok

        entry = storage.get_entry(eid)
        assert entry.content == "新内容"

    def test_delete_entry(self, storage):
        eid = storage.add_entry("memory", "要删除的")
        ok = storage.delete_entry(eid)
        assert ok

        entry = storage.get_entry(eid)
        assert entry is None

    def test_update_meta(self, storage):
        eid = storage.add_entry("memory", "测试元数据")
        ok = storage.update_meta(eid, {"retrieval_count": 5})
        assert ok

        all_meta = storage.get_all_meta()
        assert eid in all_meta
        assert all_meta[eid]["retrieval_count"] == 5

    def test_count_and_chars(self, storage):
        storage.add_entry("memory", "ABCD")
        storage.add_entry("memory", "EFGH")

        assert storage.count_entries("memory") == 2
        # 字符数应 >= 8 (不含分隔符)
        assert storage.char_count("memory") >= 8

    def test_prune_and_vacuum(self, storage):
        storage.add_entry("memory", "不重要", importance=0.1)
        storage.add_entry("memory", "重要内容", importance=0.9)

        pruned = storage.prune_below_weight("memory", 0.5)
        assert pruned >= 1

        remaining = storage.get_entries("memory")
        assert all(e.importance >= 0.5 for e in remaining)
