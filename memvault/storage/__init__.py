"""
memvault 存储后端 — 可插拔架构

默认使用 SQLite，支持：
  - SQLiteStorage: 生产推荐（增量更新、FTS5 全文索引、自动清理）
  - FileStorage:  兼容旧版 JSON 文件存储
  - MemoryStorage: 测试/临时使用（无持久化）
"""

from memvault.storage.base import AbstractStorage
from memvault.storage.sqlite import SQLiteStorage
from memvault.storage.file import FileStorage
from memvault.storage.memory import MemoryStorage

__all__ = [
    "AbstractStorage",
    "SQLiteStorage",
    "FileStorage",
    "MemoryStorage",
]
