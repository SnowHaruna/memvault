"""
AbstractStorage — 存储后端接口

实现此接口即可接入任意存储引擎（SQLite / PostgreSQL / Redis / 文件等）。

设计原则：
  - 增量更新：update_meta() 只修改变化的字段，不触发全量写入
  - 显式清理：prune_below_weight() + vacuum() 防止无限增长
  - 零假设：不依赖 JSON 文件格式或本地文件系统
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from memvault.types import MemoryEntry


class AbstractStorage(ABC):
    """存储后端抽象接口。

    所有持久化操作通过此接口完成。实现类负责：
      - 条目 CRUD
      - 元数据管理（权重、检索计数等）
      - 空间回收（清理过期/低权重条目）
    """

    @abstractmethod
    def add_entry(self, target: str, content: str,
                  importance: float = 0.5) -> str:
        """添加记忆条目。

        Args:
            target: 存储目标 ("memory" | "user")
            content: 记忆文本
            importance: 情绪重要性评分 (0.0-1.0)

        Returns:
            entry_id: 8 字符哈希 ID
        """
        ...

    @abstractmethod
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """获取单条记忆。

        Args:
            entry_id: 条目 ID

        Returns:
            MemoryEntry 或 None（未找到时）
        """
        ...

    @abstractmethod
    def get_entries(self, target: str,
                    limit: Optional[int] = None,
                    offset: int = 0,
                    min_weight: Optional[float] = None,
                    order_by: str = "weight") -> List[MemoryEntry]:
        """获取记忆条目列表。

        Args:
            target: 存储目标 ("memory" | "user")
            limit: 返回条数上限（None = 全部）
            offset: 偏移量（分页）
            min_weight: 仅返回权重 >= 此值的条目
            order_by: 排序字段 ("weight" | "created_at" | "importance")

        Returns:
            MemoryEntry 列表
        """
        ...

    @abstractmethod
    def update_entry(self, entry_id: str, content: str) -> bool:
        """更新条目内容（触发修正计数 +1）。

        Returns:
            True 表示成功
        """
        ...

    @abstractmethod
    def delete_entry(self, entry_id: str) -> bool:
        """删除单条记忆。

        Returns:
            True 表示删除成功
        """
        ...

    @abstractmethod
    def update_meta(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """增量更新单条记忆的元数据（不触发全量写入）。

        Args:
            entry_id: 条目 ID
            updates: 要更新的字段字典，如 {"retrieval_count": 5}

        Returns:
            True 表示成功
        """
        ...

    @abstractmethod
    def batch_update_meta(self, updates: Dict[str, Dict[str, Any]]) -> int:
        """批量增量更新元数据。

        Args:
            updates: {entry_id: {field: value, ...}, ...}

        Returns:
            成功更新的条目数
        """
        ...

    @abstractmethod
    def count_entries(self, target: Optional[str] = None) -> int:
        """统计条目数。

        Args:
            target: None = 全部, "memory" or "user"
        """
        ...

    @abstractmethod
    def char_count(self, target: str) -> int:
        """统计某目标的字符总数（含分隔符）。"""
        ...

    @abstractmethod
    def prune_below_weight(self, target: str, threshold: float) -> int:
        """清理低于权重阈值的条目。

        Args:
            target: 存储目标
            threshold: 权重阈值（低于此值被清理）

        Returns:
            清理的条目数
        """
        ...

    @abstractmethod
    def vacuum(self) -> int:
        """回收空间（清理已删除条目的残留数据）。

        Returns:
            回收的字节数（近似）
        """
        ...

    @abstractmethod
    def get_all_meta(self) -> Dict[str, Dict[str, Any]]:
        """获取所有元数据（用于权重计算）。

        Returns:
            {entry_id: {field: value, ...}, ...}
        """
        ...

    @abstractmethod
    def close(self):
        """关闭存储连接。"""
        ...
