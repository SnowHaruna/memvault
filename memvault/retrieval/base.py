"""
AbstractRetriever — 检索器接口

所有检索器（Dense/Sparse/ColBERT）必须实现此接口。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class AbstractRetriever(ABC):
    """检索器抽象接口。

    每个检索器负责一条独立的检索路径（如 Dense 语义、Sparse 关键词）。
    """

    @abstractmethod
    def index(self, entries: List[str]):
        """构建索引（首次或重建时调用）。

        Args:
            entries: 待索引的文本列表
        """
        ...

    @abstractmethod
    def retrieve(self, query: str,
                 top_k: int = 20) -> List[Tuple[float, str]]:
        """执行检索。

        Args:
            query: 查询文本
            top_k: 返回条数上限

        Returns:
            [(score, text), ...] 按分数降序
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """检索器是否可用（例如嵌入模型是否在线）。"""
        ...

    @property
    def name(self) -> str:
        """检索器名称（用于日志和统计）。"""
        return self.__class__.__name__
