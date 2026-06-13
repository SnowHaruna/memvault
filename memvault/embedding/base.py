"""
AbstractEmbedder — 嵌入服务接口

实现此接口即可接入任意嵌入引擎。
"""

from abc import ABC, abstractmethod
from typing import List


class AbstractEmbedder(ABC):
    """嵌入模型抽象接口。

    所有嵌入后端必须实现 check_health() 和 embed()。
    """

    @abstractmethod
    def check_health(self) -> bool:
        """健康检测。返回 True 表示可用。"""
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表，每个向量为 float 列表
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """单条查询嵌入（可优化，如使用不同的 prompt 前缀）。

        Args:
            query: 查询文本

        Returns:
            嵌入向量
        """
        ...
