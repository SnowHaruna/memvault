"""
MockEmbedder — 确定性 Mock 嵌入后端

用于 CI/CD 测试和离线开发。返回基于文本哈希的确定性向量。
无外部依赖，无需 Ollama 运行。
"""

import hashlib
import struct
from typing import Dict, List, Optional

from memvault.embedding.base import AbstractEmbedder


class MockEmbedder(AbstractEmbedder):
    """Mock 嵌入后端 — 返回确定性向量。

    每个文本生成一个 128d 的确定性向量（基于 MD5 哈希）。
    相同文本始终产生相同向量，确保测试可复现。

    Args:
        dim: 向量维度（默认 128，与 bge-m3 的 1024 区分以便识别 mock）
        seed: 哈希种子（默认 ""）
    """

    def __init__(self, dim: int = 128, seed: str = ""):
        self.dim = dim
        self.seed = seed

    def check_health(self) -> bool:
        """Mock 始终可用。"""
        return True

    def embed(self, texts: List[str]) -> List[List[float]]:
        """为每个文本生成确定性向量。"""
        return [self._deterministic_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """单条查询嵌入。"""
        return self._deterministic_vector(query)

    def _deterministic_vector(self, text: str) -> List[float]:
        """从文本哈希生成确定性向量（正态分布近似）。

        使用 MD5 哈希扩展产生 dim 个 [-1, 1] 之间的值。
        """
        key = f"{self.seed}:{text}".encode("utf-8")
        h = hashlib.md5(key).digest()

        # 扩展哈希到所需维度
        vec = []
        for i in range(self.dim):
            # 每个维度使用不同的哈希偏移
            offset_key = f"{self.seed}:{text}:{i}".encode("utf-8")
            val_hash = hashlib.md5(offset_key).digest()
            # 取 4 字节转为 float in [-1, 1]
            val = struct.unpack(">I", val_hash[:4])[0]
            normalized = (val / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(normalized)

        # L2 归一化
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec


class TopicMockEmbedder(AbstractEmbedder):
    """主题感知 Mock 嵌入器 — 模拟真实 bge-m3 的语义分组行为。

    与 MockEmbedder（纯随机向量）不同，此嵌入器根据文本中的主题关键词
    将文本映射到预定义主题质心附近，从而：
      - 同主题文本 → 高 cosine similarity（~0.9+）
      - 异主题文本 → 低 cosine similarity（~0.0-0.1）
      - 未知主题/对抗查询 → 随机方向（与所有质心 similarity ~0.1-0.2）

    这使 Dense 检索路径在 Mock 环境下也能产生有意义的结果，
    对抗查询会因低 similarity 被 relevance_threshold 过滤。

    Args:
        dim: 向量维度（默认 128）
        topics: 已知主题关键词列表（默认 5 个压力测试主题）
        noise_scale: 同主题内的微噪声幅度（默认 0.03）
    """

    def __init__(
        self,
        dim: int = 128,
        topics: Optional[List[str]] = None,
        noise_scale: float = 0.03,
    ):
        import math
        import random as _random

        self.dim = dim
        self.topics = topics or [
            "Python", "Docker", "记忆", "权重", "知识图谱",
        ]
        self.noise_scale = noise_scale
        self._centroids: Dict[str, List[float]] = self._build_centroids()

    def check_health(self) -> bool:
        return True

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_one(query)

    # ── 内部 ──

    def _build_centroids(self) -> Dict[str, List[float]]:
        """构建近似正交的主题质心向量。

        每个主题占据 dim/N 个不重叠的维度区间，确保余弦相似度接近 0。
        """
        import math

        n = len(self.topics)
        span = max(1, self.dim // n)
        centroids = {}

        for i, topic in enumerate(self.topics):
            vec = [0.0] * self.dim
            start = (i * span) % self.dim
            for j in range(span):
                vec[(start + j) % self.dim] = 1.0
            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec))
            centroids[topic] = [v / norm for v in vec]

        return centroids

    def _embed_one(self, text: str) -> List[float]:
        import math
        import random as _random

        # 检测文本命中了哪些主题
        matched = [t for t in self.topics if t in text]

        if not matched:
            # 无主题命中 → 文本哈希随机向量（远离所有质心）
            return self._random_vec(text)

        # 有主题命中 → 质心叠加 + 微噪声
        vec = [0.0] * self.dim
        for topic in matched:
            centroid = self._centroids[topic]
            for i in range(self.dim):
                vec[i] += centroid[i]

        # 微噪声：模拟同主题文本间的合理差异
        noise_seed = hash(text + "::noise") % (2 ** 31)
        rng = _random.Random(noise_seed)
        for i in range(self.dim):
            vec[i] += rng.uniform(-self.noise_scale, self.noise_scale)

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def _random_vec(self, text: str) -> List[float]:
        """为无主题文本生成随机方向向量。"""
        import math
        import random as _random

        rng = _random.Random(hash(text) % (2 ** 31))
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec
