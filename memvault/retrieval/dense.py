"""
DenseRetriever — bge-m3 稠密向量检索

通过 Ollama bge-m3 生成 1024d 嵌入，使用 LlamaIndex 做语义检索。
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from memvault.embedding.base import AbstractEmbedder
from memvault.retrieval.base import AbstractRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(AbstractRetriever):
    """bge-m3 稠密向量检索器。

    使用 LlamaIndex + Ollama bge-m3 构建向量索引，
    支持持久化（保存索引到磁盘）和增量重建。

    Args:
        embedder: 嵌入模型（AbstractEmbedder 实例）
        index_dir: 向量索引持久化目录
        embed_batch_size: 嵌入批大小
    """

    def __init__(
        self,
        embedder: AbstractEmbedder,
        index_dir: str = "./memvault_index",
        embed_batch_size: int = 16,
    ):
        self._embedder = embedder
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._embed_batch_size = embed_batch_size
        self._entries: List[str] = []
        self._index_built = False
        self._settings_configured = False

    # ── AbstractRetriever ──

    @property
    def is_available(self) -> bool:
        try:
            import llama_index.core
            return self._embedder.check_health()
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "Dense (bge-m3)"

    def index(self, entries: List[str]):
        """构建 LlamaIndex 向量索引。"""
        self._entries = [e.strip() for e in entries if e.strip()]
        if not self._entries:
            self._index_built = False
            return

        try:
            self._configure_llama_settings()
            self._do_rebuild_index()
            self._index_built = True
        except Exception as e:
            logger.warning("Dense index build failed: %s", e)
            self._index_built = False

    def retrieve(self, query: str,
                 top_k: int = 20) -> List[Tuple[float, str]]:
        """执行语义检索。"""
        if not self._index_built or not self._index_dir.exists():
            return []

        try:
            from llama_index.core import StorageContext, load_index_from_storage

            self._configure_llama_settings()
            ctx = StorageContext.from_defaults(persist_dir=str(self._index_dir))
            index = load_index_from_storage(ctx)
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)

            return [(node.score or 0.5, node.text.strip()) for node in nodes]
        except ImportError:
            logger.warning("LlamaIndex not available — dense disabled")
            return []
        except Exception as e:
            logger.warning("Dense retrieval failed: %s", e)
            return []

    # ── 内部 ──

    def _configure_llama_settings(self):
        """配置 LlamaIndex 全局设置（仅一次）。"""
        if self._settings_configured:
            return

        from llama_index.core import Settings
        from llama_index.embeddings.ollama import OllamaEmbedding

        if hasattr(self._embedder, 'model') and hasattr(self._embedder, 'base_url'):
            Settings.embed_model = OllamaEmbedding(
                model_name=self._embedder.model,
                base_url=self._embedder.base_url,
                embed_batch_size=self._embed_batch_size,
            )
        self._settings_configured = True

    def _do_rebuild_index(self):
        """重建 LlamaIndex 向量索引。"""
        from llama_index.core import VectorStoreIndex, Document, Settings
        from llama_index.embeddings.ollama import OllamaEmbedding

        self._configure_llama_settings()

        if hasattr(self._embedder, 'model') and hasattr(self._embedder, 'base_url'):
            Settings.embed_model = OllamaEmbedding(
                model_name=self._embedder.model,
                base_url=self._embedder.base_url,
                embed_batch_size=self._embed_batch_size,
            )

        docs = [Document(text=e) for e in self._entries]
        logger.info("Building dense index: %d docs (bge-m3)", len(docs))

        index = VectorStoreIndex.from_documents(docs, embed_model=Settings.embed_model)
        index.storage_context.persist(persist_dir=str(self._index_dir))

        # 记录最后构建时间
        mtime_file = self._index_dir / ".last_build_mtime"
        mtime_file.write_text(str(time.time()))

        logger.info("Dense index built — %d vectors", len(docs))

    def check_index_stale(self, entries_mtime: float) -> bool:
        """检查索引是否需要重建（基于文件修改时间）。"""
        mtime_file = self._index_dir / ".last_build_mtime"
        if not mtime_file.exists():
            return True
        try:
            last = float(mtime_file.read_text().strip())
            return entries_mtime > last
        except (ValueError, OSError):
            return True


class SimpleDenseRetriever:
    """轻量级稠密检索器 — 纯 Python，零外部依赖。

    直接使用 AbstractEmbedder 生成向量，在内存中做 cosine similarity 检索。
    作为 DenseRetriever 的降级方案，适用于：
      - 无 LlamaIndex 环境（如 CI / 压力测试）
      - Mock 嵌入器（TopicMockEmbedder 等）
      - 小规模快速验证

    Args:
        embedder: 嵌入模型实例（需实现 embed() 和 embed_query()）
    """

    def __init__(self, embedder: AbstractEmbedder):
        self._embedder = embedder
        self._texts: List[str] = []
        self._embeddings: List[List[float]] = []
        self._index_built = False

    @property
    def is_available(self) -> bool:
        return self._embedder.check_health()

    @property
    def name(self) -> str:
        return "SimpleDense (in-memory)"

    def index(self, entries: List[str]):
        """构建内存向量索引。"""
        import math

        self._texts = [e.strip() for e in entries if e.strip()]
        if not self._texts:
            self._index_built = False
            return

        self._embeddings = self._embedder.embed(self._texts)
        self._index_built = True

    def retrieve(self, query: str,
                 top_k: int = 20) -> List[Tuple[float, str]]:
        """cosine similarity 检索。"""
        import math

        if not self._index_built:
            return []

        query_vec = self._embedder.embed_query(query)

        scores = []
        for i, emb in enumerate(self._embeddings):
            # cosine similarity
            dot = sum(a * b for a, b in zip(query_vec, emb))
            norm_q = math.sqrt(sum(v * v for v in query_vec))
            norm_e = math.sqrt(sum(v * v for v in emb))
            sim = dot / (norm_q * norm_e) if norm_q > 0 and norm_e > 0 else 0.0
            scores.append((sim, self._texts[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    def check_index_stale(self, entries_mtime: float) -> bool:
        """SimpleDenseRetriever 的索引有效期由 HybridRetriever._ensure_index_fresh 管理。"""
        return False
