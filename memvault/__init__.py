"""
memvault — 一个以认知科学为理论底座的 AI 原生记忆系统

不是又一个 RAG 框架。它会遗忘。会在睡眠中巩固知识。会自己长出新规则。

三行代码跑起来:
    from memvault import MemoryVault

    vault = MemoryVault()
    vault.remember("今天修复了一个并发 bug")
    results = vault.recall("bug")

定位: 认知记忆中间件。不替代向量数据库，不替代 LangChain。
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memvault.config import MemoryVaultConfig, load_config
from memvault.core.engine import MemoryStore
from memvault.core.compressor import MemoryCompressor, create_llm_summarizer
from memvault.core.consolidator import SleepConsolidator
from memvault.core.decay import compute_weight, compute_importance
from memvault.core.formatter import format_system_prompt_block
from memvault.embedding.base import AbstractEmbedder
from memvault.embedding.ollama import OllamaEmbedder
from memvault.embedding.mock import MockEmbedder, TopicMockEmbedder
from memvault.llm.client import LLMClient
from memvault.llm.mock import MockLLM
from memvault.retrieval.hybrid import HybridRetriever
from memvault.storage.base import AbstractStorage
from memvault.storage.sqlite import SQLiteStorage
from memvault.storage.file import FileStorage
from memvault.storage.memory import MemoryStorage
from memvault.types import (
    ConsolidationResult,
    MemoryEntry,
    MemoryStats,
    SearchResult,
    UserProfile,
)

logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__all__ = [
    # ── 顶层 API ──
    "MemoryVault",
    "load_config",
    "MemoryVaultConfig",
    # ── 核心类型 ──
    "MemoryEntry",
    "UserProfile",
    "ConsolidationResult",
    "SearchResult",
    "MemoryStats",
    # ── 核心引擎（高级用户）──
    "MemoryStore",
    "MemoryCompressor",
    "SleepConsolidator",
    "compute_weight",
    "compute_importance",
    # ── 存储后端 ──
    "AbstractStorage",
    "SQLiteStorage",
    "FileStorage",
    "MemoryStorage",
    # ── 嵌入后端 ──
    "AbstractEmbedder",
    "OllamaEmbedder",
    "MockEmbedder",
    "TopicMockEmbedder",
    # ── LLM 客户端 ──
    "LLMClient",
    "MockLLM",
    # ── 工具 ──
    "format_system_prompt_block",
    "create_llm_summarizer",
]


# ═══════════════════════════════════════════════════
# MemoryVault — 顶层 API
# ═══════════════════════════════════════════════════

class MemoryVault:
    """memvault 顶层 API — 开发者主入口。

    封装 MemoryStore + Retriever + Consolidator + Compressor，
    提供一行式接口。

    用法:
        # 最简用法
        vault = MemoryVault()

        # 自定义配置
        vault = MemoryVault(
            config=MemoryVaultConfig(
                memory=MemoryConfig(half_life_days=14.0, char_limit=16000),
            ),
            storage=SQLiteStorage("my_vault.db"),
        )

        # 完全自定义
        vault = MemoryVault(
            storage=SQLiteStorage("memories.db"),
            embedder=OllamaEmbedder(model="bge-m3"),
            llm=LLMClient(provider="deepseek", model="deepseek-v4-flash"),
        )
    """

    def __init__(
        self,
        config: Optional[MemoryVaultConfig] = None,
        storage: Optional[AbstractStorage] = None,
        embedder: Optional[AbstractEmbedder] = None,
        llm: Optional[LLMClient] = None,
    ):
        """
        Args:
            config: 全局配置（默认从环境变量 + 默认值）
            storage: 存储后端（默认 SQLite）
            embedder: 嵌入模型（默认 Ollama bge-m3）
            llm: LLM 客户端（默认需手动配置 API key）
        """
        self.config = config or MemoryVaultConfig()

        # 初始化各组件
        self._store = MemoryStore(config=self.config, storage=storage)
        self._embedder = embedder or self._create_default_embedder()
        self._llm = llm or self._create_default_llm()

        # 延迟初始化（需要时再创建）
        self._retriever: Optional[HybridRetriever] = None
        self._consolidator: Optional[SleepConsolidator] = None
        self._compressor: Optional[MemoryCompressor] = None

    # ── 核心操作 ──

    def remember(self, content: str, target: str = "memory",
                 importance: Optional[float] = None) -> Dict[str, Any]:
        """添加一条记忆。

        Args:
            content: 记忆文本
            target: "memory" (情景记忆) | "user" (用户画像)
            importance: 手动指定重要性 (0.0-1.0)，None = 自动评分

        Returns:
            {"success": bool, "entry_id": str, ...}

        Example:
            vault.remember("今天修复了一个并发 bug，根因是连接池未设置 max_overflow")
            vault.remember("用户偏好 Python 和 Rust", target="user")
        """
        return self._store.add(target, content, importance=importance)

    def remember_batch(self, items: List[str],
                       target: str = "memory") -> Dict[str, Any]:
        """批量添加记忆（比逐条调用 remember() 快 10-50x）。

        Args:
            items: 记忆文本列表
            target: "memory" | "user"

        Returns:
            {"success": bool, "added": int, ...}
        """
        return self._store.add_batch(target, items)

    def recall(self, query: str, top_k: int = 10,
               target: str = "memory") -> List[SearchResult]:
        """三路混合检索 (Dense + Sparse + ColBERT)。

        自动选择可用的检索路径：
          - Dense: bge-m3 语义向量（需 Ollama 在线）
          - Sparse: BM25 bigram 关键词（纯 Python，始终可用）
          - ColBERT: Token MaxSim 重排序（纯 Python）
          - 融合: RRF(k=60) + Ebbinghaus 权重重排

        全部路径不可用时自动回退到关键词子串匹配。

        Args:
            query: 搜索查询
            top_k: 返回条数
            target: 存储目标

        Returns:
            SearchResult 列表（按 score 降序）

        Example:
            results = vault.recall("bug")
            for r in results:
                print(f"[{r.score:.3f}] [{r.source}] {r.entry.content}")
        """
        retriever = self._get_retriever()
        if retriever and retriever.is_ready:
            return retriever.retrieve(query, top_k=top_k, target=target)

        # 回退：关键词搜索
        entries = self._store.search(query, top_k=top_k)
        return [
            SearchResult(entry=e, score=e.weight, source="keyword")
            for e in entries
        ]

    def recall_with_kg(self, query: str, top_k: int = 10
                       ) -> Tuple[List[SearchResult], List[str]]:
        """检索记忆 + 相关 KG 规则。

        Args:
            query: 搜索查询
            top_k: 返回条数

        Returns:
            (search_results, kg_rules)
        """
        retriever = self._get_retriever()
        if retriever and retriever.is_ready:
            return retriever.retrieve_with_kg(query, top_k=top_k)

        return self.recall(query, top_k=top_k), []

    def rebuild_index(self) -> bool:
        """强制重建检索索引（切换嵌入模型后调用）。

        Returns:
            True 表示重建成功
        """
        retriever = self._get_retriever()
        if retriever:
            return retriever.rebuild_index()
        return False

    def context(self, target: str = "memory",
                max_entries: int = 20) -> str:
        """获取当前记忆上下文（可直接注入 LLM system prompt）。

        Args:
            target: "memory" | "user"
            max_entries: 最大条目数

        Returns:
            格式化的记忆文本块
        """
        return self._store.context(target, max_entries=max_entries)

    def forget(self, target: str = "memory",
               min_weight: float = 0.3) -> int:
        """主动遗忘低权重记忆。

        Args:
            target: "memory" | "user"
            min_weight: 权重阈值（低于此值被遗忘）

        Returns:
            清理的条目数
        """
        return self._store.forget(target, min_weight=min_weight)

    def update(self, old_text: str, new_content: str,
               target: str = "memory") -> Dict[str, Any]:
        """更新一条记忆。修正过的记忆权重会提升。

        Args:
            old_text: 要替换的原文（子串匹配）
            new_content: 新内容
            target: "memory" | "user"

        Returns:
            {"success": bool, ...}
        """
        return self._store.update(target, old_text, new_content)

    def remove(self, old_text: str,
               target: str = "memory") -> Dict[str, Any]:
        """删除一条记忆。

        Args:
            old_text: 要删除的文本（子串匹配）
            target: "memory" | "user"

        Returns:
            {"success": bool, ...}
        """
        return self._store.delete(target, old_text)

    # ── 压缩 ──

    def compress(self) -> Dict[str, Any]:
        """压缩记忆：首因+近因保留原文，中间送 LLM 摘要。

        自动处理 LLM 不可用的情况（回退为原文保留）。
        压缩前自动备份，可通过 undo_compress() 恢复。

        Returns:
            {"success": bool, "before": int, "after": int, ...}
        """
        compressor = self._get_compressor()
        entries = [e.content for e in self._store.storage.get_entries("memory")]
        result = compressor.compress(entries)

        if result.get("success") and "entries" in result:
            # 写回压缩后的条目（TODO: 更高效的增量更新）
            new_entries = result["entries"]
            # 重建存储
            from memvault.storage.memory import MemoryStorage
            if not isinstance(self._store.storage, MemoryStorage):
                # 简单策略：清空后重新添加
                current = [e.content for e in self._store.storage.get_entries("memory")]
                for eid in [e.id for e in self._store.storage.get_entries("memory")]:
                    if eid not in [_h(e) for e in new_entries]:
                        self._store.storage.delete_entry(eid)
                for entry_text in new_entries:
                    if entry_text not in current:
                        self._store.storage.add_entry("memory", entry_text)

        return result

    def undo_compress(self) -> Dict[str, Any]:
        """撤销最近一次压缩。

        Returns:
            {"success": bool, "restored": int, ...}
        """
        compressor = self._get_compressor()
        entries = [e.content for e in self._store.storage.get_entries("memory")]
        return compressor.undo(entries)

    # ── 巩固 ──

    def consolidate(self, dry_run: bool = False) -> ConsolidationResult:
        """手动触发睡眠巩固。

        聚类相似记忆 → LLM 提取规则 → 写入知识图谱。

        Args:
            dry_run: True 时只报告不写入

        Returns:
            ConsolidationResult
        """
        consolidator = self._get_consolidator()
        return consolidator.run(dry_run=dry_run)

    def start_auto_consolidate(self):
        """启动自动巩固（后台定时器）。

        根据 config.consolidation.interval_hours 定时触发 Sleep Loop。
        非阻塞，在后台线程中运行。
        """
        if not self.config.consolidation.enabled:
            logger.info("Auto-consolidation is disabled in config")
            return

        interval = self.config.consolidation.interval_hours * 3600

        def _loop():
            while getattr(self, '_auto_consolidate_running', True):
                try:
                    logger.info("Auto-consolidate: running Sleep Loop...")
                    result = self.consolidate()
                    logger.info(
                        "Auto-consolidate: scanned=%d clusters=%d rules=%d",
                        result.scanned, result.clusters_found,
                        result.rules_extracted,
                    )
                except Exception as e:
                    logger.error("Auto-consolidate error: %s", e)

                # 等待下一次触发
                for _ in range(int(interval)):
                    if not getattr(self, '_auto_consolidate_running', True):
                        break
                    import time as _time
                    _time.sleep(1)

        self._auto_consolidate_running = True
        t = threading.Thread(target=_loop, daemon=True, name="memvault-consolidator")
        t.start()
        logger.info("Auto-consolidate started (interval: %dh)",
                    self.config.consolidation.interval_hours)

    def stop_auto_consolidate(self):
        """停止自动巩固。"""
        self._auto_consolidate_running = False
        logger.info("Auto-consolidate stopped")

    def get_kg_rules(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取当前知识图谱规则。

        Args:
            limit: 返回条数

        Returns:
            规则字典列表 [{"id": ..., "rule": ..., "confidence": ...}, ...]
        """
        if hasattr(self._store.storage, 'get_kg_rules'):
            return self._store.storage.get_kg_rules(limit=limit)
        return []

    # ── 统计 ──

    def stats(self) -> MemoryStats:
        """获取记忆系统统计信息。"""
        return self._store.stats()

    def close(self):
        """关闭存储连接和自动巩固。"""
        if hasattr(self, '_auto_consolidate_running'):
            self._auto_consolidate_running = False
        self._store.close()

    # ── 内部 ──

    def _create_default_embedder(self) -> AbstractEmbedder:
        cfg = self.config.embedding
        if cfg.provider == "mock":
            return MockEmbedder()
        return OllamaEmbedder(
            model=cfg.model,
            base_url=cfg.base_url,
            timeout=cfg.timeout_seconds,
        )

    def _create_default_llm(self) -> LLMClient:
        cfg = self.config.llm
        return LLMClient(
            provider=cfg.provider,
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_seconds,
        )

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                embedder=self._embedder,
                storage=self._store.storage,
                config=self.config.retrieval,
                memory_config=self.config.memory,
            )
        return self._retriever

    def _get_compressor(self) -> MemoryCompressor:
        if self._compressor is None:
            summarizer = None
            if self.config.llm.api_key:
                summarizer = create_llm_summarizer(self._llm)
            self._compressor = MemoryCompressor(
                summarizer=summarizer,
                primacy_count=self.config.compression.primacy_count,
                recency_count=self.config.compression.recency_count,
                group_size=self.config.compression.group_size,
                max_compressed_slots=self.config.compression.max_compressed_slots,
                threshold=self.config.compression.threshold,
                backup_enabled=self.config.compression.backup_enabled,
            )
        return self._compressor

    def _get_consolidator(self) -> SleepConsolidator:
        if self._consolidator is None:
            self._consolidator = SleepConsolidator(
                embedder=self._embedder,
                llm=self._llm,
                storage=self._store.storage,
                config=self.config.consolidation,
            )
        return self._consolidator


def _h(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
