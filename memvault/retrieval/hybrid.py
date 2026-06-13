"""
HybridRetriever — 三路混合检索编排层

完整的检索管线：
  1. Dense:  bge-m3 语义向量 (LlamaIndex)
  2. Sparse: BM25 bigram 关键词 (纯 Python)
  3. ColBERT: Token MaxSim 重排序 (纯 Python)
  4. Fusion: RRF 无权重融合
  5. Weight: Ebbinghaus 权重重排

降级策略：
  - Dense 不可用 → 自动回退到纯 Sparse + ColBERT
  - Sparse 始终可用（零外部依赖）
  - 全部不可用 → 关键词子串匹配回退
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from memvault.config import RetrievalConfig, MemoryVaultConfig
from memvault.core.decay import compute_weight
from memvault.embedding.base import AbstractEmbedder
from memvault.retrieval.base import AbstractRetriever
from memvault.retrieval.dense import DenseRetriever, SimpleDenseRetriever
from memvault.retrieval.sparse import BM25SparseRetriever
from memvault.retrieval.colbert import ColbertMaxSimReranker
from memvault.retrieval.fusion import rrf_fuse
from memvault.storage.base import AbstractStorage
from memvault.types import MemoryEntry, SearchResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """三路混合检索编排器。

    管理完整的检索管线：索引构建 → 三路检索 → RRF 融合 → ColBERT → 权重重排。

    用法:
        retriever = HybridRetriever(
            embedder=ollama_embedder,
            storage=sqlite_storage,
            config=RetrievalConfig(),
        )

        # 首次或重建索引
        retriever.rebuild_index()

        # 检索
        results = retriever.retrieve("Python bug", top_k=10)
        for r in results:
            print(f"[{r.score:.3f}] {r.entry.content}")
    """

    def __init__(
        self,
        embedder: AbstractEmbedder,
        storage: AbstractStorage,
        config: Optional[RetrievalConfig] = None,
        memory_config: Optional[Any] = None,  # MemoryConfig for weight params
    ):
        """
        Args:
            embedder: 嵌入模型（用于 Dense 检索）
            storage: 存储后端
            config: 检索配置
            memory_config: 记忆配置（含衰减参数供权重重排使用）
        """
        self.embedder = embedder
        self.storage = storage
        self.config = config or RetrievalConfig()
        self.memory_config = memory_config

        # 初始化各路检索器
        # Dense 路径选择：
        #   - LlamaIndex 可用 + embedder 有 model/base_url → DenseRetriever (真实 bge-m3)
        #   - 否则 → SimpleDenseRetriever (纯 Python cosine similarity)
        try:
            import llama_index.core  # noqa: F401
            _has_llamaindex = True
        except ImportError:
            _has_llamaindex = False

        if _has_llamaindex and hasattr(embedder, 'model') and hasattr(embedder, 'base_url'):
            self._dense = DenseRetriever(
                embedder=embedder,
                embed_batch_size=16,
            )
            self._dense_impl = "llamaindex"
        else:
            self._dense = SimpleDenseRetriever(embedder=embedder)
            self._dense_impl = "simple"
            if _has_llamaindex:
                logger.info(
                    "Embedder lacks model/base_url — using SimpleDenseRetriever "
                    "(in-memory cosine similarity)"
                )
            else:
                logger.info(
                    "LlamaIndex not available — using SimpleDenseRetriever "
                    "(in-memory cosine similarity)"
                )
        self._sparse = BM25SparseRetriever()
        self._colbert = ColbertMaxSimReranker()

        # 状态
        self._index_built = False
        self._all_entries_cache: List[str] = []

        # 抑制状态（刚检索过的条目临时降权）
        self._inhibition: Dict[str, float] = {}

        # 死胡同回退
        self._last_cue: Optional[str] = None

    # ═══════════════════════════════════════════════════════
    # 公共 API
    # ═══════════════════════════════════════════════════════

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        target: str = "memory",
    ) -> List[SearchResult]:
        """执行三路混合检索。

        管线:
          1. Dense 路 (bge-m3 语义)
          2. Sparse 路 (BM25 bigram)
          3. RRF 融合 (k=60)
          4. ColBERT MaxSim 重排
          5. Ebbinghaus 权重重排
          6. 扩散抑制 + 模式分离

        Args:
            query: 查询文本
            top_k: 返回条数（默认使用 config.retrieval.top_k）
            target: 存储目标

        Returns:
            SearchResult 列表（按最终分数降序）
        """
        if top_k is None:
            top_k = self.config.top_k

        fetch_k = max(20, top_k * self.config.fetch_k_multiplier)

        # 确保索引新鲜
        self._ensure_index_fresh()

        # ── 确定启用哪些路 ──
        dense_enabled = self.config.dense and self.embedder.check_health()
        sparse_enabled = self.config.sparse  # 始终可用
        colbert_enabled = self.config.colbert

        dense_results: List[Tuple[float, str]] = []
        sparse_results: List[Tuple[float, str]] = []

        # 2a. Dense 路
        if dense_enabled and self._index_built:
            try:
                dense_results = self._dense.retrieve(query, top_k=fetch_k)
                if dense_results:
                    self._last_cue = query
            except Exception as e:
                logger.warning("Dense retrieval failed: %s", e)

        # 2b. Sparse 路
        if sparse_enabled and self._sparse.doc_count > 0:
            try:
                sparse_results = self._sparse.retrieve(query, top_k=fetch_k)
            except Exception as e:
                logger.warning("Sparse retrieval failed: %s", e)

        # 2c. 纯 Sparse 回退（无索引但有条目时）
        if not dense_results and not sparse_results and self._all_entries_cache:
            sparse_results = self._sparse.brute_retrieve(
                query, self._all_entries_cache, top_k=fetch_k
            )

        # 死胡同回退
        if not dense_results and not sparse_results:
            return self._dead_end_recovery(query, top_k)

        # ── 3. RRF Fusion ──
        fused = rrf_fuse(dense_results, sparse_results, k=self.config.rrf_k)
        fused_texts = [item for _, item in fused]

        if not fused_texts:
            return []

        # ── 4. ColBERT MaxSim 重排 ──
        colbert_candidates = fused_texts[:top_k * 2]
        if colbert_enabled:
            colbert_ranked = self._colbert.rerank(
                query, colbert_candidates, top_k=top_k
            )
        else:
            colbert_ranked = [(1.0, t) for t in colbert_candidates[:top_k]]

        # ── 5. Weight Rerank + 抑制 + 模式分离 ──
        results = self._apply_weight_rerank(query, colbert_ranked, top_k)

        # ── 6. 更新检索元数据 + 扩散抑制 ──
        retrieved_ids = [r.entry.id for r in results]
        self._update_retrieval_meta(retrieved_ids)
        self._update_inhibition(retrieved_ids)

        # 构建来源标签
        modes = []
        if dense_results:
            modes.append("D")
        if sparse_results:
            modes.append("S")
        if colbert_enabled:
            modes.append("C")

        for r in results:
            r.source = "+".join(modes)

        return results

    def rebuild_index(self, target: str = "memory") -> bool:
        """重建所有索引（Dense + Sparse）。

        从存储中读取所有条目，重建向量索引和 BM25 索引。

        Args:
            target: 存储目标（通常为 "memory"）

        Returns:
            True 表示重建成功
        """
        try:
            # 收集所有条目
            entries = self.storage.get_entries(target)
            self._all_entries_cache = [e.content for e in entries if e.content.strip()]

            if not self._all_entries_cache:
                logger.warning("No entries to index")
                self._index_built = False
                return False

            # Dense 索引
            if self.config.dense:
                try:
                    self._dense.index(self._all_entries_cache)
                except Exception as e:
                    logger.warning("Dense index build failed: %s", e)

            # Sparse 索引（始终构建，作为降级路径）
            self._sparse.index(self._all_entries_cache)

            self._index_built = True
            logger.info(
                "Index rebuilt: %d entries (dense=%s, sparse=%d docs, vocab=%d)",
                len(self._all_entries_cache),
                self.config.dense,
                self._sparse.doc_count,
                len(self._sparse._df) if hasattr(self._sparse, '_df') else 0,
            )
            return True

        except Exception as e:
            logger.error("Index rebuild failed: %s", e)
            return False

    @property
    def is_ready(self) -> bool:
        """检索器是否就绪（至少有一路可用）。"""
        return self._index_built or len(self._all_entries_cache) > 0

    # ═══════════════════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════════════════

    def _ensure_index_fresh(self):
        """检查是否需要重建索引（首次调用或存储已变更）。"""
        if not self._index_built:
            self.rebuild_index()
            return
        # 检测存储变更：条目数变化 → 索引过期，触发重建
        try:
            current_count = self.storage.count_entries("memory")
            if current_count != len(self._all_entries_cache):
                self.rebuild_index()
        except Exception:
            pass  # 非关键路径，下次显式 rebuild_index() 会修复

    def _apply_weight_rerank(
        self,
        query: str,
        colbert_ranked: List[Tuple[float, str]],
        top_k: int,
    ) -> List[SearchResult]:
        """应用 Ebbinghaus 权重重排 + 扩散抑制 + 模式分离。

        Args:
            query: 原始查询
            colbert_ranked: ColBERT 排序后的 [(score, text), ...]
            top_k: 返回条数

        Returns:
            SearchResult 列表
        """
        meta = self.storage.get_all_meta()
        now = time.time()

        # 获取权重计算参数
        if self.memory_config:
            half_life = getattr(self.memory_config, 'half_life_days', 7.0)
            emotional_hl = getattr(self.memory_config, 'emotional_half_life_days', 14.0)
            grace = getattr(self.memory_config, 'grace_period_hours', 1.0)
            usage_bonus = getattr(self.memory_config, 'usage_bonus_per_retrieval', 0.05)
            max_usage = getattr(self.memory_config, 'max_usage_bonus', 0.5)
            correction = getattr(self.memory_config, 'correction_bonus', 0.3)
            w_floor = getattr(self.memory_config, 'weight_floor', 0.3)
            w_ceiling = getattr(self.memory_config, 'weight_ceiling', 3.0)
        else:
            half_life = 7.0
            emotional_hl = 14.0
            grace = 1.0
            usage_bonus = 0.05
            max_usage = 0.5
            correction = 0.3
            w_floor = 0.3
            w_ceiling = 3.0

        final_results = []
        for c_score, text in colbert_ranked:
            # 从存储查找条目
            entries = self.storage.get_entries("memory")
            entry = None
            for e in entries:
                if e.content == text:
                    entry = e
                    break

            if entry is None:
                # 创建临时条目（不太可能，但安全处理）
                import hashlib
                eid = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
                entry = MemoryEntry(id=eid, content=text, target="memory")

            eid = entry.id

            # 扩散抑制
            inh_until = self._inhibition.get(eid, 0)
            inhibition_factor = 1.0
            if now < inh_until:
                hours_left = (inh_until - now) / 3600
                inhibition_factor = math.exp(
                    -hours_left / self.config.inhibition_half_life_hours
                )

            # Ebbinghaus 权重
            w = compute_weight(
                eid, meta, now=now,
                half_life_days=half_life,
                emotional_half_life_days=emotional_hl,
                grace_period_hours=grace,
                usage_bonus_per_retrieval=usage_bonus,
                max_usage_bonus=max_usage,
                correction_bonus=correction,
                weight_floor=w_floor,
                weight_ceiling=w_ceiling,
            )

            combined = c_score * w * inhibition_factor

            # 全局相关性过滤：分数低于阈值的条目直接丢弃
            if combined < self.config.relevance_threshold:
                continue

            final_results.append((combined, entry, c_score))

        # 模式分离：相邻条目分数过于接近 → 降权
        threshold = self.config.pattern_separation_threshold
        discount = self.config.pattern_separation_discount
        selected = []
        for i, (s, entry, c_score) in enumerate(final_results):
            if i > 0:
                prev_s = selected[-1].score
                if abs(prev_s - s) < threshold:
                    s *= discount
                    if s < self.config.relevance_threshold:
                        continue
            selected.append(SearchResult(
                entry=entry,
                score=s,
                colbert_score=c_score,
            ))

        return selected[:top_k]

    def _update_retrieval_meta(self, entry_ids: List[str]):
        """更新检索计数元数据。"""
        now = time.time()
        updates = {}
        for eid in entry_ids:
            entry = self.storage.get_entry(eid)
            if entry:
                updates[eid] = {
                    "retrieval_count": entry.retrieval_count + 1,
                    "last_retrieved_at": now,
                }
            else:
                updates[eid] = {
                    "retrieval_count": 1,
                    "last_retrieved_at": now,
                }

        if hasattr(self.storage, 'batch_update_meta'):
            self.storage.batch_update_meta(updates)
        else:
            for eid, fields in updates.items():
                self.storage.update_meta(eid, fields)

    def _update_inhibition(self, entry_ids: List[str]):
        """扩散抑制：刚检索的条目临时抑制数小时。"""
        now = time.time()
        inhibit_until = now + self.config.inhibition_half_life_hours * 3600
        for eid in entry_ids:
            self._inhibition[eid] = inhibit_until

        # 清理过期的抑制
        expired = [eid for eid, ts in self._inhibition.items() if now >= ts]
        for eid in expired:
            del self._inhibition[eid]

    def _dead_end_recovery(
        self, query: str, top_k: int
    ) -> List[SearchResult]:
        """死胡同回退：用上次成功的 query 重试。"""
        if self._last_cue and self._last_cue != query:
            logger.info("Dead-end recovery: retrying with last cue '%s'",
                        self._last_cue)
            return self.retrieve(self._last_cue, top_k=top_k)

        # 最终回退：关键词子串匹配
        return self._keyword_fallback(query, top_k)

    def _keyword_fallback(
        self, query: str, top_k: int
    ) -> List[SearchResult]:
        """最终回退：纯关键词子串匹配。"""
        if not self._all_entries_cache:
            return []

        query_lower = query.lower()
        results = []
        for text in self._all_entries_cache:
            if query_lower in text.lower():
                results.append(SearchResult(
                    entry=MemoryEntry(
                        id="", content=text, target="memory"
                    ),
                    score=0.5,  # 中性分数
                    source="keyword",
                ))

        return results[:top_k]

    # ═══════════════════════════════════════════════════════
    # 知识图谱增强检索
    # ═══════════════════════════════════════════════════════

    def retrieve_with_kg(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_kg_rules: bool = True,
    ) -> Tuple[List[SearchResult], List[str]]:
        """检索记忆 + 相关知识图谱规则。

        Args:
            query: 查询文本
            top_k: 返回条数
            include_kg_rules: 是否包含 KG 规则

        Returns:
            (search_results, kg_rules)
        """
        results = self.retrieve(query, top_k=top_k)

        kg_rules = []
        if include_kg_rules and hasattr(self.storage, 'get_kg_rules'):
            try:
                kg_entries = self.storage.get_kg_rules(limit=20)
                # 简单相关性过滤：KG 规则与 query 的子串匹配
                query_lower = query.lower()
                for kg in kg_entries:
                    rule = kg.get("rule", "")
                    if any(word in rule for word in query_lower.split()):
                        kg_rules.append(rule)
            except Exception:
                pass

        return results, kg_rules
