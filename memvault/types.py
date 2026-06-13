"""
memvault 公共类型定义

所有核心数据模型集中于此，确保全库类型一致性。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """单条记忆条目的完整表示。

    Attributes:
        id: 稳定哈希 ID（8 字符 hex）
        target: 存储目标 ("memory" | "user")
        content: 记忆文本
        created_at: 创建时间戳（epoch seconds）
        retrieval_count: 被检索次数
        correction_count: 被修正次数
        importance: 情绪重要性评分 (0.0-1.0)
        last_retrieved_at: 上次检索时间戳
        weight: 当前计算权重（缓存值，可能过期）
        meta: 扩展元数据字典
    """
    id: str
    target: str = "memory"
    content: str = ""
    created_at: float = 0.0
    retrieval_count: int = 0
    correction_count: int = 0
    importance: float = 0.5
    last_retrieved_at: Optional[float] = None
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """用户画像数据。

    Attributes:
        entries: 用户画像条目列表
        half_life_days: 用户画像半衰期（默认 14 天）
        char_limit: 字符上限
    """
    entries: List[str] = field(default_factory=list)
    half_life_days: float = 14.0
    char_limit: int = 8000


@dataclass
class ConsolidationResult:
    """Sleep Loop 巩固循环的执行结果。

    Attributes:
        clusters_found: 发现的聚类数量
        rules_extracted: 提取的抽象规则数
        rules: 提取的规则文本列表
        merged_entries: 被合并/抽象的 L1 条目数
        confidence_scores: 每条规则的置信度 (0.0-1.0)
        scanned: 扫描的 L1 条目总数
        kg_nodes_added: 新增知识图谱节点数
        contradictions: 检测到的矛盾数
        elapsed_seconds: 执行耗时
        log: 详细日志文本
    """
    clusters_found: int = 0
    rules_extracted: int = 0
    rules: List[str] = field(default_factory=list)
    merged_entries: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    scanned: int = 0
    kg_nodes_added: int = 0
    contradictions: int = 0
    elapsed_seconds: float = 0.0
    log: str = ""


@dataclass
class SearchResult:
    """单条检索结果。

    Attributes:
        entry: 记忆条目
        score: 最终检索分数（RRF + weight + ColBERT）
        dense_score: Dense 路原始相似度
        sparse_score: Sparse 路 BM25 分数
        colbert_score: ColBERT MaxSim 分数
        source: 结果来源 ("dense" | "sparse" | "fusion" | "fallback")
    """
    entry: MemoryEntry
    score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    colbert_score: Optional[float] = None
    source: str = "fusion"


@dataclass
class MemoryStats:
    """记忆系统统计信息。

    Attributes:
        total_entries: 总记忆条目数
        memory_entries: MEMORY.md 条目数
        user_entries: USER.md 条目数
        memory_chars: MEMORY.md 字符数
        user_chars: USER.md 字符数
        memory_limit: MEMORY.md 字符上限
        user_limit: USER.md 字符上限
        meta_count: 元数据条目数
        kg_nodes: 知识图谱节点数
        index_exists: 向量索引是否存在
        avg_weight: 当前平均权重
        avg_importance: 当前平均重要性
    """
    total_entries: int = 0
    memory_entries: int = 0
    user_entries: int = 0
    memory_chars: int = 0
    user_chars: int = 0
    memory_limit: int = 16000
    user_limit: int = 8000
    meta_count: int = 0
    kg_nodes: int = 0
    index_exists: bool = False
    avg_weight: float = 0.0
    avg_importance: float = 0.0
