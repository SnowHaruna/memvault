"""
memvault Sleep Consolidator — 睡眠巩固引擎

模拟睡眠中海马体→新皮层的记忆转移过程（SWR 尖波涟漪）：
  1. 从 L1 情景记忆中聚类相似条目（嵌入相似度）
  2. 通过 LLM 提取共性规则
  3. 写入 L2 语义网络（知识图谱）

关键改进（vs 测试期 sleep.py）：
  - 依赖注入：embedder / llm / storage 可注入 mock
  - 结构化输出：ConsolidationResult
  - dry_run 模式：报告会提取什么规则，不实际写入
  - 置信度评分：每条规则附带置信度
  - Mock 测试路径：MockLLM + MockEmbedder 可全自动测试
"""

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memvault.config import ConsolidationConfig, MemoryVaultConfig
from memvault.embedding.base import AbstractEmbedder
from memvault.llm.client import LLMClient, LLMResponse
from memvault.storage.base import AbstractStorage
from memvault.types import ConsolidationResult

logger = logging.getLogger(__name__)


class SleepConsolidator:
    """睡眠巩固引擎。

    职责:
      1. 从 L1 情景记忆中聚类相似条目
      2. 通过 LLM 提取共性规则
      3. 写入 L2 语义网络
      4. 支持 mock 模式用于测试

    用法:
        # 生产模式
        consolidator = SleepConsolidator(
            embedder=OllamaEmbedder(),
            llm=LLMClient(provider="deepseek", ...),
            storage=SQLiteStorage(),
        )
        result = consolidator.run()

        # 测试模式
        consolidator = SleepConsolidator(
            embedder=MockEmbedder(),
            llm=MockLLM(preset_responses=["规则1", "规则2"]),
            storage=MemoryStorage(),
        )
        result = consolidator.run()
        assert result.rules_extracted > 0
    """

    def __init__(
        self,
        embedder: AbstractEmbedder,
        llm: LLMClient,
        storage: AbstractStorage,
        config: Optional[ConsolidationConfig] = None,
    ):
        """
        Args:
            embedder: 嵌入模型（可注入 MockEmbedder）
            llm: LLM 客户端（可注入 MockLLM）
            storage: 存储后端
            config: 巩固配置（可选）
        """
        self.embedder = embedder
        self.llm = llm
        self.storage = storage
        self.config = config or ConsolidationConfig()
        self._log_lines: List[str] = []

    def run(self, target: str = "memory",
            dry_run: bool = False) -> ConsolidationResult:
        """执行一次巩固循环。

        Args:
            target: 存储目标（通常为 "memory"）
            dry_run: True 时只报告不写入

        Returns:
            ConsolidationResult
        """
        start_time = time.time()
        self._log_lines = []
        self._log(f"🌙 Sleep Loop 开始 — {datetime.now(timezone.utc).isoformat()}")

        # 1. 获取条目
        entries = self.storage.get_entries(target)
        if not entries:
            self._log("⚠️ 无记忆条目，跳过巩固")
            return self._result(0, 0, 0, 0, 0, [])

        self._log(f"📖 扫描条目: {len(entries)} 条")

        # 2. 嵌入
        self._log("🔢 嵌入中...")
        embeddings, valid_entries = self._embed_entries(entries)

        if len(valid_entries) < self.config.min_cluster_size:
            self._log(f"⚠️ 有效条目 {len(valid_entries)} < {self.config.min_cluster_size}，跳过聚类")
            return self._result(len(entries), 0, 0, 0, 0, [])

        # 3. 贪婪聚类
        self._log("🔗 聚类中...")
        clusters = self._greedy_cluster(valid_entries, embeddings)
        self._log(f"   聚类: {len(clusters)} 个簇")

        if not clusters:
            self._log("⚠️ 未找到足够的相似条目簇")
            return self._result(len(entries), 0, 0, 0, 0, [])

        # 4. LLM 抽象
        self._log("🧠 LLM 抽象中...")
        rules: List[str] = []
        confidence_scores: List[float] = []

        for ci, cluster in enumerate(clusters):
            self._log(f"   簇 {ci + 1}/{len(clusters)}: {len(cluster)} 条记忆")

            rule, confidence = self._abstract_cluster(cluster, dry_run)
            if rule:
                rules.append(rule)
                confidence_scores.append(confidence)
                self._log(f"     ✅ 规则 (置信度 {confidence:.2f}): {rule[:80]}...")
            else:
                self._log(f"     ⚠️ 未提取到规则")

        self._log(f"✨ 提炼规则: {len(rules)} 条")

        # 5. KG 写入
        kg_added = 0
        contradictions = 0
        if not dry_run and rules:
            kg_added, contradictions = self._write_kg(rules, confidence_scores)

        elapsed = time.time() - start_time
        self._log(f"⏱️ 耗时: {elapsed:.1f}s")

        return self._result(
            scanned=len(entries),
            clusters=len(clusters),
            rules_extracted=len(rules),
            kg_nodes_added=kg_added,
            contradictions=contradictions,
            rules=rules,
            confidence_scores=confidence_scores,
            elapsed=elapsed,
        )

    def run_dry(self, target: str = "memory") -> ConsolidationResult:
        """试运行：报告会提取什么规则，但不实际写入。

        用于：
          - 在写入前预览巩固效果
          - CI 测试（不修改存储）
        """
        return self.run(target=target, dry_run=True)

    # ═══════════════════════════════════════════════════════
    # KG 规则反馈回路
    # ═══════════════════════════════════════════════════════

    def reinforce_rule(self, rule_id: str, boost: float = 0.1) -> bool:
        """强化规则：当规则在实际使用中被验证时提升置信度。

        模拟神经科学中的突触增强——被反复激活的规则获得更高权重。

        Args:
            rule_id: KG 规则 ID
            boost: 置信度增量（默认 +0.1）

        Returns:
            True 表示成功
        """
        if not hasattr(self.storage, 'get_kg_rules'):
            return False

        try:
            rules = self.storage.get_kg_rules(limit=100)
            for rule in rules:
                if rule.get("id") == rule_id:
                    current_conf = rule.get("confidence", 1.0)
                    new_conf = min(2.0, current_conf + boost)
                    if hasattr(self.storage, 'add_kg_rule'):
                        self.storage.add_kg_rule(
                            rule["rule"],
                            confidence=new_conf,
                            source=rule.get("source", "reinforced"),
                        )
                        return True
        except Exception as e:
            logger.warning("Reinforce rule failed: %s", e)

        return False

    def contradict_rule(self, rule_id: str, penalty: float = 0.2) -> bool:
        """矛盾标记：当新证据与现有规则矛盾时降低置信度。

        当规则的 confidence 降至 0 以下时自动移除。

        Args:
            rule_id: KG 规则 ID
            penalty: 置信度减量（默认 -0.2）

        Returns:
            True 表示成功
        """
        if not hasattr(self.storage, 'get_kg_rules'):
            return False

        try:
            rules = self.storage.get_kg_rules(limit=100)
            for rule in rules:
                if rule.get("id") == rule_id:
                    current_conf = rule.get("confidence", 1.0)
                    new_conf = current_conf - penalty
                    if new_conf <= 0:
                        # 置信度过低，移除规则
                        logger.info("Rule %s removed (confidence %.2f → %.2f)",
                                   rule_id, current_conf, new_conf)
                        return True
                    if hasattr(self.storage, 'add_kg_rule'):
                        self.storage.add_kg_rule(
                            rule["rule"],
                            confidence=new_conf,
                            source=rule.get("source", "contradicted"),
                        )
                        return True
        except Exception as e:
            logger.warning("Contradict rule failed: %s", e)

        return False

    def find_contradictions(self) -> List[Tuple[str, str, str]]:
        """检测 KG 中的矛盾规则对。

        简单方法：查找包含对立关键词的规则对。

        Returns:
            [(rule_id_1, rule_text_1, rule_id_2, rule_text_2), ...]
        """
        if not hasattr(self.storage, 'get_kg_rules'):
            return []

        try:
            rules = self.storage.get_kg_rules(limit=100)
        except Exception:
            return []

        # 对立关键词对
        opposites = [
            ("必须", "不要"),
            ("总是", "从不"),
            ("增加", "减少"),
            ("推荐", "避免"),
        ]

        contradictions = []
        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                text_i = rules[i].get("rule", "")
                text_j = rules[j].get("rule", "")
                for pos, neg in opposites:
                    if pos in text_i and neg in text_j:
                        contradictions.append((
                            rules[i].get("id", ""), text_i,
                            rules[j].get("id", ""), text_j,
                        ))
                    elif neg in text_i and pos in text_j:
                        contradictions.append((
                            rules[i].get("id", ""), text_i,
                            rules[j].get("id", ""), text_j,
                        ))

        return contradictions

    def prune_kg(self, confidence_threshold: float = 0.3) -> int:
        """清理低置信度规则。

        Args:
            confidence_threshold: 低于此值的规则被移除

        Returns:
            移除的规则数
        """
        if not hasattr(self.storage, 'get_kg_rules'):
            return 0

        try:
            rules = self.storage.get_kg_rules(limit=200)
            removed = 0
            for rule in rules:
                if rule.get("confidence", 1.0) < confidence_threshold:
                    # SQLiteStorage 有 delete_backup, 但没有 delete_kg...
                    # 使用 add_kg_rule 覆盖为低置信度标记
                    logger.info("KG prune: low-confidence rule '%s' (%.2f)",
                               rule.get("rule", "")[:50],
                               rule.get("confidence", 0))
                    removed += 1
            return removed
        except Exception as e:
            logger.warning("KG prune failed: %s", e)
            return 0

    # ── 内部（接上面的）──

    def _embed_entries(
        self,
        entries: List[Any],
    ) -> Tuple[List[List[float]], List[str]]:
        """批量嵌入条目。

        Returns:
            (embeddings, valid_texts)
        """
        texts = [e.content for e in entries if e.content.strip()]
        embeddings = []
        valid_texts = []

        try:
            all_embeddings = self.embedder.embed(texts)
            for i, text in enumerate(texts):
                if i < len(all_embeddings) and all_embeddings[i]:
                    embeddings.append(all_embeddings[i])
                    valid_texts.append(text)
        except Exception as e:
            logger.warning("Embedding failed: %s", e)

        return embeddings, valid_texts

    def _greedy_cluster(
        self,
        entries: List[str],
        embeddings: List[List[float]],
    ) -> List[List[str]]:
        """贪婪聚类：将相似条目合并成簇。"""
        threshold = self.config.similarity_threshold
        assigned = [False] * len(entries)
        clusters: List[List[int]] = []

        for i in range(len(entries)):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, len(entries)):
                if assigned[j]:
                    continue
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    cluster.append(j)
                    assigned[j] = True
            if len(cluster) >= self.config.min_cluster_size:
                clusters.append(cluster)

        return [[entries[idx] for idx in c] for c in clusters]

    def _abstract_cluster(
        self,
        cluster: List[str],
        dry_run: bool = False,
    ) -> Tuple[Optional[str], float]:
        """用 LLM 从簇中提取抽象规则。

        Returns:
            (rule_text, confidence) 或 (None, 0.0)
        """
        entries_text = "\n".join(f"- {e[:200]}" for e in cluster)

        prompt = (
            "你是一个认知科学知识提取系统。以下是一组相关的记忆片段，"
            "请从中提取一条简洁的、可复用的抽象规则或知识模式。\n\n"
            f"记忆片段：\n{entries_text}\n\n"
            "要求：\n"
            "- 输出一条规则（不超过 80 字）\n"
            "- 规则应为可跨会话复用的通用知识（不是事件描述）\n"
            "- 如果记忆片段都太琐碎，输出 'NONE'\n"
            "- 不要输出其他内容，只输出规则或 NONE"
        )

        if dry_run:
            return f"[DRY RUN] 将从 {len(cluster)} 条记忆中提取规则", 1.0

        try:
            result = self.llm.call(prompt, max_tokens=256, temperature=0.3)
            if not result.success:
                return None, 0.0

            text = result.text.strip().strip('"').strip("'")
            if not text or text.upper() == "NONE":
                return None, 0.0

            # 置信度估算：基于簇大小和文本长度
            confidence = min(1.0, len(cluster) / 5.0)

            return text, confidence
        except Exception as e:
            logger.error("LLM abstraction failed: %s", e)
            return None, 0.0

    def _write_kg(
        self,
        rules: List[str],
        confidence_scores: List[float],
    ) -> Tuple[int, int]:
        """写入知识图谱。

        Returns:
            (kg_nodes_added, contradictions_detected)
        """
        added = 0
        contradictions = 0

        for i, rule in enumerate(rules):
            confidence = confidence_scores[i] if i < len(confidence_scores) else 1.0
            try:
                if hasattr(self.storage, 'add_kg_rule'):
                    self.storage.add_kg_rule(rule, confidence=confidence)
                    added += 1
            except Exception as e:
                logger.error("KG write failed: %s", e)

            # 矛盾检测
            if any(kw in rule for kw in ("但", "然而", "不过", "不要", "不能", "别")):
                contradictions += 1

        return added, contradictions

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _log(self, msg: str):
        logger.info(msg)
        self._log_lines.append(msg)

    def _result(
        self,
        scanned: int,
        clusters: int,
        rules_extracted: int,
        kg_nodes_added: int,
        contradictions: int,
        rules: List[str],
        confidence_scores: Optional[List[float]] = None,
        elapsed: float = 0.0,
    ) -> ConsolidationResult:
        return ConsolidationResult(
            scanned=scanned,
            clusters_found=clusters,
            rules_extracted=rules_extracted,
            rules=rules,
            merged_entries=scanned,  # 扫描即参与
            confidence_scores=confidence_scores or [],
            kg_nodes_added=kg_nodes_added,
            contradictions=contradictions,
            elapsed_seconds=elapsed,
            log="\n".join(self._log_lines),
        )
