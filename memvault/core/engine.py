"""
memvault 核心引擎 — MemoryStore

存储协调层：通过可插拔的 AbstractStorage 后端执行 CRUD，
不直接进行文件 I/O。

与测试期 memcore/engine.py 的关键区别：
  - 存储后端可插拔（SQLite / File / Memory），不再硬编码 JSON 文件
  - 增量元数据更新（不再每次 add() 写全量 meta）
  - 批量添加 add_batch()（不再逐条 save_to_disk()）
  - 显式字符上限裁剪（不再假设 2200 硬编码）
  - 支持 prune + vacuum 清理机制
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from memvault.config import MemoryVaultConfig
from memvault.core.decay import compute_weight, compute_importance
from memvault.core.formatter import render_markdown_block
from memvault.storage.base import AbstractStorage
from memvault.storage.sqlite import SQLiteStorage
from memvault.types import MemoryEntry, MemoryStats

# 与旧版兼容的常量
ENTRY_DELIMITER = "\n§\n"


class MemoryStore:
    """有界、存储后端无关的记忆引擎。

    两个存储目标：
      - "memory": 情景记忆（AI 的私人笔记）
      - "user":  用户画像（偏好、风格、期望）

    核心职责：
      1. 条目 CRUD（委托给 AbstractStorage）
      2. 字符预算管理（超限自动裁剪最低权重条目）
      3. 权重计算协调（委托给 decay.py）
      4. 快照生成（注入 LLM system prompt）
    """

    def __init__(
        self,
        config: Optional[MemoryVaultConfig] = None,
        storage: Optional[AbstractStorage] = None,
    ):
        """
        Args:
            config: 全局配置（默认 MemoryVaultConfig()）
            storage: 存储后端（默认 SQLiteStorage）
        """
        if config is None:
            config = MemoryVaultConfig()
        self.config = config

        if storage is None:
            # 根据配置选择存储后端
            backend = config.storage.backend
            if backend == "sqlite":
                storage = SQLiteStorage(config.storage.path)
            elif backend == "file":
                from memvault.storage.file import FileStorage
                storage = FileStorage(config.storage.path)
            elif backend == "memory":
                from memvault.storage.memory import MemoryStorage
                storage = MemoryStorage()
            else:
                storage = SQLiteStorage(config.storage.path)

        self._storage = storage
        self._snapshot: Dict[str, str] = {"memory": "", "user": ""}

    # ── 存储后端访问 ──
    @property
    def storage(self) -> AbstractStorage:
        """获取底层存储后端（供检索器等使用）。"""
        return self._storage

    # ═══════════════════════════════════════════════════════
    # CRUD
    # ═══════════════════════════════════════════════════════

    def add(
        self,
        target: str,
        content: str,
        importance: Optional[float] = None,
        skip_compress: bool = False,
    ) -> Dict[str, Any]:
        """添加一条记忆。

        自动评分 + 超限裁剪（按权重摘除最低条目）。

        Args:
            target: "memory" | "user"
            content: 记忆文本
            importance: 手动指定重要性（None 则自动评分）
            skip_compress: 延迟压缩标记（批量导入时使用）

        Returns:
            {"success": bool, "entry_id": str, "trimmed": int, ...}
        """
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # 检查重复
        existing = self._storage.get_entries(target, limit=1000)
        for entry in existing:
            if entry.content == content:
                return {
                    "success": True,
                    "entry_id": entry.id,
                    "message": "Entry already exists (no duplicate added).",
                    "trimmed": 0,
                }

        # 自动评分
        if importance is None:
            try:
                score_result = compute_importance(content)
                importance = score_result.get("importance", 0.5)
            except Exception:
                importance = 0.5

        # 字符预算检查 → 超限自动裁剪
        limit = self._char_limit(target)
        current_chars = self._storage.char_count(target)
        new_chars = current_chars + len(content) + len(ENTRY_DELIMITER)
        trimmed = 0

        while new_chars > limit:
            # 找出权重最低的条目
            worst_id = self._find_lowest_weight_entry(target)
            if worst_id is None:
                break

            removed = self._storage.get_entry(worst_id)
            if removed:
                new_chars -= len(removed.content) + len(ENTRY_DELIMITER)

            self._storage.delete_entry(worst_id)
            trimmed += 1

        # 写入
        entry_id = self._storage.add_entry(target, content, importance)

        # 更新快照
        self._rebuild_snapshot(target)

        msg = "Entry added."
        if trimmed > 0:
            msg += f" (auto-trimmed {trimmed} stale entries)"

        return {
            "success": True,
            "entry_id": entry_id,
            "target": target,
            "trimmed": trimmed,
            "importance": importance,
            "char_usage": f"{self._storage.char_count(target):,}/{limit:,}",
            "message": msg,
            "skip_compress": skip_compress,
        }

    def add_batch(
        self,
        target: str,
        items: List[str],
        skip_compress: bool = True,
    ) -> Dict[str, Any]:
        """批量添加记忆。

        与逐条 add() 相比，跳过逐条裁剪检查，添加完成后统一裁剪。
        适合大规模导入场景（从 539s → < 10s 的关键优化）。

        Args:
            target: "memory" | "user"
            items: 记忆文本列表
            skip_compress: 批量导入时建议 True（导入后统一压缩）

        Returns:
            {"success": bool, "added": int, "skipped": int, ...}
        """
        entries = []
        for content in items:
            content = content.strip()
            if not content:
                continue
            # 自动评分
            try:
                score = compute_importance(content)
                imp = score.get("importance", 0.5)
            except Exception:
                imp = 0.5
            entries.append((content, imp))

        if not entries:
            return {"success": False, "error": "No valid entries to add."}

        # 批量插入（单事务）
        try:
            ids = self._storage.add_batch(target, entries)
        except AttributeError:
            # FileStorage 没有 add_batch，回退到逐条
            ids = []
            for content, importance in entries:
                eid = self._storage.add_entry(target, content, importance)
                ids.append(eid)

        # 统一裁剪
        limit = self._char_limit(target)
        trimmed = 0
        while self._storage.char_count(target) > limit:
            worst_id = self._find_lowest_weight_entry(target)
            if worst_id is None:
                break
            self._storage.delete_entry(worst_id)
            trimmed += 1

        self._rebuild_snapshot(target)

        return {
            "success": True,
            "added": len(ids),
            "skipped": len(items) - len(ids),
            "trimmed": trimmed,
            "target": target,
            "char_usage": f"{self._storage.char_count(target):,}/{limit:,}",
            "skip_compress": skip_compress,
        }

    def get(self, target: str = "memory",
            limit: int = 50, offset: int = 0,
            min_weight: Optional[float] = None) -> List[MemoryEntry]:
        """获取记忆条目列表。

        Args:
            target: "memory" | "user"
            limit: 返回条数上限
            offset: 偏移量
            min_weight: 仅返回权重 >= 此值的条目

        Returns:
            MemoryEntry 列表
        """
        entries = self._storage.get_entries(
            target, limit=limit, offset=offset,
            min_weight=min_weight,
        )

        # 计算实时权重
        meta = self._storage.get_all_meta()
        now = time.time()
        for entry in entries:
            entry.weight = compute_weight(
                entry.id, meta, now=now,
                half_life_days=self.config.memory.half_life_days,
                emotional_half_life_days=self.config.memory.emotional_half_life_days,
                grace_period_hours=self.config.memory.grace_period_hours,
                weight_floor=self.config.memory.weight_floor,
                weight_ceiling=self.config.memory.weight_ceiling,
            )

        return entries

    def search(
        self,
        query: str,
        target: str = "memory",
        top_k: int = 10,
    ) -> List[MemoryEntry]:
        """全文搜索（关键词匹配）。

        优先使用 FTS5（SQLite），回退到遍历子串匹配。

        Args:
            query: 搜索关键词
            target: 存储目标
            top_k: 返回条数

        Returns:
            匹配的 MemoryEntry 列表（按权重降序）
        """
        # 尝试 FTS5（SQLite 特有）
        if hasattr(self._storage, 'fts_search'):
            entries = self._storage.fts_search(target, query, limit=top_k * 3)
        else:
            # 回退：获取所有条目，子串匹配
            all_entries = self._storage.get_entries(target)
            entries = [e for e in all_entries if query.lower() in e.content.lower()]
            entries = entries[:top_k * 3]

        # 计算权重并排序
        meta = self._storage.get_all_meta()
        now = time.time()
        for entry in entries:
            entry.weight = compute_weight(
                entry.id, meta, now=now,
                half_life_days=self.config.memory.half_life_days,
                emotional_half_life_days=self.config.memory.emotional_half_life_days,
                grace_period_hours=self.config.memory.grace_period_hours,
                weight_floor=self.config.memory.weight_floor,
                weight_ceiling=self.config.memory.weight_ceiling,
            )

        entries.sort(key=lambda e: e.weight, reverse=True)
        return entries[:top_k]

    def update(self, target: str, old_text: str,
               new_content: str) -> Dict[str, Any]:
        """替换包含 old_text 的条目。

        Args:
            target: "memory" | "user"
            old_text: 要替换的文本（子串匹配）
            new_content: 新内容

        Returns:
            {"success": bool, ...}
        """
        old_text = old_text.strip()
        new_content = new_content.strip()

        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use delete() to remove."}

        entries = self._storage.get_entries(target)
        matches = [e for e in entries if old_text in e.content]

        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text[:80]}'."}

        if len(matches) > 1:
            unique_contents = {e.content for e in matches}
            if len(unique_contents) > 1:
                previews = [e.content[:80] + ("..." if len(e.content) > 80 else "")
                           for e in matches]
                return {
                    "success": False,
                    "error": "Multiple entries matched. Be more specific.",
                    "matches": previews,
                }

        entry = matches[0]

        # 检查替换后预算
        old_len = len(entry.content)
        new_len = len(new_content)
        current = self._storage.char_count(target)
        limit = self._char_limit(target)
        if current - old_len + new_len > limit:
            return {
                "success": False,
                "error": (
                    f"Replacement would exceed memory limit "
                    f"({current - old_len + new_len:,}/{limit:,} chars)."
                ),
            }

        # 执行替换
        ok = self._storage.update_entry(entry.id, new_content)
        if ok:
            self._rebuild_snapshot(target)
            return {"success": True, "entry_id": entry.id, "message": "Entry replaced."}
        return {"success": False, "error": "Entry not found."}

    def delete(self, target: str, old_text: str) -> Dict[str, Any]:
        """删除包含 old_text 的条目。

        Args:
            target: "memory" | "user"
            old_text: 要删除的文本（子串匹配）

        Returns:
            {"success": bool, ...}
        """
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        entries = self._storage.get_entries(target)
        matches = [e for e in entries if old_text in e.content]

        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text[:80]}'."}

        if len(matches) > 1:
            unique_contents = {e.content for e in matches}
            if len(unique_contents) > 1:
                previews = [e.content[:80] + ("..." if len(e.content) > 80 else "")
                           for e in matches]
                return {
                    "success": False,
                    "error": "Multiple entries matched. Be more specific.",
                    "matches": previews,
                }

        ok = self._storage.delete_entry(matches[0].id)
        if ok:
            self._rebuild_snapshot(target)
            return {"success": True, "entry_id": matches[0].id, "message": "Entry removed."}
        return {"success": False, "error": "Entry not found."}

    def forget(self, target: str = "memory",
               min_weight: float = 0.3) -> int:
        """主动遗忘：清理低于权重阈值的条目。

        Args:
            target: 存储目标
            min_weight: 权重阈值

        Returns:
            清理的条目数
        """
        return self._storage.prune_below_weight(target, min_weight)

    # ═══════════════════════════════════════════════════════
    # 快照
    # ═══════════════════════════════════════════════════════

    def get_snapshot(self, target: str = "memory") -> str:
        """返回当前冻结快照文本（用于 system prompt 注入）。"""
        return self._rebuild_snapshot(target)

    def context(self, target: str = "memory",
                max_entries: int = 20) -> str:
        """获取格式化后的记忆上下文（便捷方法）。

        Args:
            target: "memory" | "user"
            max_entries: 最大条目数

        Returns:
            格式化的记忆文本块
        """
        entries = self._storage.get_entries(target, limit=max_entries)
        if not entries:
            return ""

        contents = [e.content for e in entries]
        return render_markdown_block(
            target=target,
            entries=contents,
            char_limit=self._char_limit(target),
        )

    # ═══════════════════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════════════════

    def stats(self) -> MemoryStats:
        """获取系统统计信息。"""
        memory_count = self._storage.count_entries("memory")
        user_count = self._storage.count_entries("user")
        memory_chars = self._storage.char_count("memory")
        user_chars = self._storage.char_count("user")
        meta = self._storage.get_all_meta()
        meta_count = len(meta)

        # 计算平均权重和重要性
        now = time.time()
        weights = []
        imps = []
        for entry in self._storage.get_entries("memory", limit=100):
            w = compute_weight(
                entry.id, meta, now=now,
                half_life_days=self.config.memory.half_life_days,
                weight_floor=self.config.memory.weight_floor,
                weight_ceiling=self.config.memory.weight_ceiling,
            )
            weights.append(w)
            imps.append(entry.importance)

        return MemoryStats(
            total_entries=memory_count + user_count,
            memory_entries=memory_count,
            user_entries=user_count,
            memory_chars=memory_chars,
            user_chars=user_chars,
            memory_limit=self.config.memory.char_limit,
            user_limit=self.config.memory.user_char_limit,
            meta_count=meta_count,
            avg_weight=sum(weights) / len(weights) if weights else 0,
            avg_importance=sum(imps) / len(imps) if imps else 0,
        )

    # ═══════════════════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════════════════

    def prune(self, target: str = "memory",
              threshold: Optional[float] = None) -> int:
        """清理低权重条目。

        Args:
            target: 存储目标
            threshold: 权重阈值（默认使用 config.weight_floor）

        Returns:
            清理的条目数
        """
        if threshold is None:
            threshold = self.config.memory.weight_floor
        return self._storage.prune_below_weight(target, threshold)

    def vacuum(self) -> int:
        """回收存储空间。

        Returns:
            回收的字节数
        """
        return self._storage.vacuum()

    def close(self):
        """关闭存储连接。"""
        self._storage.close()

    # ═══════════════════════════════════════════════════════
    # 内部
    # ═══════════════════════════════════════════════════════

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.config.memory.user_char_limit
        return self.config.memory.char_limit

    def _find_lowest_weight_entry(self, target: str) -> Optional[str]:
        """找出目标中权重最低的条目 ID。"""
        entries = self._storage.get_entries(target, limit=1000)
        if not entries:
            return None

        meta = self._storage.get_all_meta()
        now = time.time()

        worst_id = None
        worst_weight = float("inf")

        for entry in entries:
            w = compute_weight(
                entry.id, meta, now=now,
                half_life_days=self.config.memory.half_life_days,
                emotional_half_life_days=self.config.memory.emotional_half_life_days,
                grace_period_hours=self.config.memory.grace_period_hours,
                weight_floor=self.config.memory.weight_floor,
                weight_ceiling=self.config.memory.weight_ceiling,
            )
            if w < worst_weight:
                worst_weight = w
                worst_id = entry.id

        return worst_id

    def _rebuild_snapshot(self, target: str) -> str:
        """重建冻结快照。"""
        entries = self._storage.get_entries(target, limit=100)
        contents = [e.content for e in entries]
        snapshot = render_markdown_block(
            target=target,
            entries=contents,
            char_limit=self._char_limit(target),
        )
        self._snapshot[target] = snapshot
        return snapshot

    def update_meta_on_retrieve(self, entry_ids: List[str]):
        """检索后批量更新检索计数。"""
        now = time.time()
        updates = {}
        for eid in entry_ids:
            # 获取当前计数
            entry = self._storage.get_entry(eid)
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

        if hasattr(self._storage, 'batch_update_meta'):
            self._storage.batch_update_meta(updates)
        else:
            for eid, fields in updates.items():
                self._storage.update_meta(eid, fields)
