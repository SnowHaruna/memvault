"""
MemoryStorage — 内存存储后端（测试/临时使用）

纯内存存储，无持久化。适用于：
  - 单元测试（每次测试独立，不影响磁盘）
  - 临时会话（不需要保存记忆）
  - CI/CD 环境（无文件系统依赖）
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from memvault.storage.base import AbstractStorage
from memvault.types import MemoryEntry


def _entry_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


class MemoryStorage(AbstractStorage):
    """纯内存存储后端。

    所有数据存储在 Python dict 中，进程退出后清空。
    线程安全（使用 threading.Lock）。
    """

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._entries: Dict[str, MemoryEntry] = {}  # entry_id → MemoryEntry
        self._by_target: Dict[str, List[str]] = {"memory": [], "user": []}
        # entry_id list, 保持插入顺序

    def add_entry(self, target: str, content: str,
                  importance: float = 0.5) -> str:
        content = content.strip()
        if not content:
            raise ValueError("Content cannot be empty")

        entry_id = _entry_hash(content)
        now = time.time()

        with self._lock:
            # 检查重复
            for eid in self._by_target.get(target, []):
                if self._entries[eid].content == content:
                    return entry_id

            entry = MemoryEntry(
                id=entry_id,
                target=target,
                content=content,
                created_at=now,
                importance=importance,
            )
            self._entries[entry_id] = entry
            self._by_target.setdefault(target, []).append(entry_id)

        return entry_id

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    def get_entries(self, target: str,
                    limit: Optional[int] = None,
                    offset: int = 0,
                    min_weight: Optional[float] = None,
                    order_by: str = "weight") -> List[MemoryEntry]:
        with self._lock:
            ids = self._by_target.get(target, [])
            entries = [self._entries[eid] for eid in ids if eid in self._entries]

            # 排序
            if order_by == "importance":
                entries.sort(key=lambda x: x.importance, reverse=True)
            elif order_by == "created_at":
                entries.sort(key=lambda x: x.created_at, reverse=True)
            # "weight" 由应用层计算

            if min_weight is not None:
                entries = [e for e in entries if e.importance >= min_weight * 0.7]

            return entries[offset:offset + limit] if limit else entries[offset:]

    def update_entry(self, entry_id: str, content: str) -> bool:
        with self._lock:
            if entry_id in self._entries:
                self._entries[entry_id].content = content.strip()
                self._entries[entry_id].correction_count += 1
                return True
        return False

    def delete_entry(self, entry_id: str) -> bool:
        with self._lock:
            if entry_id in self._entries:
                target = self._entries[entry_id].target
                if entry_id in self._by_target.get(target, []):
                    self._by_target[target].remove(entry_id)
                del self._entries[entry_id]
                return True
        return False

    def update_meta(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        with self._lock:
            if entry_id not in self._entries:
                return False
            entry = self._entries[entry_id]
            for key, value in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
        return True

    def batch_update_meta(self, updates: Dict[str, Dict[str, Any]]) -> int:
        count = 0
        with self._lock:
            for entry_id, fields in updates.items():
                if entry_id in self._entries:
                    entry = self._entries[entry_id]
                    for key, value in fields.items():
                        if hasattr(entry, key):
                            setattr(entry, key, value)
                    count += 1
        return count

    def count_entries(self, target: Optional[str] = None) -> int:
        with self._lock:
            if target:
                return len(self._by_target.get(target, []))
            return len(self._entries)

    def char_count(self, target: str) -> int:
        with self._lock:
            ids = self._by_target.get(target, [])
            return sum(len(self._entries[eid].content)
                      for eid in ids if eid in self._entries)

    def prune_below_weight(self, target: str, threshold: float) -> int:
        with self._lock:
            ids = self._by_target.get(target, [])
            to_remove = []
            for eid in ids:
                entry = self._entries.get(eid)
                if entry and entry.importance < threshold:
                    to_remove.append(eid)

            for eid in to_remove:
                ids.remove(eid)
                del self._entries[eid]

            self._by_target[target] = ids
            return len(to_remove)

    def vacuum(self) -> int:
        """清理不再被引用的条目。"""
        with self._lock:
            before = len(self._entries)
            all_ids = set()
            for ids in self._by_target.values():
                all_ids.update(ids)
            stale = [eid for eid in self._entries if eid not in all_ids]
            for eid in stale:
                del self._entries[eid]
            return before - len(self._entries)

    def get_all_meta(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            result = {}
            for eid, entry in self._entries.items():
                result[eid] = {
                    "created_at": entry.created_at,
                    "retrieval_count": entry.retrieval_count,
                    "correction_count": entry.correction_count,
                    "importance": entry.importance,
                    "last_retrieved_at": entry.last_retrieved_at,
                }
            return result

    # ── 知识图谱支持 ──

    def add_kg_rule(self, rule: str, confidence: float = 1.0,
                    source: str = "sleep_loop") -> str:
        """添加知识图谱规则。"""
        rule_id = _entry_hash(rule)
        with self._lock:
            if not hasattr(self, '_kg_rules'):
                self._kg_rules: Dict[str, Dict[str, Any]] = {}
            self._kg_rules[rule_id] = {
                "id": rule_id,
                "rule": rule,
                "confidence": confidence,
                "source": source,
                "created_at": time.time(),
            }
        return rule_id

    def get_kg_rules(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取知识图谱规则。"""
        with self._lock:
            if not hasattr(self, '_kg_rules'):
                return []
            rules = sorted(
                self._kg_rules.values(),
                key=lambda x: x.get("confidence", 0),
                reverse=True,
            )
            return rules[:limit]

    def kg_node_count(self) -> int:
        """KG 节点数。"""
        with self._lock:
            if not hasattr(self, '_kg_rules'):
                return 0
            return len(self._kg_rules)

    def close(self):
        """清空内存。"""
        with self._lock:
            self._entries.clear()
            self._by_target = {"memory": [], "user": []}
            if hasattr(self, '_kg_rules'):
                self._kg_rules.clear()
