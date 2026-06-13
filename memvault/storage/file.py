"""
FileStorage — 文件存储后端（兼容旧版 JSON 格式）

保留原有的 MEMORY.md / USER.md / memory_meta.json 文件格式。
适用于：
  - 从 memvault-test 迁移（无缝过渡）
  - 轻量部署（零数据库依赖）
  - 人工可读可编辑（纯文本 Markdown）

与旧版的关键区别：
  - 实现 AbstractStorage 接口，可无缝替换
  - writeback_strategy 支持 "immediate" 和 "batch"
  - 去掉了 WebUI 硬编码的字符上限
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from memvault.storage.base import AbstractStorage
from memvault.types import MemoryEntry

# 与旧版兼容
import hashlib

ENTRY_DELIMITER = "\n§\n"


def _entry_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


class FileStorage(AbstractStorage):
    """文件存储后端。

    文件结构:
      {storage_dir}/
        MEMORY.md           ← 情景记忆（§ 分隔）
        USER.md             ← 用户画像（§ 分隔）
        memory_meta.json    ← 元数据（JSON）
    """

    def __init__(self, storage_dir: str = "./memdata",
                 writeback_strategy: str = "immediate"):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._writeback = writeback_strategy
        self._lock = threading.Lock()
        self._dirty: Dict[str, bool] = {"memory": False, "user": False}

        # 内存缓存
        self._memory_entries: List[str] = []
        self._user_entries: List[str] = []
        self._loaded = False

    # ── 文件路径 ──

    @property
    def memory_file(self) -> Path:
        return self._dir / "MEMORY.md"

    @property
    def user_file(self) -> Path:
        return self._dir / "USER.md"

    @property
    def meta_file(self) -> Path:
        return self._dir / "memory_meta.json"

    # ── 懒加载 ──

    def _ensure_loaded(self):
        """首次访问时从磁盘加载。"""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            self._memory_entries = self._read_markdown(self.memory_file)
            self._user_entries = self._read_markdown(self.user_file)
            self._loaded = True

    @staticmethod
    def _read_markdown(path: Path) -> List[str]:
        """读取 § 分隔的 Markdown 文件。"""
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []
        if not raw.strip():
            return []
        return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]

    def _write_markdown(self, path: Path, entries: List[str]):
        """写入 § 分隔的 Markdown 文件。"""
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        path.write_text(content, encoding="utf-8")

    def _entries_for(self, target: str) -> List[str]:
        self._ensure_loaded()
        if target == "user":
            return self._user_entries
        return self._memory_entries

    def _flush(self, target: str):
        """写回磁盘（仅在 immediate 模式下立即写入）。"""
        if self._writeback == "immediate":
            path = self.user_file if target == "user" else self.memory_file
            entries = self._entries_for(target)
            self._write_markdown(path, entries)

    def flush_all(self):
        """批量模式下手动写回所有文件。"""
        with self._lock:
            if self._dirty.get("memory"):
                self._write_markdown(self.memory_file, self._memory_entries)
                self._dirty["memory"] = False
            if self._dirty.get("user"):
                self._write_markdown(self.user_file, self._user_entries)
                self._dirty["user"] = False

    # ── CRUD ──

    def add_entry(self, target: str, content: str,
                  importance: float = 0.5) -> str:
        content = content.strip()
        if not content:
            raise ValueError("Content cannot be empty")

        self._ensure_loaded()
        entries = self._entries_for(target)
        entry_id = _entry_hash(content)

        if content in entries:
            return entry_id  # 重复，静默忽略

        entries.append(content)
        self._store_importance(entry_id, content, importance)
        self._flush(target)
        return entry_id

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        self._ensure_loaded()

        # 遍历查找（FileStorage 无索引）
        for target in ("memory", "user"):
            entries = self._entries_for(target)
            for e in entries:
                if _entry_hash(e) == entry_id:
                    meta = self._load_meta().get(entry_id, {})
                    return MemoryEntry(
                        id=entry_id,
                        target=target,
                        content=e,
                        created_at=meta.get("created_at", 0),
                        retrieval_count=meta.get("retrieval_count", 0),
                        correction_count=meta.get("correction_count", 0),
                        importance=meta.get("importance", 0.5),
                        last_retrieved_at=meta.get("last_retrieved_at"),
                    )
        return None

    def get_entries(self, target: str,
                    limit: Optional[int] = None,
                    offset: int = 0,
                    min_weight: Optional[float] = None,
                    order_by: str = "weight") -> List[MemoryEntry]:
        self._ensure_loaded()
        entries = self._entries_for(target)
        meta = self._load_meta()

        result = []
        for e in entries:
            entry_id = _entry_hash(e)
            m = meta.get(entry_id, {})
            result.append(MemoryEntry(
                id=entry_id,
                target=target,
                content=e,
                created_at=m.get("created_at", 0),
                retrieval_count=m.get("retrieval_count", 0),
                correction_count=m.get("correction_count", 0),
                importance=m.get("importance", 0.5),
                last_retrieved_at=m.get("last_retrieved_at"),
            ))

        # 排序
        if order_by == "importance":
            result.sort(key=lambda x: x.importance, reverse=True)
        elif order_by == "created_at":
            result.sort(key=lambda x: x.created_at, reverse=True)
        # "weight" 留待应用层计算

        return result[offset:offset + limit] if limit else result[offset:]

    def update_entry(self, entry_id: str, content: str) -> bool:
        self._ensure_loaded()
        content = content.strip()

        for target in ("memory", "user"):
            entries = self._entries_for(target)
            for i, e in enumerate(entries):
                if _entry_hash(e) == entry_id:
                    entries[i] = content
                    # 保留旧元数据，只更新修正计数
                    meta = self._load_meta()
                    if entry_id in meta:
                        meta[entry_id]["correction_count"] = \
                            meta[entry_id].get("correction_count", 0) + 1
                    self._save_meta(meta)
                    self._flush(target)
                    return True
        return False

    def delete_entry(self, entry_id: str) -> bool:
        self._ensure_loaded()

        for target in ("memory", "user"):
            entries = self._entries_for(target)
            for i, e in enumerate(entries):
                if _entry_hash(e) == entry_id:
                    entries.pop(i)
                    self._flush(target)
                    return True
        return False

    # ── 元数据 ──

    def update_meta(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        meta = self._load_meta()
        if entry_id not in meta:
            meta[entry_id] = {}
        meta[entry_id].update(updates)
        self._save_meta(meta)
        return True

    def batch_update_meta(self, updates: Dict[str, Dict[str, Any]]) -> int:
        meta = self._load_meta()
        count = 0
        for entry_id, fields in updates.items():
            if entry_id in meta:
                meta[entry_id].update(fields)
                count += 1
        if count > 0:
            self._save_meta(meta)
        return count

    # ── 统计与清理 ──

    def count_entries(self, target: Optional[str] = None) -> int:
        self._ensure_loaded()
        if target == "user":
            return len(self._user_entries)
        if target == "memory":
            return len(self._memory_entries)
        return len(self._memory_entries) + len(self._user_entries)

    def char_count(self, target: str) -> int:
        self._ensure_loaded()
        entries = self._entries_for(target)
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def prune_below_weight(self, target: str, threshold: float) -> int:
        """清理低权重条目（需先计算权重，这里使用 importance 近似）。"""
        self._ensure_loaded()
        entries = self._entries_for(target)
        meta = self._load_meta()

        keep = []
        removed = 0
        for e in entries:
            entry_id = _entry_hash(e)
            m = meta.get(entry_id, {})
            imp = m.get("importance", 0.5)
            if imp < threshold:
                removed += 1
            else:
                keep.append(e)

        if removed > 0:
            if target == "user":
                self._user_entries = keep
            else:
                self._memory_entries = keep
            self._flush(target)

        return removed

    def vacuum(self) -> int:
        """清理已删除条目的 meta 残留。"""
        self._ensure_loaded()
        meta = self._load_meta()
        current_ids = set()

        for entries in (self._memory_entries, self._user_entries):
            for e in entries:
                current_ids.add(_entry_hash(e))

        # 找出过期 ID
        stale = [eid for eid in meta if eid not in current_ids]
        for eid in stale:
            del meta[eid]

        if stale:
            self._save_meta(meta)

        return len(stale)

    def get_all_meta(self) -> Dict[str, Dict[str, Any]]:
        return self._load_meta()

    def close(self):
        self.flush_all()

    # ── 内部 ──

    def _load_meta(self) -> Dict[str, Any]:
        mp = self.meta_file
        if not mp.exists():
            return {}
        try:
            return json.loads(mp.read_text(encoding="utf-8")) or {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_meta(self, meta: Dict[str, Any]):
        mp = self.meta_file
        mp.parent.mkdir(parents=True, exist_ok=True)
        tmp = mp.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                          encoding="utf-8")
            tmp.replace(mp)  # 原子替换
        except OSError:
            pass

    def _store_importance(self, entry_id: str, content: str,
                          importance: float):
        """存储 initial importance 到元数据。"""
        meta = self._load_meta()
        if entry_id not in meta:
            meta[entry_id] = {
                "created_at": time.time(),
                "importance": importance,
            }
        self._save_meta(meta)
