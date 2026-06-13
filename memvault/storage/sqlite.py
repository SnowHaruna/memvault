"""
SQLiteStorage — 生产级存储后端

使用 SQLite 替代 JSON 文件，解决三个核心问题：
  1. 增量更新：UPDATE ... WHERE id = ?（不再写全量文件）
  2. 自动清理：prune_below_weight() + VACUUM（不再无限增长）
  3. 全文索引：FTS5 替代内存子串匹配

v0.1.1 — 连接池 + 自动清理优化:
  4. 读写连接分离：写连接独占，读连接池化
  5. 增量 VACUUM：incremental_vacuum() 渐进式回收
  6. 自动优化：optimize() 定期重建索引统计
  7. 数据库大小监控：db_size() 返回字节数

零额外依赖：Python 标准库 sqlite3 + threading。
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from memvault.storage.base import AbstractStorage
from memvault.types import MemoryEntry


def _entry_hash(text: str) -> str:
    """稳定的 8 字符 ID，从条目内容计算。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


class SQLiteStorage(AbstractStorage):
    """SQLite 存储后端（v0.1.1 — 连接池 + 自动清理）。

    Schema:
      episodic_memory:
        id, target, content, created_at, retrieval_count,
        correction_count, importance, last_retrieved_at, meta_json

      FTS5 全文索引:
        episodic_memory_fts(content)

    连接架构:
      - 写连接：线程本地独占（含 WAL 写入权限）
      - 读连接池：线程安全的只读连接，支持并发读取
      - 连接复用：池内连接按需创建、LRU 回收
    """

    # 表创建 SQL（v0.1.1: 添加 auto_vacuum 支持）
    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS episodic_memory (
        id TEXT PRIMARY KEY,
        target TEXT NOT NULL CHECK(target IN ('memory', 'user')),
        content TEXT NOT NULL,
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
        retrieval_count INTEGER NOT NULL DEFAULT 0,
        correction_count INTEGER NOT NULL DEFAULT 0,
        importance REAL NOT NULL DEFAULT 0.5,
        last_retrieved_at REAL,
        meta_json TEXT DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_target_created
        ON episodic_memory(target, created_at);

    CREATE INDEX IF NOT EXISTS idx_target_weight
        ON episodic_memory(target, created_at DESC);

    -- FTS5 全文搜索
    CREATE VIRTUAL TABLE IF NOT EXISTS episodic_memory_fts
        USING fts5(content, content=episodic_memory, content_rowid=rowid);

    -- FTS5 同步触发器
    CREATE TRIGGER IF NOT EXISTS episodic_memory_ai AFTER INSERT ON episodic_memory BEGIN
        INSERT INTO episodic_memory_fts(rowid, content) VALUES (new.rowid, new.content);
    END;

    CREATE TRIGGER IF NOT EXISTS episodic_memory_ad AFTER DELETE ON episodic_memory BEGIN
        INSERT INTO episodic_memory_fts(episodic_memory_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
    END;

    CREATE TRIGGER IF NOT EXISTS episodic_memory_au AFTER UPDATE ON episodic_memory BEGIN
        INSERT INTO episodic_memory_fts(episodic_memory_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
        INSERT INTO episodic_memory_fts(rowid, content) VALUES (new.rowid, new.content);
    END;

    -- 知识图谱表
    CREATE TABLE IF NOT EXISTS knowledge_graph (
        id TEXT PRIMARY KEY,
        rule TEXT NOT NULL,
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
        source TEXT DEFAULT 'sleep_loop',
        confidence REAL DEFAULT 1.0,
        meta_json TEXT DEFAULT '{}'
    );

    -- 压缩备份表
    CREATE TABLE IF NOT EXISTS compression_backup (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target TEXT NOT NULL DEFAULT 'memory',
        entries_json TEXT NOT NULL,
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
    );
    """

    # 默认只读连接池大小
    _DEFAULT_READ_POOL_SIZE = 3
    # 读连接最大空闲时间（秒），超时后关闭
    _READ_CONN_MAX_IDLE = 300

    def __init__(
        self,
        db_path: str = "./memvault.db",
        pool_size: int = _DEFAULT_READ_POOL_SIZE,
        auto_vacuum: bool = False,
        vacuum_threshold_mb: float = 50.0,
    ):
        """
        Args:
            db_path: 数据库文件路径
            pool_size: 读连接池大小（0 = 禁用池，所有读复用写连接）
            auto_vacuum: 是否在 prune 后自动 incremental_vacuum
            vacuum_threshold_mb: 触发 auto_vacuum 的数据库大小阈值（MB）
        """
        self._db_path = Path(db_path)
        self._pool_size = max(0, pool_size)
        self._auto_vacuum = auto_vacuum
        self._vacuum_threshold_bytes = int(vacuum_threshold_mb * 1024 * 1024)

        # 线程本地存储：写连接
        self._local = threading.local()

        # 读连接池（线程安全）
        self._read_pool: List[tuple] = []  # [(conn, last_used_at), ...]
        self._pool_lock = threading.Lock()

        self._initialize()

    # ── 连接管理（v0.1.1: 读写分离）──

    def _initialize(self):
        """创建数据库和表结构。"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        try:
            conn.executescript(self._SCHEMA_SQL)
            if self._auto_vacuum:
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.commit()
        finally:
            pass  # 连接由 _get_conn 管理

    def _get_conn(self) -> sqlite3.Connection:
        """获取线程本地写连接（自动创建，WAL 模式）。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            # v0.1.1: 性能优化 PRAGMAs
            conn.execute("PRAGMA cache_size=-8000")  # 8MB 缓存
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            conn.execute("PRAGMA synchronous=NORMAL")  # WAL 模式下安全
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return self._local.conn

    def _get_read_conn(self) -> sqlite3.Connection:
        """从连接池获取只读连接（线程安全）。

        读操作使用池化连接，避免与写操作竞争。
        池耗尽时回退到写连接。
        """
        if self._pool_size == 0:
            return self._get_conn()

        now = time.time()

        with self._pool_lock:
            # 回收过期连接
            self._read_pool = [
                (c, t) for c, t in self._read_pool
                if now - t < self._READ_CONN_MAX_IDLE
            ]

            if self._read_pool:
                # LRU: 返回最近使用的
                conn, _ = self._read_pool.pop()
                return conn

        # 池空：创建新连接或回退
        current_pool_size = len(self._read_pool)
        if current_pool_size < self._pool_size:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            return conn

        # 池满，回退到写连接
        return self._get_conn()

    def _return_read_conn(self, conn: sqlite3.Connection):
        """归还只读连接到池中。"""
        if self._pool_size == 0 or conn is self._local.conn:
            return  # 是写连接，不归还

        with self._pool_lock:
            self._read_pool.append((conn, time.time()))

    def close(self):
        """关闭所有数据库连接（写连接 + 读连接池）。"""
        # 关闭写连接
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

        # 关闭读连接池
        with self._pool_lock:
            for conn, _ in self._read_pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._read_pool.clear()

    # ── CRUD ──

    def add_entry(self, target: str, content: str,
                  importance: float = 0.5) -> str:
        content = content.strip()
        if not content:
            raise ValueError("Content cannot be empty")

        entry_id = _entry_hash(content)
        now = time.time()

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO episodic_memory
                   (id, target, content, created_at, importance)
                   VALUES (?, ?, ?, ?, ?)""",
                (entry_id, target, content, now, importance),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # 重复条目，静默忽略

        return entry_id

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodic_memory WHERE id = ?",
            (entry_id,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_entries(self, target: str,
                    limit: Optional[int] = None,
                    offset: int = 0,
                    min_weight: Optional[float] = None,
                    order_by: str = "weight") -> List[MemoryEntry]:
        conn = self._get_conn()

        # 构建查询
        if order_by == "weight":
            order_clause = "created_at DESC"  # 用创建时间近似（权重需后计算）
        elif order_by == "importance":
            order_clause = "importance DESC"
        else:
            order_clause = "created_at DESC"

        conditions = ["target = ?"]
        params: list = [target]

        if min_weight is not None:
            # 用 importance 近似（实际权重需在应用层计算）
            conditions.append("importance >= ?")
            params.append(min_weight * 0.7)

        where = " AND ".join(conditions)
        sql = f"SELECT * FROM episodic_memory WHERE {where} ORDER BY {order_clause}"

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def update_entry(self, entry_id: str, content: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            """UPDATE episodic_memory
               SET content = ?,
                   correction_count = correction_count + 1
               WHERE id = ?""",
            (content.strip(), entry_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_entry(self, entry_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM episodic_memory WHERE id = ?",
            (entry_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    # ── 元数据（增量更新）──

    def update_meta(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """增量更新：只更新变化的字段。"""
        if not updates:
            return False

        conn = self._get_conn()

        # 分离已知字段和扩展 meta_json 字段
        known_fields = {
            "created_at", "retrieval_count", "correction_count",
            "importance", "last_retrieved_at",
        }

        set_parts = []
        params: list = []

        for field, value in updates.items():
            if field in known_fields:
                set_parts.append(f"{field} = ?")
                params.append(value)

        if not set_parts:
            return False

        params.append(entry_id)

        sql = f"UPDATE episodic_memory SET {', '.join(set_parts)} WHERE id = ?"
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount > 0

    def batch_update_meta(self, updates: Dict[str, Dict[str, Any]]) -> int:
        """批量更新：单事务内批量 UPDATE。"""
        conn = self._get_conn()
        count = 0

        try:
            for entry_id, fields in updates.items():
                if not fields:
                    continue
                set_parts = []
                params = []
                for field, value in fields.items():
                    if field in ("created_at", "retrieval_count", "correction_count",
                                 "importance", "last_retrieved_at"):
                        set_parts.append(f"{field} = ?")
                        params.append(value)
                if set_parts:
                    params.append(entry_id)
                    sql = f"UPDATE episodic_memory SET {', '.join(set_parts)} WHERE id = ?"
                    conn.execute(sql, params)
                    count += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return count

    # ── 统计与清理 ──

    def count_entries(self, target: Optional[str] = None) -> int:
        conn = self._get_conn()
        if target:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM episodic_memory WHERE target = ?",
                (target,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM episodic_memory"
            ).fetchone()
        return row["cnt"] if row else 0

    def char_count(self, target: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) as total FROM episodic_memory WHERE target = ?",
            (target,),
        ).fetchone()
        return row["total"] if row else 0

    def prune_below_weight(self, target: str, threshold: float) -> int:
        """清理低于权重阈值的条目。

        注意：权重是运行时计算的，这里用 importance 作为近似。
        实际使用中，应先计算权重并在应用层筛选要删除的 ID。

        如果启用 auto_vacuum，prune 后自动调用 incremental_vacuum。
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM episodic_memory WHERE target = ? AND importance < ?",
            (target, threshold),
        )
        conn.commit()
        removed = cursor.rowcount

        # 自动回收空间
        if removed > 0 and self._auto_vacuum:
            self.auto_vacuum_check()

        return removed

    def vacuum(self) -> int:
        """全量回收空间（阻塞，可能较慢）。

        对于大型数据库，推荐使用 incremental_vacuum() 渐进式回收。
        """
        conn = self._get_conn()
        before = self._db_path.stat().st_size if self._db_path.exists() else 0
        conn.execute("VACUUM")
        after = self._db_path.stat().st_size
        return max(0, before - after)

    def incremental_vacuum(self, pages: int = 100) -> int:
        """渐进式空间回收（非阻塞，推荐用于大型数据库）。

        每次回收指定数量的空闲页，可多次小批量调用，
        避免全量 VACUUM 造成的长时间阻塞。

        Args:
            pages: 每次回收的页数（默认 100，约 400KB）

        Returns:
            回收的字节数（近似）

        注意：需要 PRAGMA auto_vacuum=INCREMENTAL 已启用。
        """
        conn = self._get_conn()
        before = self._db_path.stat().st_size if self._db_path.exists() else 0

        try:
            conn.execute(f"PRAGMA incremental_vacuum({pages})")
        except sqlite3.OperationalError:
            # auto_vacuum 未启用，无操作
            return 0

        after = self._db_path.stat().st_size
        return max(0, before - after)

    def optimize(self) -> Dict[str, Any]:
        """数据库性能优化（建议定期调用，例如日终）。

        执行操作：
          1. ANALYZE — 更新查询计划统计
          2. 重建 FTS5 索引 — 清理碎片的全文索引
          3. PRAGMA wal_checkpoint(TRUNCATE) — 截断 WAL 日志

        Returns:
            {"analyzed": bool, "fts_optimized": bool, "wal_checkpointed": bool,
             "size_before": int, "size_after": int}
        """
        conn = self._get_conn()
        before = self._db_path.stat().st_size if self._db_path.exists() else 0

        # 1. ANALYZE
        try:
            conn.execute("ANALYZE episodic_memory")
            analyzed = True
        except sqlite3.Error:
            analyzed = False

        # 2. FTS5 优化
        try:
            conn.execute("INSERT INTO episodic_memory_fts(episodic_memory_fts) VALUES ('optimize')")
            fts_optimized = True
        except sqlite3.Error:
            fts_optimized = False

        # 3. WAL checkpoint
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            wal_checkpointed = True
        except sqlite3.Error:
            wal_checkpointed = False

        after = self._db_path.stat().st_size

        return {
            "analyzed": analyzed,
            "fts_optimized": fts_optimized,
            "wal_checkpointed": wal_checkpointed,
            "size_before": before,
            "size_after": after,
            "freed_bytes": max(0, before - after),
        }

    def db_size(self) -> int:
        """返回数据库文件当前大小（字节）。"""
        if self._db_path.exists():
            return self._db_path.stat().st_size
        return 0

    def db_size_mb(self) -> float:
        """返回数据库文件当前大小（MB）。"""
        return self.db_size() / (1024 * 1024)

    def analyze(self) -> bool:
        """更新 SQLite 查询计划统计（ANALYZE）。

        在大量写入后调用可提升查询性能。
        """
        conn = self._get_conn()
        try:
            conn.execute("ANALYZE")
            return True
        except sqlite3.Error:
            return False

    def checkpoint(self, mode: str = "PASSIVE") -> int:
        """WAL 检查点：将 WAL 日志写入主数据库。

        Args:
            mode: PASSIVE（不阻塞）/ FULL（阻塞直到完成）/ TRUNCATE（截断WAL）

        Returns:
            WAL 页数（busy=0, logged=1, restarted=2）
        """
        conn = self._get_conn()
        row = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return row[2] if row else -1  # restarted pages

    def auto_vacuum_check(self) -> bool:
        """如果数据库超过阈值，自动执行 incremental_vacuum。

        Returns:
            True 表示执行了回收操作
        """
        if not self._auto_vacuum:
            return False

        size = self.db_size()
        if size < self._vacuum_threshold_bytes:
            return False

        freed = self.incremental_vacuum(pages=200)
        return freed > 0

    def get_all_meta(self) -> Dict[str, Dict[str, Any]]:
        """获取所有条目的元数据（兼容旧 API）。"""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, created_at, retrieval_count, correction_count,
                      importance, last_retrieved_at, meta_json
               FROM episodic_memory"""
        ).fetchall()

        result = {}
        for row in rows:
            entry_meta = {
                "created_at": row["created_at"],
                "retrieval_count": row["retrieval_count"],
                "correction_count": row["correction_count"],
                "importance": row["importance"],
                "last_retrieved_at": row["last_retrieved_at"],
            }
            # 合并扩展元数据
            try:
                extra = json.loads(row["meta_json"] or "{}")
                entry_meta.update(extra)
            except json.JSONDecodeError:
                pass
            result[row["id"]] = entry_meta

        return result

    # ── 全文搜索 ──

    def fts_search(self, target: str, query: str,
                   limit: int = 20) -> List[MemoryEntry]:
        """FTS5 全文搜索（SQLite 内建，零网络开销）。

        Args:
            target: 存储目标
            query: 搜索关键词
            limit: 返回上限

        Returns:
            匹配的 MemoryEntry 列表
        """
        conn = self._get_conn()
        # 转义 FTS5 特殊字符
        safe_query = query.replace('"', '""')
        rows = conn.execute(
            """SELECT e.* FROM episodic_memory e
               JOIN episodic_memory_fts fts ON e.rowid = fts.rowid
               WHERE e.target = ? AND episodic_memory_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (target, f'"{safe_query}"', limit),
        ).fetchall()

        return [self._row_to_entry(r) for r in rows]

    # ── 知识图谱 ──

    def add_kg_rule(self, rule: str, confidence: float = 1.0,
                    source: str = "sleep_loop") -> str:
        """添加知识图谱规则。"""
        rule_id = _entry_hash(rule)
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO knowledge_graph (id, rule, confidence, source)
               VALUES (?, ?, ?, ?)""",
            (rule_id, rule, confidence, source),
        )
        conn.commit()
        return rule_id

    def get_kg_rules(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取知识图谱规则。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM knowledge_graph ORDER BY confidence DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def kg_node_count(self) -> int:
        """KG 节点数。"""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM knowledge_graph").fetchone()
        return row["cnt"] if row else 0

    # ── 压缩备份 ──

    def save_compression_backup(self, target: str, entries: List[str]) -> int:
        """保存压缩前备份。返回备份 ID。"""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO compression_backup (target, entries_json) VALUES (?, ?)",
            (target, json.dumps(entries, ensure_ascii=False)),
        )
        conn.commit()
        return cursor.lastrowid

    def get_latest_backup(self, target: str = "memory") -> Optional[Dict[str, Any]]:
        """获取最近的压缩备份。"""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT * FROM compression_backup
               WHERE target = ?
               ORDER BY created_at DESC LIMIT 1""",
            (target,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["entries"] = json.loads(result["entries_json"])
        return result

    def delete_backup(self, backup_id: int) -> bool:
        """删除指定备份。"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM compression_backup WHERE id = ?",
            (backup_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    # ── 批量操作 ──

    def add_batch(self, target: str,
                  entries: List[tuple]) -> List[str]:
        """批量添加条目（单事务）。

        Args:
            target: 存储目标
            entries: [(content, importance), ...]

        Returns:
            成功添加的 entry_id 列表
        """
        conn = self._get_conn()
        now = time.time()
        ids = []

        try:
            for content, importance in entries:
                entry_id = _entry_hash(content.strip())
                conn.execute(
                    """INSERT OR IGNORE INTO episodic_memory
                       (id, target, content, created_at, importance)
                       VALUES (?, ?, ?, ?, ?)""",
                    (entry_id, target, content.strip(), now, importance),
                )
                ids.append(entry_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return ids

    # ── 内部 ──

    def _row_to_entry(self, row) -> MemoryEntry:
        """将 SQLite Row 转换为 MemoryEntry。"""
        return MemoryEntry(
            id=row["id"],
            target=row["target"],
            content=row["content"],
            created_at=row["created_at"],
            retrieval_count=row["retrieval_count"],
            correction_count=row["correction_count"],
            importance=row["importance"],
            last_retrieved_at=row["last_retrieved_at"],
        )
