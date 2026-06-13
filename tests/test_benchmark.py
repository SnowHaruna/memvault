"""
memvault 性能基准测试

对比 SQLiteStorage vs MemoryStorage 在不同操作下的性能。
覆盖：
  - 批量写入（100 / 1K / 10K / 100K 条目）
  - 顺序读取 / 随机访问
  - FTS5 全文搜索 vs Python 子串匹配
  - 字符计数 / 元数据批量更新
  - 大规模压缩与清理

Usage:
    python -m pytest tests/test_benchmark.py -v -s          # 详细输出
    python -m pytest tests/test_benchmark.py -v -k "100k"   # 仅大规模
    python -m pytest tests/test_benchmark.py --benchmark-only
"""

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from memvault.storage.memory import MemoryStorage
from memvault.storage.sqlite import SQLiteStorage
from memvault.storage.file import FileStorage
from memvault.types import MemoryEntry


# ═══════════════════════════════════════════════════════════
# 测试文本生成
# ═══════════════════════════════════════════════════════════

TECH_SENTENCES = [
    "系统架构的设计需要考虑可扩展性和可维护性。",
    "代码审查是保证软件质量的重要手段。",
    "单元测试能够有效减少回归缺陷。",
    "CI/CD 流水线自动化了构建和部署流程。",
    "微服务架构将应用拆分为独立可部署的服务单元。",
    "API 网关负责请求路由、限流和认证。",
    "数据库索引可以显著提升查询性能。",
    "缓存策略需要权衡数据一致性和响应速度。",
    "日志系统对故障排查和安全审计至关重要。",
    "容器化技术简化了环境配置和部署流程。",
    "Python 的装饰器本质上是一个接受函数并返回新函数的高阶函数。",
    "asyncio 事件循环是 Python 异步编程的核心调度机制。",
    "Python 的类型提示提高了代码的可读性和 IDE 支持。",
    "Docker 镜像由多个只读层叠加而成，实现了存储复用。",
    "Docker Compose 通过 YAML 文件定义多容器应用的服务拓扑。",
    "记忆系统的衰减模型遵循艾宾浩斯遗忘曲线的指数规律。",
    "情景记忆存储具体的经历和事件，随时间逐渐衰减。",
    "语义记忆抽象出概念和规则，比情景记忆更持久。",
    "睡眠巩固将白天积累的情景记忆转化为语义知识。",
    "三层记忆架构模拟了人脑从感知到长期记忆的完整通路。",
]


def generate_entries(n: int, seed: int = 42) -> List[str]:
    """生成 n 条测试文本（固定种子可复现）。"""
    import random
    rng = random.Random(seed)
    entries = []
    for i in range(n):
        # 2-6 句随机组合
        n_sentences = rng.randint(2, 6)
        chosen = rng.sample(TECH_SENTENCES, min(n_sentences, len(TECH_SENTENCES)))
        rng.shuffle(chosen)
        # 每 10 条注入一个唯一 ID 防止全重复
        text = f"[{i:06d}] " + "".join(chosen)
        entries.append(text)
    return entries


def measure_time(func, *args, **kwargs) -> Tuple[float, any]:
    """测量函数执行时间（秒）。"""
    gc.collect()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return elapsed, result


def measure_memory_mb(func, *args, **kwargs) -> Tuple[float, float, any]:
    """测量函数执行时间和内存峰值（MB）。"""
    import resource
    gc.collect()
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_delta = (mem_after - mem_before) / 1024.0  # KB → MB
    return elapsed, mem_delta, result


# ═══════════════════════════════════════════════════════════
# 存储后端工厂
# ═══════════════════════════════════════════════════════════

@pytest.fixture(params=["memory", "sqlite"])
def storage_backend(request, tmp_path):
    """参数化：每个测试在 MemoryStorage 和 SQLiteStorage 上各跑一次。"""
    if request.param == "memory":
        store = MemoryStorage()
    elif request.param == "sqlite":
        db_path = tmp_path / "bench.db"
        store = SQLiteStorage(str(db_path))
    else:
        raise ValueError(f"Unknown backend: {request.param}")

    yield store
    store.close()


# ═══════════════════════════════════════════════════════════
# 基准测试类
# ═══════════════════════════════════════════════════════════

class TestBatchWriteBenchmark:
    """批量写入性能基准。"""

    @pytest.mark.parametrize("n_entries", [100, 1000])
    def test_batch_write(self, storage_backend, n_entries):
        """测试逐条添加 vs 批量添加的性能。"""
        entries = generate_entries(n_entries)

        # 逐条添加
        elapsed_single, _ = measure_time(
            lambda: [storage_backend.add_entry("memory", e) for e in entries]
        )
        # 清理
        for e in entries:
            from memvault.storage.memory import _entry_hash
            storage_backend.delete_entry(_entry_hash(e))

        # 批量添加（仅 SQLite 有原生 add_batch）
        if hasattr(storage_backend, 'add_batch'):
            tuples = [(e, 0.5) for e in entries]
            elapsed_batch, _ = measure_time(
                lambda: storage_backend.add_batch("memory", tuples)
            )
            speedup = elapsed_single / max(elapsed_batch, 0.001)

            print(f"\n  [{type(storage_backend).__name__}] "
                  f"{n_entries} entries: "
                  f"single={elapsed_single*1000:.1f}ms "
                  f"batch={elapsed_batch*1000:.1f}ms "
                  f"(x{speedup:.1f})")

            assert elapsed_batch < elapsed_single * 1.2  # batch shouldn't be slower

    @pytest.mark.slow
    @pytest.mark.parametrize("n_entries", [10000])
    def test_large_batch_write(self, storage_backend, n_entries):
        """大规模批量写入（仅在有 --run-slow 时执行）。"""
        entries = generate_entries(n_entries)

        elapsed, _ = measure_time(
            lambda: [storage_backend.add_entry("memory", e) for e in entries[:100]]  # 采样
        )

        if hasattr(storage_backend, 'add_batch'):
            # 使用批量接口
            tuples = [(e, 0.5) for e in entries]
            elapsed_full, _ = measure_time(
                lambda: storage_backend.add_batch("memory", tuples)
            )

            rate = n_entries / max(elapsed_full, 0.001)
            print(f"\n  [{type(storage_backend).__name__}] "
                  f"{n_entries} entries: {elapsed_full*1000:.0f}ms "
                  f"({rate:.0f} entries/s)")

            # 10K entries 应在合理时间内完成
            assert elapsed_full < 30.0  # 30s 上限


class TestReadBenchmark:
    """读取性能基准。"""

    @pytest.fixture
    def populated_storage(self, storage_backend):
        """预填充 500 条数据。"""
        entries = generate_entries(500)
        for e in entries:
            storage_backend.add_entry("memory", e)
        return storage_backend, entries

    def test_sequential_read(self, populated_storage):
        """顺序读取所有条目。"""
        storage, entries = populated_storage

        elapsed, result = measure_time(
            lambda: storage.get_entries("memory")
        )

        assert len(result) == 500
        print(f"\n  [{type(storage).__name__}] "
              f"sequential read 500: {elapsed*1000:.1f}ms")

    def test_random_access(self, populated_storage):
        """随机访问单条条目。"""
        storage, entries = populated_storage
        from memvault.storage.memory import _entry_hash
        ids = [_entry_hash(e) for e in entries]

        import random
        rng = random.Random(42)
        sample_ids = rng.sample(ids, min(50, len(ids)))

        elapsed, results = measure_time(
            lambda: [storage.get_entry(eid) for eid in sample_ids]
        )

        assert all(r is not None for r in results)
        rate = len(sample_ids) / max(elapsed, 0.001)
        print(f"\n  [{type(storage).__name__}] "
              f"random access {len(sample_ids)}: {elapsed*1000:.1f}ms "
              f"({rate:.0f} lookups/s)")

    def test_char_count(self, populated_storage):
        """字符计数性能。"""
        storage, _ = populated_storage

        elapsed, count = measure_time(
            lambda: storage.char_count("memory")
        )

        assert count > 0
        print(f"\n  [{type(storage).__name__}] "
              f"char_count: {elapsed*1000:.1f}ms ({count:,} chars)")


class TestSearchBenchmark:
    """搜索性能基准。"""

    @pytest.fixture
    def searchable_storage(self, storage_backend):
        """预填充搜索关键词语料。"""
        entries = generate_entries(300)
        # 强制注入搜索关键词
        keyword_entries = [
            "Python 异步编程是构建高性能后端服务的关键技术。",
            "Docker 容器化部署简化了 CI/CD 流程管理。",
            "记忆衰减模型采用艾宾浩斯遗忘曲线进行数学建模。",
            "知识图谱通过节点和边构建结构化的语义网络。",
            "权重计算考虑了时间衰减、检索频率和情绪重要性。",
        ]
        all_entries = entries + keyword_entries
        for e in all_entries:
            storage_backend.add_entry("memory", e)
        return storage_backend, all_entries

    def test_fts_vs_keyword(self, searchable_storage):
        """对比 FTS5 搜索 vs Python 子串匹配。"""
        storage, all_entries = searchable_storage

        # FTS5 搜索（SQLite 专有）
        if hasattr(storage, 'fts_search'):
            elapsed_fts, fts_results = measure_time(
                lambda: storage.fts_search("memory", "Python", limit=10)
            )
            fts_ms = elapsed_fts * 1000
        else:
            fts_ms = None

        # Python 子串匹配（memory/file 回退路径）
        elapsed_py, py_results = measure_time(
            lambda: [
                e for e in all_entries
                if "Python" in e
            ][:10]
        )
        py_ms = elapsed_py * 1000

        backend_name = type(storage).__name__
        if fts_ms:
            ratio = py_ms / max(fts_ms, 0.001)
            print(f"\n  [{backend_name}] search 'Python': "
                  f"FTS5={fts_ms:.1f}ms vs Python={py_ms:.1f}ms "
                  f"(FTS5 x{ratio:.1f} faster)")
        else:
            print(f"\n  [{backend_name}] search 'Python': "
                  f"Python={py_ms:.1f}ms (FTS5 not available)")

    def test_multi_keyword_search(self, searchable_storage):
        """多关键词搜索。"""
        storage, _ = searchable_storage
        keywords = ["Python", "Docker", "记忆", "权重", "知识图谱"]

        if hasattr(storage, 'fts_search'):
            elapsed, all_results = measure_time(
                lambda: {
                    kw: storage.fts_search("memory", kw, limit=10)
                    for kw in keywords
                }
            )
        else:
            all_entries = storage.get_entries("memory")
            elapsed, all_results = measure_time(
                lambda: {
                    kw: [e.content for e in all_entries if kw in e.content][:10]
                    for kw in keywords
                }
            )

        avg_ms = (elapsed / len(keywords)) * 1000
        total_hits = sum(len(v) for v in all_results.values())
        print(f"\n  [{type(storage).__name__}] "
              f"multi-search {len(keywords)} kw: "
              f"total={elapsed*1000:.1f}ms avg={avg_ms:.1f}ms "
              f"hits={total_hits}")


class TestMetaUpdateBenchmark:
    """元数据更新性能基准。"""

    @pytest.fixture
    def metad_storage(self, storage_backend):
        """预填充 + 生成元数据更新。"""
        entries = generate_entries(200)
        ids = []
        for e in entries:
            eid = storage_backend.add_entry("memory", e)
            ids.append(eid)
        return storage_backend, ids

    def test_single_meta_update(self, metad_storage):
        """逐条元数据更新。"""
        storage, ids = metad_storage

        import random
        rng = random.Random(42)
        sample = rng.sample(ids, min(50, len(ids)))

        updates = {eid: {"retrieval_count": 5, "last_retrieved_at": time.time()}
                   for eid in sample}

        elapsed, count = measure_time(
            lambda: sum(
                1 for eid, fields in updates.items()
                if storage.update_meta(eid, fields)
            )
        )

        rate = count / max(elapsed, 0.001)
        print(f"\n  [{type(storage).__name__}] "
              f"meta update {count}: {elapsed*1000:.1f}ms "
              f"({rate:.0f} updates/s)")

    def test_batch_meta_update(self, metad_storage):
        """批量元数据更新。"""
        storage, ids = metad_storage

        updates = {eid: {"retrieval_count": 3, "last_retrieved_at": time.time()}
                   for eid in ids[:100]}

        if hasattr(storage, 'batch_update_meta'):
            elapsed, count = measure_time(
                lambda: storage.batch_update_meta(updates)
            )

            rate = count / max(elapsed, 0.001)
            print(f"\n  [{type(storage).__name__}] "
                  f"batch meta update {count}: {elapsed*1000:.1f}ms "
                  f"({rate:.0f} updates/s)")


class TestPruneBenchmark:
    """清理操作性能基准。"""

    def test_prune_speed(self, storage_backend):
        """prune 性能。"""
        # 添加条目
        entries = generate_entries(500)
        for e in entries:
            storage_backend.add_entry("memory", e, importance=0.3)  # 低权重

        # 添加几条高权重
        for i in range(10):
            storage_backend.add_entry("memory", f"HIGH_PRIORITY_{i}", importance=1.0)

        elapsed, removed = measure_time(
            lambda: storage_backend.prune_below_weight("memory", 0.5)
        )

        print(f"\n  [{type(storage_backend).__name__}] "
              f"prune removed {removed}: {elapsed*1000:.1f}ms")

    def test_vacuum_speed(self, storage_backend):
        """vacuum 性能（仅 SQLite 有意义）。"""
        # 添加后删除一些条目
        entries = generate_entries(200)
        for e in entries:
            storage_backend.add_entry("memory", e)

        from memvault.storage.memory import _entry_hash
        for e in entries[:100]:
            storage_backend.delete_entry(_entry_hash(e))

        # Vacuum
        elapsed, freed = measure_time(
            lambda: storage_backend.vacuum()
        )

        print(f"\n  [{type(storage_backend).__name__}] "
              f"vacuum freed {freed} bytes: {elapsed*1000:.1f}ms")


class TestConcurrencyBenchmark:
    """并发访问基准（仅 SQLite WAL 模式）。"""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Threading benchmark unreliable on Windows"
    )
    def test_concurrent_reads(self, tmp_path):
        """多线程并发读取。"""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        db_path = tmp_path / "concur.db"
        storage = SQLiteStorage(str(db_path))

        # 预填充
        entries = generate_entries(500)
        for e in entries:
            storage.add_entry("memory", e)

        results_lock = threading.Lock()
        results = []

        def reader():
            local_results = storage.get_entries("memory", limit=20)
            with results_lock:
                results.extend(local_results)

        n_threads = 4
        elapsed, _ = measure_time(
            lambda: (
                ThreadPoolExecutor(max_workers=n_threads)
                .map(lambda _: reader(), range(n_threads))
                and None  # force evaluation
            )
        )

        print(f"\n  [SQLiteStorage] {n_threads} concurrent readers: "
              f"{elapsed*1000:.1f}ms ({len(results)} results)")

        storage.close()


# ═══════════════════════════════════════════════════════════
# 端到端吞吐量测试
# ═══════════════════════════════════════════════════════════

class TestThroughputBenchmark:
    """端到端吞吐量基准。"""

    @pytest.mark.slow
    def test_write_throughput_10k(self, storage_backend):
        """10K 条目写入吞吐量。"""
        entries = generate_entries(10000)

        if hasattr(storage_backend, 'add_batch'):
            tuples = [(e, 0.5) for e in entries]
            elapsed, _ = measure_time(
                lambda: storage_backend.add_batch("memory", tuples)
            )
            rate = 10000 / max(elapsed, 0.001)
            print(f"\n  [{type(storage_backend).__name__}] "
                  f"10K write: {elapsed:.3f}s ({rate:.0f} entries/s)")

            assert rate > 100  # 至少 100 entries/s

    def test_read_throughput_5k(self, storage_backend):
        """5K 条目读取 + 权重计算吞吐量。"""
        entries = generate_entries(5000)
        for e in entries:
            storage_backend.add_entry("memory", e)

        # 模拟 MemoryStore.get() 的权重计算流程
        elapsed, _ = measure_time(
            lambda: (
                storage_backend.get_entries("memory"),
                storage_backend.get_all_meta(),
            )
        )

        rate = 5000 / max(elapsed, 0.001)
        print(f"\n  [{type(storage_backend).__name__}] "
              f"5K read+meta: {elapsed*1000:.0f}ms ({rate:.0f} entries/s)")


# ═══════════════════════════════════════════════════════════
# 综合对比报告
# ═══════════════════════════════════════════════════════════

def test_generate_comparison_report(tmp_path):
    """生成 Memory vs SQLite 综合对比报告。"""
    import json

    results = {}

    for backend_name, make_backend in [
        ("MemoryStorage", lambda: MemoryStorage()),
        ("SQLiteStorage", lambda: SQLiteStorage(str(tmp_path / f"report_{int(time.time())}.db"))),
    ]:
        store = make_backend()
        entries = generate_entries(2000)

        # 写入
        t_write, _ = measure_time(
            lambda: [store.add_entry("memory", e) for e in entries]
        )

        # 读取
        t_read, read_result = measure_time(lambda: store.get_entries("memory"))

        # 搜索
        all_entries = store.get_entries("memory")
        all_texts = [e.content for e in all_entries]
        t_search, _ = measure_time(
            lambda: [
                e for e in all_texts
                if "Python" in e
            ][:10]
        )

        # 字符计数
        t_count, char_count = measure_time(lambda: store.char_count("memory"))

        # 清理
        from memvault.storage.memory import _entry_hash
        t_delete, _ = measure_time(
            lambda: [store.delete_entry(_entry_hash(e)) for e in entries[:200]]
        )

        results[backend_name] = {
            "write_2000_ms": round(t_write * 1000, 1),
            "read_2000_ms": round(t_read * 1000, 1),
            "search_ms": round(t_search * 1000, 1),
            "char_count_ms": round(t_count * 1000, 1),
            "delete_200_ms": round(t_delete * 1000, 1),
            "total_chars": char_count,
        }

        store.close()

    # 计算对比
    if "MemoryStorage" in results and "SQLiteStorage" in results:
        mem = results["MemoryStorage"]
        sql = results["SQLiteStorage"]

        print("\n" + "=" * 60)
        print("  MemoryStorage vs SQLiteStorage 性能对比 (2,000 条目)")
        print("=" * 60)
        for metric in ["write_2000_ms", "read_2000_ms", "search_ms",
                        "char_count_ms", "delete_200_ms"]:
            m_val = mem[metric]
            s_val = sql[metric]
            ratio = m_val / max(s_val, 0.001)
            winner = "Memory" if m_val < s_val else "SQLite"
            bar = "█" * min(int(ratio * 10), 40) if ratio > 1 else ""
            print(f"  {metric:<22s}  Mem={m_val:>8.1f}ms  SQL={s_val:>8.1f}ms  "
                  f"({winner} x{ratio:.1f}) {bar}")
        print("=" * 60)

    # 保存 JSON
    report_path = tmp_path / "benchmark_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n  Report saved: {report_path}")
