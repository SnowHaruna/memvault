"""
最小可用示例 — memvault 3 行代码跑起来
"""

from memvault import MemoryVault

# 初始化（使用内存后端，无需数据库文件）
vault = MemoryVault()

# 添加记忆
vault.remember("今天修复了一个并发 bug，根因是数据库连接池未设置 max_overflow")
vault.remember("用户反馈登录页面在 Safari 上白屏，可能是 WebKit 兼容问题")
vault.remember("Python 的 asyncio 在处理大量并发连接时非常高效")

# 搜索
print("=== 搜索 'bug' ===")
results = vault.recall("bug")
for r in results:
    print(f"  [{r.weight:.2f}] {r.content}")

print("\n=== 搜索 'Python' ===")
results = vault.recall("Python")
for r in results:
    print(f"  [{r.weight:.2f}] {r.content}")

# 获取上下文
print("\n=== 记忆上下文 ===")
print(vault.context())

# 统计
stats = vault.stats()
print(f"\n=== 统计 ===")
print(f"记忆条目: {stats.memory_entries}")
print(f"平均权重: {stats.avg_weight:.3f}")
print(f"字符使用: {stats.memory_chars:,}/{stats.memory_limit:,}")
