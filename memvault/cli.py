"""
memvault CLI — 命令行入口

用法：
    memvault add "这是一条记忆"              # 添加记忆
    memvault add "用户喜欢 Python" --user    # 添加用户画像
    memvault recall "关键词"                 # 搜索记忆
    memvault stats                           # 统计信息
    memvault consolidate                     # 手动触发巩固
    memvault config                          # 查看当前配置
    memvault init                            # 生成默认配置文件
"""

import argparse
import json
import sys
from pathlib import Path

from memvault.config import MemoryVaultConfig, load_config
from memvault import MemoryVault


def get_vault(args) -> MemoryVault:
    """从命令行参数创建 MemoryVault。"""
    config = None

    if hasattr(args, 'config_file') and args.config_file:
        config = load_config(args.config_file)

    if config is None:
        config = MemoryVaultConfig()

    # 覆盖存储路径
    if hasattr(args, 'storage') and args.storage:
        config.storage.path = args.storage
        config.storage.backend = "sqlite"

    return MemoryVault(config=config)


def cmd_add(args):
    vault = get_vault(args)
    target = "user" if args.user else "memory"
    result = vault.remember(args.content, target=target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_recall(args):
    vault = get_vault(args)
    results = vault.recall(args.query, top_k=args.top_k)

    if not results:
        print("(未找到相关记忆)")
        return

    for i, entry in enumerate(results, 1):
        imp = f"[importance: {entry.importance:.2f}]"
        print(f"{i}. [{entry.weight:.3f}] {imp}")
        print(f"   {entry.content[:120]}")
        print()


def cmd_stats(args):
    vault = get_vault(args)
    s = vault.stats()

    print(f"存储后端: {vault.config.storage.backend}")
    print(f"MEMORY:   {s.memory_entries} 条, {s.memory_chars:,} chars / {s.memory_limit:,} limit")
    print(f"USER:     {s.user_entries} 条, {s.user_chars:,} chars / {s.user_limit:,} limit")
    print(f"KG 节点:  {s.kg_nodes}")
    print(f"平均权重:  {s.avg_weight:.3f}")
    print(f"平均重要性: {s.avg_importance:.3f}")
    print(f"半衰期:    {vault.config.memory.half_life_days} 天")


def cmd_consolidate(args):
    vault = get_vault(args)
    result = vault.consolidate(dry_run=args.dry_run)

    print(f"🌙 Sleep Loop 完成")
    print(f"   扫描: {result.scanned} 条")
    print(f"   聚类: {result.clusters_found} 个簇")
    print(f"   规则: {result.rules_extracted} 条")
    print(f"   KG 新增: {result.kg_nodes_added}")
    print(f"   耗时: {result.elapsed_seconds:.1f}s")

    if result.rules:
        print("\n提炼的规则:")
        for i, r in enumerate(result.rules):
            conf = result.confidence_scores[i] if i < len(result.confidence_scores) else 0
            print(f"   [{conf:.2f}] {r}")


def cmd_config(args):
    vault = get_vault(args)
    config_dict = vault.config.to_dict()

    print(json.dumps(config_dict, ensure_ascii=False, indent=2))


def cmd_init(args):
    """生成默认配置文件。"""
    path = args.output or "memvault.yaml"
    if Path(path).exists() and not args.force:
        print(f"配置文件已存在: {path}")
        print("使用 --force 强制覆盖")
        sys.exit(1)

    config = MemoryVaultConfig()
    yaml_str = config.to_yaml(path)
    print(f"配置文件已生成: {path}")
    print()
    print(yaml_str)


def main():
    parser = argparse.ArgumentParser(
        description="memvault — 认知记忆系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  memvault add "修复了一个并发 bug"
  memvault recall "bug"
  memvault stats
  memvault consolidate --dry-run
  memvault init
""",
    )
    parser.add_argument("--storage", help="数据库路径 (SQLite)")
    parser.add_argument("--config-file", help="配置文件路径 (YAML)")

    sub = parser.add_subparsers(dest="command", help="子命令")

    # add
    p_add = sub.add_parser("add", help="添加记忆")
    p_add.add_argument("content", help="记忆内容")
    p_add.add_argument("--user", action="store_true", help="添加到用户画像")
    p_add.set_defaults(func=cmd_add)

    # recall
    p_search = sub.add_parser("recall", help="搜索记忆")
    p_search.add_argument("query", help="搜索关键词")
    p_search.add_argument("--top-k", type=int, default=10, help="返回条数")
    p_search.set_defaults(func=cmd_recall)

    # stats
    p_stats = sub.add_parser("stats", help="统计信息")
    p_stats.set_defaults(func=cmd_stats)

    # consolidate
    p_sleep = sub.add_parser("consolidate", help="手动触发 Sleep Loop")
    p_sleep.add_argument("--dry-run", action="store_true", help="只预览不写入")
    p_sleep.set_defaults(func=cmd_consolidate)

    # config
    p_conf = sub.add_parser("config", help="查看当前配置")
    p_conf.set_defaults(func=cmd_config)

    # init
    p_init = sub.add_parser("init", help="生成默认配置文件")
    p_init.add_argument("--output", "-o", default="memvault.yaml", help="输出路径")
    p_init.add_argument("--force", "-f", action="store_true", help="强制覆盖")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
