"""
memvault 核心引擎

  decay.py      — Ebbinghaus 权重计算（纯函数，可替换）
  formatter.py  — 记忆格式化（Markdown / JSON / 自定义）
  engine.py     — MemoryStore（存储协调层）
  compressor.py — 首因+近因压缩（含 LLM 回退）
  consolidator.py — Sleep Loop 巩固循环
"""

from memvault.core.decay import compute_weight, compute_importance, score_importance
from memvault.core.formatter import (
    format_memory_block,
    format_memory_block_simple,
    format_kg_block,
    format_system_prompt_block,
)
from memvault.core.engine import MemoryStore

__all__ = [
    "compute_weight",
    "compute_importance",
    "score_importance",
    "format_memory_block",
    "format_memory_block_simple",
    "format_kg_block",
    "format_system_prompt_block",
    "MemoryStore",
]
