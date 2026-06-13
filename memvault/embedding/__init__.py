"""
memvault 嵌入服务 — 可插拔架构

  base.py    — AbstractEmbedder 接口
  ollama.py  — Ollama + bge-m3 嵌入
  mock.py    — Mock Embedding（测试/降级）
"""

from memvault.embedding.base import AbstractEmbedder
from memvault.embedding.ollama import OllamaEmbedder
from memvault.embedding.mock import MockEmbedder

__all__ = [
    "AbstractEmbedder",
    "OllamaEmbedder",
    "MockEmbedder",
]
