"""
memvault LLM 集成 — 统一客户端 + Mock

  client.py — 统一 LLM 客户端（OpenAI / Anthropic / DeepSeek）
  mock.py   — Mock LLM（CI / 测试 / 离线模式）
"""

from memvault.llm.client import LLMClient, LLMResponse
from memvault.llm.mock import MockLLM

__all__ = [
    "LLMClient",
    "LLMResponse",
    "MockLLM",
]
