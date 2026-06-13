"""
MockLLM — Mock LLM 客户端（测试/离线模式）

返回预设响应，用于 CI/CD 测试和离线开发。
不产生网络请求，零成本运行。
"""

from typing import Dict, List, Optional

from memvault.llm.client import LLMResponse


class MockLLM:
    """Mock LLM 客户端。

    两种模式：
      1. 预设响应（preset_responses 列表）— 按顺序返回
      2. 回音模式（echo=True）— 返回输入的前 80 字

    用法:
        # 预设模式
        llm = MockLLM(preset_responses=["规则1", "规则2", "规则3"])
        r1 = llm.call("任意输入")  # → "规则1"
        r2 = llm.call("任意输入")  # → "规则2"

        # 回音模式
        llm = MockLLM(echo=True)
        r = llm.call("测试文本")  # → "Mock echo: 测试文本"
    """

    def __init__(
        self,
        preset_responses: Optional[List[str]] = None,
        echo: bool = False,
        echo_prefix: str = "Mock echo: ",
        latency_ms: int = 5,
    ):
        """
        Args:
            preset_responses: 预设响应列表（按调用顺序返回）
            echo: 启用回音模式（返回截断的输入文本）
            echo_prefix: 回音前缀
            latency_ms: 模拟延迟（毫秒）
        """
        self._preset = preset_responses or []
        self._index = 0
        self._echo = echo
        self._echo_prefix = echo_prefix
        self._latency_ms = latency_ms
        self.provider = "mock"
        self.model = "mock-llm"
        self.api_key = "mock-key"
        self.call_count = 0

    def call(self, prompt: str,
             system: Optional[str] = None,
             max_tokens: Optional[int] = None,
             temperature: float = 0.7) -> LLMResponse:
        """返回预设响应或回音。

        Args:
            prompt: 用户消息（回音模式使用）
            system: 系统消息（Mock 忽略）
            max_tokens: Mock 忽略
            temperature: Mock 忽略

        Returns:
            LLMResponse
        """
        self.call_count += 1

        # 预设响应模式
        if self._preset and self._index < len(self._preset):
            text = self._preset[self._index]
            self._index += 1
            return LLMResponse(
                success=True,
                text=text,
                latency_ms=self._latency_ms,
                provider="mock",
                model="mock-llm",
            )

        # 回音模式
        if self._echo:
            text = self._echo_prefix + prompt[:80]
            return LLMResponse(
                success=True,
                text=text,
                latency_ms=self._latency_ms,
                provider="mock",
                model="mock-llm",
            )

        # 默认：返回固定文本
        return LLMResponse(
            success=True,
            text="mock: 根据输入提取的规则（Mock 默认响应）",
            latency_ms=self._latency_ms,
            provider="mock",
            model="mock-llm",
        )

    def test_connection(self) -> LLMResponse:
        """Mock 连接始终成功。"""
        return LLMResponse(
            success=True,
            text="OK",
            latency_ms=1,
            provider="mock",
            model="mock-llm",
        )

    def reset(self):
        """重置预设响应索引和计数。"""
        self._index = 0
        self.call_count = 0

    def set_responses(self, responses: List[str]):
        """动态设置预设响应。"""
        self._preset = responses
        self._index = 0
