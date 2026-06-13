"""
memvault 统一 LLM 客户端

支持三种协议：
  - OpenAI (Chat Completions)
  - Anthropic (Messages API)
  - DeepSeek (Anthropic-compatible Messages API)

处理 DeepSeek thinking block → text 回退。
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    """LLM 调用结果。

    Attributes:
        success: 调用是否成功
        text: 回复文本
        latency_ms: 延迟毫秒
        provider: 使用的 provider
        model: 使用的模型
        url: 实际请求 URL
        error: 错误简述（失败时）
        raw_error: 原始错误体（失败时）
    """
    success: bool = False
    text: str = ""
    latency_ms: int = 0
    provider: str = ""
    model: str = ""
    url: str = ""
    error: str = ""
    raw_error: str = ""


class LLMClient:
    """统一 LLM 客户端。

    用法:
        client = LLMClient(
            provider="deepseek",
            model="deepseek-v4-flash",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
        result = client.call("你好")
        if result.success:
            print(result.text)
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-v4-flash",
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 1024,
        timeout: int = 60,
    ):
        """
        Args:
            provider: "openai" | "anthropic" | "deepseek"
            model: 模型名
            api_key: API 密钥
            base_url: API 地址（留空使用默认）
            max_tokens: 最大输出 token
            timeout: HTTP 超时（秒）
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

    def call(self, prompt: str,
             system: Optional[str] = None,
             max_tokens: Optional[int] = None,
             temperature: float = 0.7) -> LLMResponse:
        """调用 LLM。

        Args:
            prompt: 用户消息
            system: 系统消息（可选）
            max_tokens: 覆盖默认 max_tokens
            temperature: 温度参数

        Returns:
            LLMResponse
        """
        if not self.api_key:
            return LLMResponse(
                success=False,
                error="未配置 API Key",
                provider=self.provider,
                model=self.model,
            )

        t0 = time.time()
        url = ""
        mt = max_tokens or self.max_tokens

        try:
            if self.provider == "openai":
                url, body, headers = self._build_openai_request(
                    prompt, system, mt, temperature
                )
            else:
                url, body, headers = self._build_anthropic_request(
                    prompt, system, mt, temperature
                )

            req = urllib.request.Request(url, data=body, headers=headers)
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            latency = round((time.time() - t0) * 1000)
            data = json.loads(resp.read().decode())

            text = self._extract_text(data)
            return LLMResponse(
                success=True,
                text=text,
                latency_ms=latency,
                provider=self.provider,
                model=self.model,
                url=url,
            )

        except urllib.error.HTTPError as e:
            latency = round((time.time() - t0) * 1000)
            err_body = e.read().decode(errors="replace")[:500] if e.fp else ""
            return LLMResponse(
                success=False,
                latency_ms=latency,
                provider=self.provider,
                model=self.model,
                url=url,
                error=f"HTTP {e.code}",
                raw_error=err_body,
            )

        except Exception as e:
            latency = round((time.time() - t0) * 1000)
            return LLMResponse(
                success=False,
                latency_ms=latency,
                provider=self.provider,
                model=self.model,
                url=url,
                error=str(e)[:300],
            )

    def test_connection(self) -> LLMResponse:
        """测试 LLM 连接（发送简单 ping）。"""
        return self.call("请只回复'OK'两个字，不要添加任何其他内容。", max_tokens=10)

    # ── 请求构建 ──

    def _build_openai_request(self, prompt: str, system: Optional[str],
                               max_tokens: int, temperature: float):
        """构建 OpenAI 协议请求。"""
        base = self.base_url or "https://api.openai.com/v1"
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return url, body, headers

    def _build_anthropic_request(self, prompt: str, system: Optional[str],
                                  max_tokens: int, temperature: float):
        """构建 Anthropic 协议请求（含 DeepSeek）。"""
        if self.provider == "deepseek":
            base = self.base_url or "https://api.deepseek.com/anthropic"
        else:
            base = self.base_url or "https://api.anthropic.com"

        url = base if base.endswith("/messages") else f"{base}/v1/messages"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }).encode("utf-8")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        return url, body, headers

    # ── 响应解析 ──

    def _extract_text(self, data: dict) -> str:
        """从 API 响应提取文本，处理 thinking block 回退。"""
        if self.provider == "openai":
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""

        # Anthropic / DeepSeek 协议
        content = data.get("content", [])
        if not isinstance(content, list):
            return ""

        text = ""
        thinking = ""
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                t = block.get("text", "")
                if t.strip():
                    text = t.strip()
                    break  # 找到有效 text 就停
            elif block.get("type") == "thinking":
                thinking = block.get("thinking", "")

        if text:
            return text
        if thinking.strip():
            return thinking.strip()
        return ""
