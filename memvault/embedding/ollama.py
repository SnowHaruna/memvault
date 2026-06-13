"""
OllamaEmbedder — Ollama 本地嵌入后端

默认使用 bge-m3 模型，通过 Ollama REST API 调用。
"""

import urllib.request
from typing import List

from memvault.embedding.base import AbstractEmbedder


class OllamaEmbedder(AbstractEmbedder):
    """Ollama 本地嵌入后端。

    默认模型: bge-m3 (1024d 稠密向量)

    Args:
        model: 模型名
        base_url: Ollama API 地址
        timeout: 请求超时（秒）
    """

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 2.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> bool:
        """检查 Ollama 服务是否可用。"""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url)
            urllib.request.urlopen(req, timeout=self.timeout)
            return True
        except Exception:
            return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入（使用 LlamaIndex Ollama 集成）。"""
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding

            embed_model = OllamaEmbedding(
                model_name=self.model,
                base_url=self.base_url,
                embed_batch_size=16,
            )
            return [embed_model.get_text_embedding(t) for t in texts]
        except ImportError:
            # 回退：纯 HTTP 调用
            return self._embed_http(texts)

    def embed_query(self, query: str) -> List[float]:
        """单条查询嵌入。"""
        results = self.embed([query])
        return results[0] if results else []

    def _embed_http(self, texts: List[str]) -> List[List[float]]:
        """纯 HTTP 嵌入（无需 LlamaIndex 依赖）。"""
        import json

        results = []
        for text in texts:
            body = json.dumps({
                "model": self.model,
                "prompt": text,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=self.timeout * 5) as resp:
                data = json.loads(resp.read().decode())
                results.append(data.get("embedding", []))

        return results
