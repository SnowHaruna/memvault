"""
ColbertMaxSimReranker — ColBERT-lite 重排序器

Token 级 MaxSim 重排序，模拟 ColBERT 的 late interaction。
使用字符 bigram 的 Jaccard 相似度作为 token 级相似度的近似。

纯 Python 实现，零额外模型依赖。
"""

from typing import List, Tuple

from memvault.retrieval.sparse import char_bigrams


class ColbertMaxSimReranker:
    """ColBERT-lite Token MaxSim 重排序器。

    对候选条目做细粒度 token 对齐，模拟 ColBERT 的 late interaction。
    使用字符 bigram 的 Jaccard 相似度作为 token 级相似度的近似。

    管线位置：
      Dense + Sparse → RRF Fusion → ColBERT MaxSim → Weight Rerank
    """

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 10,
    ) -> List[Tuple[float, str]]:
        """ColBERT MaxSim 重排序。

        Args:
            query: 查询文本
            candidates: 候选条目列表（通常为 RRF 融合后的 top-N）
            top_k: 返回条数

        Returns:
            [(maxsim_score, text), ...] 按分数降序
        """
        if not candidates:
            return []

        query_tokens = char_bigrams(query)
        if not query_tokens:
            return [(0.0, c) for c in candidates[:top_k]]

        results = []
        for text in candidates:
            doc_tokens = char_bigrams(text)
            if not doc_tokens:
                results.append((0.0, text))
                continue

            # MaxSim: 对每个 query token，找最相似的 doc token
            total_sim = 0.0
            for qt in query_tokens:
                max_sim = 0.0
                for dt in doc_tokens:
                    sim = self._token_similarity(qt, dt)
                    if sim > max_sim:
                        max_sim = sim
                total_sim += max_sim

            # 归一化（除以 query token 数）
            score = total_sim / max(1, len(query_tokens))
            results.append((score, text))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        """两个 token 的相似度。

        精确匹配 = 1.0，否则用字符级 Jaccard 重叠率。
        """
        if a == b:
            return 1.0

        set_a, set_b = set(a), set(b)
        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)
