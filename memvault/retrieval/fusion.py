"""
检索融合 — RRF + 加权融合

融合多路检索结果：
  - RRF (Reciprocal Rank Fusion): 无权重调参，自动平衡各路贡献
  - Weighted Fusion: 允许手动指定各路权重
"""

from typing import Any, Dict, List, Tuple


def rrf_fuse(
    *ranked_lists: List[Tuple[float, Any]],
    k: int = 60,
) -> List[Tuple[float, Any]]:
    """Reciprocal Rank Fusion — 无权重融合多路检索结果。

    公式: RRF_score(d) = Σ 1/(k + rank_i(d))

    Args:
        ranked_lists: 各路排序结果 [(score, item), ...]
        k: RRF 平滑参数（默认 60，控制排名的影响力）

    Returns:
        [(fused_score, item), ...] 按融合分数降序

    Example:
        dense_results = [(0.9, "text_a"), (0.7, "text_b")]
        sparse_results = [(0.8, "text_b"), (0.6, "text_c")]
        fused = rrf_fuse(dense_results, sparse_results, k=60)
    """
    scores: Dict[str, Tuple[float, Any]] = {}

    for lst in ranked_lists:
        for rank, (_, item) in enumerate(lst):
            # 使用字符串 key
            key = _item_key(item)
            rrf_score = 1.0 / (k + rank + 1)

            if key in scores:
                scores[key] = (scores[key][0] + rrf_score, item)
            else:
                scores[key] = (rrf_score, item)

    return sorted(scores.values(), key=lambda x: x[0], reverse=True)


def weighted_fuse(
    ranked_lists: List[Tuple[float, List[Tuple[float, Any]]]],
    k: int = 60,
) -> List[Tuple[float, Any]]:
    """加权 RRF 融合。

    Args:
        ranked_lists: [(weight, [(score, item), ...]), ...]
                      每路一个权重 + 排序结果
        k: RRF 平滑参数

    Returns:
        [(fused_score, item), ...] 按融合分数降序

    Example:
        fused = weighted_fuse([
            (1.0, dense_results),   # dense 权重 1.0
            (0.5, sparse_results),  # sparse 权重 0.5
        ])
    """
    scores: Dict[str, Tuple[float, Any]] = {}

    for weight, lst in ranked_lists:
        for rank, (_, item) in enumerate(lst):
            key = _item_key(item)
            rrf_score = weight / (k + rank + 1)

            if key in scores:
                scores[key] = (scores[key][0] + rrf_score, item)
            else:
                scores[key] = (rrf_score, item)

    return sorted(scores.values(), key=lambda x: x[0], reverse=True)


def _item_key(item: Any) -> str:
    """从检索结果中提取稳定 key。"""
    if isinstance(item, str):
        return item
    if hasattr(item, 'text'):
        return item.text
    return str(item)
