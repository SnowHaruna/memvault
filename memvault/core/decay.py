"""
memvault 权重衰减引擎 — Ebbinghaus 遗忘曲线 + 多源强度模型

纯函数，零外部依赖。基于：
  - Ebbinghaus (1885): 指数遗忘曲线
  - Anderson & Schooler (1991): 多源强度模型
  - Kahana (2012): 记忆检索权重的神经机制

公式：
  weight = decay + usage_bonus + correction + importance_bonus
         = e^(-t/half_life) + min(n×0.05, 0.5) + m×0.3 + (imp-0.5)×1.5

设计约束（不可妥协）：
  1. 此公式与前端 calcWeight() 双重实现，修改时必须同步
  2. 权重 != 删除开关（低权重条目仍可被关键词命中）
  3. 权重区间 [floor, ceiling]，永不完全遗忘
"""

import math
import re
import time
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════

# 情绪词典（正则匹配）
SENTIMENT_LEXICON = {
    "strong": [
        re.compile(r"非常|极其|特别|无比|万分|极度"),
        re.compile(r"痛苦|绝望|崩溃|愤怒|恐惧|害怕|恐怖"),
        re.compile(r"悲伤|哭泣|流泪|心碎|撕心裂肺"),
        re.compile(r"恨|憎恨|仇恨|厌恶|痛恨"),
        re.compile(r"爱|深爱|热爱|挚爱"),
        re.compile(r"死|自杀|杀了|毁灭"),
        re.compile(r"背叛|抛弃|孤独|寂寞|空虚"),
        re.compile(r"原谅不了|无法原谅"),
    ],
    "moderate": [
        re.compile(r"难过|伤心|失落|沮丧|消沉|低落"),
        re.compile(r"担心|担忧|焦虑|烦躁|紧张|不安"),
        re.compile(r"遗憾|后悔|愧疚|内疚|自责|惭愧"),
        re.compile(r"想念|怀念|思念|牵挂"),
        re.compile(r"委屈|憋屈|压抑|沉闷"),
        re.compile(r"累了|疲惫|倦了"),
        re.compile(r"感激|感动|欣慰|温暖"),
        re.compile(r"迷茫|困惑|彷徨|犹豫"),
    ],
    "mild": [
        re.compile(r"有点|有些|稍微|略"),
        re.compile(r"感觉|觉得|似乎|好像"),
        re.compile(r"还好|还行|凑合"),
    ],
}

# 话题分类
TOPIC_CATEGORIES = {
    "faith": re.compile(r"神|主|上帝|耶稣|信仰|祈祷|圣经|教会|救赎|天堂|天使"),
    "guilt": re.compile(r"罪|忏悔|原谅|宽恕|错误|愧疚|内疚|自责|赎罪"),
    "family": re.compile(r"家人|父母|爸爸|妈妈|父亲|母亲|孩子|家庭"),
    "love": re.compile(r"爱|爱恋|暗恋|喜欢|失恋|分手|感情|恋爱"),
    "work": re.compile(r"工作|上班|同事|老板|辞职|失业|工资"),
    "life": re.compile(r"生活|人生|命运|未来|意义|目的"),
    "death": re.compile(r"死亡|去世|离去|告别|葬礼|失去(了|亲人)"),
    "hope": re.compile(r"希望|改变|努力|坚持|相信|勇气|力量"),
    "conflict": re.compile(r"矛盾|争吵|冲突|吵架|伤害|欺负"),
}


# ═══════════════════════════════════════════════════
# 权重计算
# ═══════════════════════════════════════════════════

def compute_weight(
    entry_id: str,
    meta: Dict[str, Any],
    now: Optional[float] = None,
    half_life_days: float = 7.0,
    emotional_half_life_days: float = 14.0,
    grace_period_hours: float = 1.0,
    usage_bonus_per_retrieval: float = 0.05,
    max_usage_bonus: float = 0.5,
    correction_bonus: float = 0.3,
    weight_floor: float = 0.3,
    weight_ceiling: float = 3.0,
) -> float:
    """计算单条记忆的检索权重。

    ⚠️ 此公式与前端 calcWeight() 双重实现。
    修改本函数时务必同步更新 JS 版本。

    weight = decay + usage_bonus + correction + importance_bonus

    Args:
        entry_id: 条目哈希 ID
        meta: 元数据字典 {entry_id: {created_at, retrieval_count, ...}}
        now: 当前时间戳（默认 time.time()）
        half_life_days: 默认衰减半衰期（天）
        emotional_half_life_days: 高情绪内容半衰期
        grace_period_hours: 宽限期（小时内不衰减）
        usage_bonus_per_retrieval: 每次检索权重增量
        max_usage_bonus: 检索加成上限
        correction_bonus: 单次修正权重增量
        weight_floor: 权重底线
        weight_ceiling: 权重上限

    Returns:
        float: 最终权重 [weight_floor, weight_ceiling]
    """
    if now is None:
        now = time.time()

    info = meta.get(entry_id, {})
    if not info:
        return 1.0  # 无元数据 → 中性权重

    created = info.get("created_at", now)
    age_days = max(0, (now - created) / 86400)
    age_hours = age_days * 24

    # 宽限期：刚写入的内容不衰减
    if age_hours < grace_period_hours:
        decay = 1.0
    else:
        # 高情绪内容用更长的半衰期
        importance = info.get("importance", 0.5)
        hl = emotional_half_life_days if importance > 0.7 else half_life_days
        decay = math.exp(-age_days / hl)

    # 检索加成（LTP 长时程增强）
    retrieval_count = info.get("retrieval_count", 0)
    usage_bonus = min(retrieval_count * usage_bonus_per_retrieval, max_usage_bonus)

    # 修正加成（前额叶标记）
    correction_count = info.get("correction_count", 0)
    correction = correction_count * correction_bonus

    # 重要性加成（多源强度模型）
    importance = info.get("importance", 0.5)
    importance_bonus = (importance - 0.5) * 1.5

    weight = decay + usage_bonus + correction + importance_bonus
    return max(weight_floor, min(weight_ceiling, weight))


def compute_weights_batch(
    entry_ids: List[str],
    meta: Dict[str, Any],
    now: Optional[float] = None,
    **kwargs,
) -> Dict[str, float]:
    """批量计算权重（优化：只遍历一次 meta）。

    Args:
        entry_ids: 条目 ID 列表
        meta: 元数据字典
        **kwargs: 传递给 compute_weight 的参数

    Returns:
        {entry_id: weight, ...}
    """
    if now is None:
        now = time.time()

    results = {}
    for eid in entry_ids:
        results[eid] = compute_weight(eid, meta, now=now, **kwargs)
    return results


# ═══════════════════════════════════════════════════
# 情绪评分
# ═══════════════════════════════════════════════════

def compute_importance(
    text: str,
    role: str = "user",
    previous_topics: Optional[List[str]] = None,
    importance_default: float = 0.5,
    importance_high_keywords: tuple = (),
    importance_high_bonus: float = 0.20,
    importance_medium_bonus: float = 0.10,
) -> Dict[str, Any]:
    """自动评分一条消息的情绪权重和话题。

    Returns:
        {
            "importance": float (0.0-1.0),
            "emotional_score": float (0.0-2.0),
            "topics": List[str],
            "topic_shift": bool,
        }
    """
    emotional_score = 0.0

    # 情绪评分
    for intensity, patterns in SENTIMENT_LEXICON.items():
        for p in patterns:
            m = p.findall(text)
            if m:
                if intensity == "strong":
                    emotional_score += len(m) * 0.4
                elif intensity == "moderate":
                    emotional_score += len(m) * 0.2
                elif intensity == "mild":
                    emotional_score += len(m) * 0.05
    emotional_score = min(2.0, emotional_score)

    # 话题识别
    topics = []
    for topic, p in TOPIC_CATEGORIES.items():
        if p.search(text):
            topics.append(topic)

    # 话题转移检测
    topic_shift = False
    if previous_topics and len(previous_topics) > 0:
        new = [t for t in topics if t not in previous_topics]
        if len(new) >= 2:
            topic_shift = True

    # 综合重要性
    importance = importance_default
    importance += emotional_score * 0.15
    importance += (len(topics) / max(1, len(TOPIC_CATEGORIES))) * 0.10
    if role == "user":
        importance += 0.10
    if topic_shift:
        importance += 0.10

    # 关键词加成
    for kw in importance_high_keywords:
        if kw in text:
            importance += importance_high_bonus
            break
    else:
        if emotional_score > 0.5:
            importance += importance_medium_bonus

    importance = min(1.0, max(0.05, importance))

    return {
        "importance": importance,
        "emotional_score": emotional_score,
        "topics": topics,
        "topic_shift": topic_shift,
    }


def score_importance(
    text: str,
    config: Optional[Any] = None,
    role: str = "user",
    previous_topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """便捷函数：使用 MemoryVaultConfig 自动评分。

    Args:
        text: 待评分的文本
        config: MemoryVaultConfig 实例（可选）
        role: 发言角色
        previous_topics: 之前的话题列表

    Returns:
        importance 评分 dict
    """
    if config is None:
        return compute_importance(text, role=role, previous_topics=previous_topics)
    return compute_importance(
        text=text,
        role=role,
        previous_topics=previous_topics,
        importance_default=config.memory.importance_default,
        importance_high_keywords=(),
        importance_high_bonus=0.20,
        importance_medium_bonus=0.10,
    )
