"""
衰减公式正确性测试

验证 Ebbinghaus 遗忘曲线的数学正确性：
  - 指数衰减行为
  - 宽限期效应
  - 情绪半衰期
  - 检索加成
  - 修正加成
"""

import math
import time

import pytest

from memvault.core.decay import compute_weight, compute_importance


class TestComputeWeight:
    """权重计算测试。"""

    def test_basic_decay(self):
        """基础衰减：新条目权重应 > 1.0，旧条目应衰减。"""
        now = time.time()
        meta = {
            "test1": {"created_at": now, "importance": 0.5},
            "test2": {"created_at": now - 7 * 86400, "importance": 0.5},
        }

        w_fresh = compute_weight("test1", meta, now=now)
        w_old = compute_weight("test2", meta, now=now)

        # 新条目：decay=1.0 + importance_bonus=(0.5-0.5)*1.5=0 = 1.0
        assert w_fresh >= 1.0, f"新条目权重应 >= 1.0, got {w_fresh}"
        assert w_old < w_fresh, f"旧条目应衰减: {w_old} >= {w_fresh}"

    def test_grace_period(self):
        """宽限期：1 小时内不应衰减。"""
        now = time.time()
        meta = {
            "fresh": {"created_at": now - 1800, "importance": 0.5},
        }

        w = compute_weight("fresh", meta, now=now, grace_period_hours=1.0)
        assert w >= 1.0, f"宽限期内不应衰减, got {w}"

    def test_emotional_half_life(self):
        """高情绪内容使用更长的半衰期。"""
        now = time.time()
        age = 10 * 86400  # 10 天

        meta_normal = {"n": {"created_at": now - age, "importance": 0.5}}
        meta_emotional = {"e": {"created_at": now - age, "importance": 0.9}}

        w_normal = compute_weight("n", meta_normal, now=now)
        w_emotional = compute_weight("e", meta_emotional, now=now)

        assert w_emotional > w_normal, (
            f"高情绪条目应衰减更慢: {w_emotional} <= {w_normal}"
        )

    def test_usage_bonus(self):
        """检索加成：被检索多次的条目权重应更高。"""
        now = time.time()
        meta_used = {"u": {"created_at": now - 5 * 86400, "retrieval_count": 10, "importance": 0.5}}
        meta_unused = {"nu": {"created_at": now - 5 * 86400, "retrieval_count": 0, "importance": 0.5}}

        w_used = compute_weight("u", meta_used, now=now)
        w_unused = compute_weight("nu", meta_unused, now=now)

        assert w_used > w_unused, f"高检索条目应权重更高: {w_used} <= {w_unused}"

    def test_correction_bonus(self):
        """修正加成：被修正的条目权重应更高。"""
        now = time.time()
        meta_corrected = {"c": {"created_at": now - 5 * 86400, "correction_count": 3, "importance": 0.5}}
        meta_normal = {"n": {"created_at": now - 5 * 86400, "correction_count": 0, "importance": 0.5}}

        w_c = compute_weight("c", meta_corrected, now=now)
        w_n = compute_weight("n", meta_normal, now=now)

        assert w_c > w_n, f"被修正条目应权重更高: {w_c} <= {w_n}"

    def test_weight_bounds(self):
        """权重应始终在 [floor, ceiling] 范围内。"""
        now = time.time()

        # 非常旧的条目
        meta_old = {"old": {"created_at": now - 365 * 86400, "importance": 0.1}}
        w_old = compute_weight("old", meta_old, now=now,
                               weight_floor=0.3, weight_ceiling=3.0)
        assert w_old >= 0.3, f"不应低于 floor: {w_old}"
        assert w_old <= 3.0, f"不应高于 ceiling: {w_old}"

        # 新条目 + 高检索 + 高修正
        meta_hot = {"hot": {
            "created_at": now,
            "retrieval_count": 100,
            "correction_count": 10,
            "importance": 1.0,
        }}
        w_hot = compute_weight("hot", meta_hot, now=now,
                               weight_floor=0.3, weight_ceiling=3.0)
        assert w_hot <= 3.0, f"不应高于 ceiling: {w_hot}"

    def test_nonexistent_entry(self):
        """不存在的条目应返回中性权重 1.0。"""
        w = compute_weight("no_such_id", {}, now=time.time())
        assert w == 1.0

    def test_seven_day_half_life(self):
        """7 天半衰期：正好 7 天时 decay = 1/e ≈ 0.368。"""
        now = time.time()
        meta = {"t": {"created_at": now - 7 * 86400, "importance": 0.5}}

        w = compute_weight("t", meta, now=now)

        # decay = e^(-1) ≈ 0.368, + importance_bonus=0
        expected_decay = 1 / math.e
        # weight = decay + 0 + 0 + 0 = decay
        assert abs(w - expected_decay) < 0.05, (
            f"7天后权重应约等于 {expected_decay:.3f}, got {w:.3f}"
        )


class TestComputeImportance:
    """自动重要性评分测试。"""

    def test_default_importance(self):
        """普通文本应得到接近默认值的重要性。"""
        result = compute_importance("今天天气不错")
        assert 0.4 <= result["importance"] <= 0.7

    def test_emotional_text(self):
        """情绪文本应得到更高的重要性。"""
        result = compute_importance("我非常痛苦，感觉无比绝望")
        assert result["emotional_score"] > 0.5
        assert result["importance"] > 0.6

    def test_topic_detection(self):
        """应正确识别话题。"""
        result = compute_importance("我对我的信仰产生了怀疑，去教会祈祷")
        assert "faith" in result["topics"]

    def test_importance_bounds(self):
        """重要性应在 [0.05, 1.0] 范围内。"""
        # 空文本
        result = compute_importance("")
        assert 0.05 <= result["importance"] <= 1.0

        # 极高情绪文本
        result = compute_importance(
            "我极度痛苦绝望崩溃愤怒恐惧，杀了我也不能改变"
        )
        assert 0.05 <= result["importance"] <= 1.0
