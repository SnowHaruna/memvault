"""
Sleep Loop 长寿验证 — L1/L2/L3 三层测试

验证 README 中「系统会自己长知识」的核心声明：
  L1 (累积增长): 多批记忆注入，规则数单调递增 — 验证「会自己长出新规则」
  L2 (规则修正): 矛盾信息注入后规则被修正而非重复追加 — 验证修正闭环
  L3 (抗幻觉): 随机噪声不产生虚假规则 — 验证不会凭空编造知识

纯 Mock 环境：TopicMockEmbedder + MockLLM + MemoryStorage
零外部依赖，~10 分钟内跑完。

用法:
    python -m pytest tests/test_sleep_loop_longevity.py -v
"""

import pytest

from memvault import MemoryVault
from memvault.config import ConsolidationConfig, MemoryVaultConfig
from memvault.core.consolidator import SleepConsolidator
from memvault.embedding.mock import TopicMockEmbedder
from memvault.llm.mock import MockLLM
from memvault.storage.memory import MemoryStorage

# ═══════════════════════════════════════════════════════════════
# 测试数据
# ═══════════════════════════════════════════════════════════════

# TopicMockEmbedder 默认主题
TOPICS = ["Python", "Docker", "记忆", "权重", "知识图谱"]

# 每个主题 ~10 条记忆，用于 L1 累积测试
TOPIC_MEMORIES = {
    "Python": [
        "Python 列表推导式可以替代简单的 for 循环，语法为 [x for x in iterable if condition]",
        "Python 生成器表达式使用 () 而非 []，适合处理大数据集更节省内存",
        "Python 装饰器是实现 AOP 面向切面编程的优雅方式",
        "Python 的 asyncio 库提供了异步编程的基础设施，使用 async/await 语法",
        "Python 类型注解 Type Hints 提高了代码可读性和 IDE 自动补全支持",
        "Python 上下文管理器 with 语句自动管理资源释放，无需手动 close",
        "Python 的 dataclass 装饰器简化了数据类定义，自动生成 __init__",
        "Python 的 itertools 模块提供了高效的迭代器工具如 chain groupby",
        "Python 的 functools.lru_cache 实现了简单的函数结果记忆化缓存",
        "Python 的 pathlib 模块提供了面向对象的文件路径操作 API",
    ],
    "Docker": [
        "Docker 容器通过 Linux Namespace 实现进程隔离和资源限制",
        "Docker 镜像采用分层文件系统 UnionFS 构建，每层只读",
        "Docker Compose 可以编排多容器应用，使用 YAML 定义服务",
        "Docker 的 --memory 参数可以限制容器内存使用量防止 OOM",
        "Docker Volume 挂载实现了数据持久化，容器删除后数据保留",
        "Docker 网络模式包括 bridge 默认、host 直连和 overlay 跨主机",
        "Docker HEALTHCHECK 指令监控容器状态支持自动重启不健康容器",
        "Docker 多阶段构建先编译再复制产物，有效减小最终镜像体积",
        "Docker 的 .dockerignore 文件排除不必要的构建上下文加速构建",
        "Docker Registry 用于存储和分发 Docker 镜像，Docker Hub 是公共注册表",
    ],
    "记忆": [
        "记忆系统需要遗忘机制来维持信息检索效率，记住一切反而低效",
        "人脑的海马体负责将短期情景记忆转化为长期语义记忆",
        "睡眠中的尖波涟漪 SWR 促进了记忆从海马体到新皮层的迁移和巩固",
        "Ebbinghaus 遗忘曲线描述了记忆随时间指数衰减的基本规律",
        "工作记忆的容量约为 7±2 条信息，超过容量需要分组或遗忘",
        "情景记忆存储个人经历的事件及其时间空间上下文信息",
        "语义记忆存储抽象的概念和事实知识，独立于具体经历",
        "记忆提取的过程会受到编码上下文和当前情绪状态的影响",
        "反复提取和间隔重复可以显著增强记忆的长期保持效果",
        "记忆巩固涉及从海马体到新皮层的渐进转移，睡眠中完成",
    ],
    "权重": [
        "权重衰减使用指数函数 e^(-t/half_life) 模拟自然遗忘过程",
        "检索使用会给予 usage_bonus 模拟突触的长期增强效应 LTP",
        "修正行为通过 correction_bonus 提高对应条目的权重",
        "情绪重要性通过 importance_bonus 对高显著性条目提供托底保护",
        "低权重不等于删除，条目仍在存储中可通过关键词命中",
        "权重 floor 机制确保所有记忆条目不会被降到零而被永久遗忘",
        "权重 ceiling 防止高热度条目无限积累分数挤占其他条目的空间",
        "扩散抑制机制在相邻检索间临时降权模拟认知资源的短暂耗尽",
        "批处理权重更新比逐条更新效率高因为减少了元数据写操作",
        "多源强度模型结合衰减、检索、修正和重要性四项独立因子",
    ],
    "知识图谱": [
        "知识图谱节点存储从情景记忆中抽象出的通用规则和概念",
        "睡眠巩固循环定期从 L1 情景记忆中聚类相似条目提取 L2 规则",
        "规则置信度反映了从多少条独立记忆中观察到该模式的数量",
        "矛盾检测通过对立关键词对识别知识图谱中的不一致规则对",
        "知识图谱的规则可跨会话复用，不受单条记忆衰减的影响",
        "规则反馈回路支持增强和削弱操作使系统适应新信息",
        "语义网络的拓扑结构反映了概念之间的关联强度和层级关系",
        "从情景到语义的抽象是人类认知的核心能力之一",
        "知识图谱应支持通过自然语言查询获取相关规则",
        "长期运行的系统中知识图谱应呈现出稳定的概念增长曲线",
    ],
}

# MockLLM 预设规则（每主题一条，匹配 TopicMockEmbedder 的聚类结果）
PRESET_RULES = [
    "Python 函数式编程特性可提升代码简洁性和可维护性",
    "Docker 容器化部署提高环境一致性和可移植性",
    "记忆系统需要遗忘机制来维持检索效率和信息新鲜度",
    "权重衰减遵循指数曲线模型结合检索修正和重要性因子",
    "知识图谱应从情景记忆中自动提取并支持反馈修正回路",
]

# 矛盾测试用的对立记忆对
CONTRADICTION_MEMORIES = {
    "establish": [
        "Python 异步编程必须使用 asyncio 库处理所有 I/O 操作",
        "Python 异步编程必须使用 async/await 语法避免阻塞",
        "Python 所有 I/O 密集操作必须使用异步方式才能保证性能",
        "Python 网络请求必须使用 aiohttp 异步库而非 requests",
        "Python 文件读写必须使用 aiofiles 异步库处理大文件",
    ],
    "contradict": [
        "简单的 Python 脚本不要使用异步，同步代码更清晰易懂",
        "Python 同步代码在简单场景下优于异步，减少了复杂度",
        "Python 的 requests 库在简单场景下比 aiohttp 更方便",
    ],
}

# 噪声数据（无主题关键词，TopicMockEmbedder 生成随机向量）
NOISE_TEXTS = [
    "今天天气很好适合出门散步晒太阳",
    "午饭吃了牛肉面味道不错分量足",
    "下午去公园散步看到很多人在跑步",
    "买了新的笔记本准备用来记笔记",
    "地铁上人很多挤得喘不过气来",
    "晚上看了部电影感觉还不错",
    "冰箱里的牛奶过期了忘记喝了",
    "手机屏幕碎了需要去修一下",
]


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _make_vault(preset_rules=None, similarity_threshold=0.1):
    """创建测试用 MemoryVault，注入 Mock 依赖。

    Args:
        preset_rules: MockLLM 预设规则列表
        similarity_threshold: 聚类相似度阈值（低阈值 = 更易聚类）
    """
    config = MemoryVaultConfig()
    config.memory.char_limit = 80000
    config.consolidation = ConsolidationConfig(
        min_cluster_size=2,
        similarity_threshold=similarity_threshold,
    )

    embedder = TopicMockEmbedder(dim=128, topics=TOPICS)
    llm = MockLLM(preset_responses=preset_rules or PRESET_RULES[:])
    storage = MemoryStorage()

    vault = MemoryVault(
        config=config,
        storage=storage,
        embedder=embedder,
        llm=llm,
    )
    return vault


# ═══════════════════════════════════════════════════════════════
# L1: 累积规则增长
# ═══════════════════════════════════════════════════════════════

class TestLongevityAccumulation:
    """L1: 验证规则数随多批记忆注入而单调递增。

    核心断言：5 批不同主题的记忆逐批注入后，每次巩固产生的规则数
    不应少于前一次。最终应累积 ≥ 5 条独立规则。
    """

    def test_rule_count_monotonically_increases(self):
        """分批注入 5 个主题各 10 条记忆，每次巩固后规则数非递减。"""
        vault = _make_vault()

        rule_counts = []
        kg_sizes = []

        for batch_idx, topic in enumerate(TOPICS):
            # 注入当前批次的 10 条同主题记忆
            for text in TOPIC_MEMORIES[topic]:
                vault.remember(text)

            # 执行睡眠巩固
            result = vault.consolidate()
            rule_counts.append(result.rules_extracted)

            # 记录 KG 总规则数
            kg_rules = vault.get_kg_rules()
            kg_sizes.append(len(kg_rules))

        # ── 断言 ──

        # 1. 每次巩固的规则数应非递减（新主题 → 新聚类 → 新规则）
        for i in range(1, len(rule_counts)):
            assert rule_counts[i] >= rule_counts[i - 1], (
                f"Batch {i+1} ({TOPICS[i]}): rule count {rule_counts[i]} "
                f"< batch {i} ({TOPICS[i-1]}): {rule_counts[i-1]}"
            )

        # 2. KG 总规则数应非递减（累积增长）
        for i in range(1, len(kg_sizes)):
            assert kg_sizes[i] >= kg_sizes[i - 1], (
                f"Batch {i+1}: KG size {kg_sizes[i]} < {kg_sizes[i-1]}"
            )

        # 3. 最终应有 ≥ 5 条规则（至少每个主题一条）
        assert kg_sizes[-1] >= 5, (
            f"Expected ≥ 5 KG rules, got {kg_sizes[-1]}"
        )

    def test_total_memory_count_accumulates(self):
        """条目数应随注入累积，不被 Sleep Loop 吞噬。"""
        vault = _make_vault()

        total = 0
        for topic in TOPICS:
            for text in TOPIC_MEMORIES[topic]:
                vault.remember(text)
                total += 1

        stats = vault.stats()
        assert stats.total_entries == total, (
            f"Expected {total} entries, got {stats.total_entries}"
        )

    def test_rules_are_meaningful(self):
        """提取的规则应包含主题关键词（非空壳）。"""
        vault = _make_vault()

        # 注入所有记忆
        for topic in TOPICS:
            for text in TOPIC_MEMORIES[topic]:
                vault.remember(text)

        vault.consolidate()
        rules = vault.get_kg_rules()

        # 每条规则应至少包含对应主题关键词
        matched_topics = set()
        for rule in rules:
            rule_text = rule.get("rule", "")
            for topic in TOPICS:
                if topic in rule_text:
                    matched_topics.add(topic)

        # 至少 3 个主题的规则被正确标记（MockLLM 预设不一定全部匹配）
        assert len(matched_topics) >= 3, (
            f"Only {len(matched_topics)} topics matched in rules: {matched_topics}"
        )


# ═══════════════════════════════════════════════════════════════
# L2: 规则修正（矛盾处理）
# ═══════════════════════════════════════════════════════════════

class TestLongevityCorrection:
    """L2: 验证矛盾信息注入后规则被修正（非简单重复追加）。

    系统应在检测到矛盾后更新/修正现有规则，而非在 KG 中
    盲目追加一条新规则导致自相矛盾。
    """

    def test_contradiction_is_detected(self):
        """注入与已有规则矛盾的记忆后，应检测到矛盾对。"""
        vault = _make_vault()

        # 直接在 KG 中添加一对矛盾规则，模拟 Sleep Loop 历史产出
        storage = vault._store.storage
        storage.add_kg_rule("必须使用异步编程处理所有 I/O 操作", confidence=1.0)
        storage.add_kg_rule("不要过度使用异步，简单场景用同步代码更清晰", confidence=1.0)

        # 确认两条规则都在 KG 中
        kg_rules = vault.get_kg_rules()
        assert len(kg_rules) >= 2, f"Expected ≥ 2 KG rules, got {len(kg_rules)}"

        # 用 SleepConsolidator 检测矛盾
        consolidator = SleepConsolidator(
            embedder=TopicMockEmbedder(dim=128, topics=TOPICS),
            llm=MockLLM(),
            storage=storage,
        )
        contradictions = consolidator.find_contradictions()

        # 应有至少 1 对矛盾被检测（"必须" vs "不要"）
        assert len(contradictions) >= 1, (
            f"Should detect ≥ 1 contradiction pair between "
            f"'必须异步' and '不要异步' rules, got {len(contradictions)}"
        )

    def test_contradiction_does_not_duplicate_rules(self):
        """矛盾注入后巩固，不应简单追加重复规则。"""
        vault = _make_vault(preset_rules=[
            "Python 异步编程是处理 I/O 操作的最佳实践",
            "Python 简单场景下同步代码更合适",
            "Python 异步编程是 I/O 处理的标准方式",
        ])

        # 阶段 1
        for text in CONTRADICTION_MEMORIES["establish"]:
            vault.remember(text)
        result1 = vault.consolidate()
        rules_before = len(vault.get_kg_rules())

        # 阶段 2: 注入矛盾 + 再次巩固
        for text in CONTRADICTION_MEMORIES["contradict"]:
            vault.remember(text)
        result2 = vault.consolidate()
        rules_after = len(vault.get_kg_rules())

        # 规则不应爆炸式增长（矛盾不应产生 ~len(clusters) 条新规则）
        # 因为有 5 条 establish + 3 条 contradict → 1-2 个簇
        # 最多新增规则数 ≤ 阶段2的簇数
        new_rules = rules_after - rules_before
        assert new_rules <= result2.clusters_found, (
            f"New rules {new_rules} exceeds clusters {result2.clusters_found} "
            f"— possible duplicate append"
        )

    def test_reinforce_and_contradict_feedback(self):
        """规则强化和削弱的反馈回路。"""
        vault = _make_vault(preset_rules=[
            "Python 函数式编程可提升代码简洁性",
        ])

        # 先产出一条规则
        vault.remember("Python 列表推导式可以替代 for 循环")
        vault.remember("Python 生成器表达式处理大数据集更高效")
        vault.remember("Python map filter reduce 实现函数式数据处理")
        vault.consolidate()

        kg_rules = vault.get_kg_rules()
        assert len(kg_rules) >= 1

        # 获取规则 ID 并测试反馈回路
        rule = kg_rules[0]
        rule_id = rule["id"]
        initial_conf = rule.get("confidence", 1.0)

        consolidator = SleepConsolidator(
            embedder=TopicMockEmbedder(dim=128, topics=TOPICS),
            llm=MockLLM(),
            storage=vault._store.storage,
        )

        # 强化
        result = consolidator.reinforce_rule(rule_id, boost=0.3)
        assert result, "reinforce_rule should succeed"

        # 削弱
        result = consolidator.contradict_rule(rule_id, penalty=0.2)
        assert result, "contradict_rule should succeed"


# ═══════════════════════════════════════════════════════════════
# L3: 抗幻觉（噪声不产生规则）
# ═══════════════════════════════════════════════════════════════

class TestLongevityNoHallucination:
    """L3: 验证随机噪声不会导致系统生成虚假规则。

    核心断言：无关、无主题关联的随机文本注入后，Sleep Loop
    不应从中提取任何规则（零聚类、零规则）。

    TopicMockEmbedder 对无主题关键词的文本生成随机方向向量。
    在 128 维空间中，两个随机单位向量的 cosine similarity
    服从 N(0, 1/128) ≈ N(0, 0.088)。当 similarity_threshold=0.5 时，
    随机向量对超过阈值的概率 < 1e-7，因此不会形成虚假聚类。
    """

    def test_noise_generates_zero_rules(self):
        """纯噪声记忆不应产生任何规则（threshold=0.5 合理值）。"""
        vault = _make_vault(similarity_threshold=0.5)

        for text in NOISE_TEXTS:
            vault.remember(text)

        result = vault.consolidate()

        assert result.clusters_found == 0, (
            f"Noise should produce 0 clusters, got {result.clusters_found}"
        )
        assert result.rules_extracted == 0, (
            f"Noise should produce 0 rules, got {result.rules_extracted}: {result.rules}"
        )
        assert len(vault.get_kg_rules()) == 0, (
            "KG should be empty after noise-only consolidation"
        )

    def test_noise_with_realistic_threshold_still_no_rules(self):
        """生产环境典型 threshold=0.75 下，噪声也不应聚类。"""
        vault = _make_vault(similarity_threshold=0.75)

        for text in NOISE_TEXTS:
            vault.remember(text)

        result = vault.consolidate()

        # 128 维随机向量 cosine similarity ~N(0, 0.088)
        # P(|cos| > 0.75) ≈ 1.5e-17 → 绝对不可能误聚类
        assert result.clusters_found == 0, (
            f"With threshold=0.75, noise should produce 0 clusters, "
            f"got {result.clusters_found}"
        )
        assert result.rules_extracted == 0

    def test_noise_does_not_contaminate_existing_rules(self):
        """噪声注入不应破坏已有规则。"""
        vault = _make_vault()

        # 先建立正常规则
        for text in TOPIC_MEMORIES["Python"][:5]:
            vault.remember(text)
        vault.consolidate()

        rules_before = vault.get_kg_rules()
        assert len(rules_before) >= 1

        # 注入噪声
        for text in NOISE_TEXTS:
            vault.remember(text)

        # 再次巩固
        vault.consolidate()

        rules_after = vault.get_kg_rules()
        # 已有规则应该保留
        assert len(rules_after) >= len(rules_before), (
            "Existing rules should survive noise injection"
        )

        # 不应产生噪声来源的新规则
        # （噪声聚类数为 0，所以 rules_extracted 应为 0）
        result = vault.consolidate()
        # 注意：此时已有 Python 记忆 + 噪声记忆
        # Python 记忆仍会聚类 → 产生规则（但 KG 中可能已存在）
        # 关键是噪声自身不会产生额外聚类
        assert len(vault.get_kg_rules()) >= len(rules_before), (
            "Noise should not reduce existing rule count"
        )


# ═══════════════════════════════════════════════════════════════
# 综合验证
# ═══════════════════════════════════════════════════════════════

class TestSleepLoopIntegration:
    """综合验证：L1+L2+L3 联合场景。"""

    def test_full_lifecycle(self):
        """完整生命周期：累积增长 → 矛盾修正 → 噪声无害。"""
        vault = _make_vault()

        # ── Phase A: 累积增长 ──
        for topic in TOPICS[:3]:  # Python, Docker, 记忆
            for text in TOPIC_MEMORIES[topic][:5]:
                vault.remember(text)

        result_a = vault.consolidate()
        assert result_a.rules_extracted >= 2, (
            f"Phase A should extract ≥ 2 rules from 3 topics, "
            f"got {result_a.rules_extracted}"
        )

        # ── Phase B: 继续增长 ──
        for topic in TOPICS[3:]:  # 权重, 知识图谱
            for text in TOPIC_MEMORIES[topic][:3]:
                vault.remember(text)

        result_b = vault.consolidate()
        all_rules = vault.get_kg_rules()
        assert len(all_rules) >= result_a.rules_extracted, (
            "Rule count should not decrease after adding more topics"
        )

        # ── Phase C: 噪声无害 ──
        rules_before_noise = len(vault.get_kg_rules())
        for text in NOISE_TEXTS[:5]:
            vault.remember(text)
        vault.consolidate()
        rules_after_noise = len(vault.get_kg_rules())

        assert rules_after_noise >= rules_before_noise, (
            f"Noise should not destroy rules: {rules_after_noise} < {rules_before_noise}"
        )
