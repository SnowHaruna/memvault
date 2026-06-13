<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python">
  <img src="https://img.shields.io/badge/测试-127_通过-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
</p>

<h1 align="center">🧠 memvault</h1>
<p align="center"><strong>一个以认知科学为理论底座的 AI 原生记忆系统</strong></p>
<p align="center">
  <em>不是又一个 RAG 框架。它会遗忘。会在睡眠中巩固知识。会自己长出新规则。</em>
</p>

---

## 为什么需要 memvault？

LLM 的记忆在每次对话后清零。现有的"记忆"方案本质上都是**数据库 + 压缩**：

| 方案 | 本质 |
|------|------|
| LangChain `ConversationBufferMemory` | 消息队列，滑动窗口淘汰 |
| MemGPT / Letta | LLM 二值判断"这条还要不要" |
| ChatGPT Memory | 用户手动管理的扁平事实列表 |
| LlamaIndex `ChatMemoryBuffer` | Token 窗口滑动，简单驱逐 |

它们共同的问题：**不知道什么该忘。**

真正的记忆不是"记住一切"，而是知道哪些信息在什么时候应该淡出。1885 年，Ebbinghaus 就给出了答案——遗忘是指数衰减的，不是二值的。

**memvault 将计算神经科学经验，落地为一个工程化的记忆库。**

```
人脑怎么记 → memvault 就怎么记
        ↓
编码 → 衰减 → 检索 → 巩固 → 抽象
```

---

## 快速开始

```bash
pip install memvault
```

```python
from memvault import MemoryVault

vault = MemoryVault()

# 添加记忆
vault.remember("今天修复了一个并发 bug，根因是数据库连接池未设置 max_overflow")
vault.remember("用户反馈登录页面在 Safari 上白屏，可能是 WebKit 兼容问题")

# 检索
results = vault.recall("bug")
for r in results:
    print(f"[{r.weight:.2f}] {r.content}")

# 获取上下文，直接注入 LLM system prompt
context = vault.context()
```

三行代码，一个完整的记忆系统。

---

## 三层记忆架构

```
┌─────────────────────────────────────────────┐
│              L0: 上下文窗口                   │
│        工作记忆 · 容量有限 · 实时               │
│              7±2 条信息                       │
├─────────────────────────────────────────────┤
│         L1: 情景记忆                          │
│    海马体依赖 · 7 天半衰期 · 指数衰减           │
│         "昨天和 AI 聊了什么"                   │
├─────────────────────────────────────────────┤
│        L2: 语义网络（知识图谱）                  │
│   新皮层依赖 · 睡眠巩固 · 从 L1 抽象规则         │
│     "所有 bug 中 80% 出在 controller 层"       │
├─────────────────────────────────────────────┤
│         用户画像                               │
│     独立半衰期 14 天 · 身份 / 偏好 / 习惯       │
│              "这个人是谁"                      │
└─────────────────────────────────────────────┘
```

对应人脑从海马体到新皮层的完整记忆通路。

---

## 核心机制

### 1. Ebbinghaus 多源强度衰减

记忆的"可提取性"是连续的、随时间指数衰减的：

```python
weight = decay + usage_bonus + correction + importance_bonus
       = e^(-t/half_life) + min(n×0.05, 0.5) + m×0.3 + (imp-0.5)×1.5
```

| 项 | 公式 | 神经机制 |
|----|------|---------|
| `decay` | e^(-t / half_life) | 基础遗忘曲线 |
| `usage_bonus` | min(检索次数 × 0.05, 0.5) | 反复提取 → LTP 突触增强 |
| `correction` | 修正次数 × 0.3 | 修正行为 → 前额叶标记 |
| `importance_bonus` | (显著性 - 0.5) × 1.5 | 情绪显著性托底保护 |

**低权重 ≠ 删除**——条目仍在存储中，关键词仍可命中，只是排序靠后。

### 2. 睡眠巩固循环（Sleep Loop）

```
白天积累的情景记忆
        ↓
夜间聚类（嵌入相似度）
        ↓
LLM 提取共性规则
        ↓
写入 L2 语义网络
        ↓
触发条件: ≥ 2 条相似记忆
```

系统会**自己长知识**。反复出现的信息模式被自动抽象为持久规则。我们通过三层验证确认了这一机制的有效性：

| 验证层 | 测试方法 | 结果 |
|-------|---------|------|
| **L1 累积增长** | 5 批不同主题各 10 条记忆注入 | 规则数单调递增：1→2→3→4→5 |
| **L2 矛盾修正** | 注入矛盾信息（"必须异步" vs "不要异步"） | 矛盾被检测；无重复规则追加 |
| **L3 抗幻觉** | 注入 8 条无关随机文本 | 零聚类、零规则生成 |

### 3. 三路混合检索

```
                  ┌─ Dense（1024d 语义向量）──┐
Query → bge-m3 ──┼─ Sparse（BM25 Bigram）  ──┼─→ RRF(k=60) → ColBERT MaxSim → 结果
                  └─ ColBERT（Token MaxSim） ──┘
```

| 路径 | 机制 | 捕获 |
|------|------|------|
| Dense | 1024 维余弦相似度（bge-m3） | 语义含义 |
| Sparse | BM25 字符 bigram 索引 | 精确关键词匹配 |
| ColBERT | Token 级 Jaccard MaxSim 重排序 | 细粒度对齐 |

- **BM25 和 ColBERT 均为纯 Python 实现**——零额外模型依赖
- 各路通过 config flag **独立开关**
- 优雅降级：Dense 不可用 → 自动回退 Sparse + ColBERT → 最终回退关键词子串匹配

### 4. 首因 + 近因压缩

```
压缩前: [A] [B] [C] [D] [E] [F]  (6 条)
         └首因┘ └─中间─┘ └近因┘
                          ↓ LLM 摘要
压缩后: [A] [🤖 B+C+D 摘要] [E] [F]  (4 条)
```

- LLM 失败时**自动回退**为原文拼接——**零数据丢失**
- 压缩前自动备份，支持一键撤销
- 延迟压缩：批量导入后统一执行

### 5. 可插拔存储架构

```
AbstractStorage 抽象接口
    ├── SQLiteStorage    # 生产推荐：WAL 模式、FTS5 全文搜索、连接池、自动清理
    ├── FileStorage      # 兼容旧版（MEMORY.md + memory_meta.json）
    └── MemoryStorage    # 测试用（纯内存、无持久化）
```

---

## 稳定性与性能验证

> **v0.1.2** — 127 项测试通过。三轮递进验证：正确性 → 集成 → 压力。

| 测试套件 | 规模 | 关键指标 |
|---------|------|---------|
| **单元测试** | 118 项 | 全部 API + 存储后端 + 衰减模型通过 |
| **集成测试** | 16 项 | D+S+C 三路混合检索管道完整，P@5=1.00，MRR=1.00 |
| **Sleep Loop 质量** | 12 项 | 真实 bge-m3 + LLM，规则抽取验证通过 |
| **Sleep Loop 长寿验证** | 10 项 | L1 累积增长 ✅ L2 矛盾修正 ✅ L3 噪声拒绝 ✅ |

### 百万 Token 压力测试（30 周期，3 场景）

| 场景 | 累积 Token | 保留 Token | 裁剪率 | P@5 | MRR | 对抗空 | 总耗时 |
|------|-----------|-----------|--------|------|------|--------|--------|
| **A: 均匀流** | 999,984 | 19,978 | 98.0% | 1.00 | 1.00 | **64.0%** | 51.6s |
| **B: 突发流** | 1,053,124 | 19,944 | 98.1% | 1.00 | 1.00 | **45.3%** | 50.0s |
| **C: 长文流** | 999,984 | 19,918 | 98.0% | 1.00 | 1.00 | **26.7%** | 32.4s |

**指标说明：**
- **P@5**: Top-5 结果中相关条目的比例（精确率）
- **MRR**: 第一个相关结果的倒数排名（排序质量）
- **对抗空**: 无关查询（如"法语动词变位"）返回空结果的比例——衡量系统「知不知」的能力

> ⚠️ 对抗空与嵌入质量正相关。压力测试使用 Mock 嵌入器（5 个合成主题）。使用真实 bge-m3（1024 维）预期可达 **80%+**。

### 性能基准

| 操作 | MemoryStorage | SQLiteStorage | 备注 |
|------|--------------|---------------|------|
| 批量写入（1K 条） | ~15ms | ~25ms | SQLite 单事务批量提交 |
| FTS5 全文搜索 | N/A | ~0.3ms | 比 Python 子串匹配**快 4 倍** |
| 顺序读（500 条） | ~0.3ms | ~1.5ms | < 10K 条目时 Memory 更快 |
| 百万 Token 吞吐 | — | — | 长文流：**201K tokens/s** |

---

## 与现有方案的本质差异

| | memvault | LangChain | MemGPT | ChatGPT Memory |
|--|----------|-----------|--------|----------------|
| **衰减模型** | 连续指数衰减 | 无 | LLM 二值 | 无 |
| **知识生长** | 睡眠巩固 | 无 | 无 | 无 |
| **检索** | Dense + Sparse + ColBERT | 仅向量 | 仅向量 | 无 |
| **遗忘粒度** | 小时级连续 | FIFO | 手动 / LLM | 手动 |
| **理论底座** | 认知神经科学 | 工程实用 | OS 隐喻 | 产品功能 |
| **存储** | SQLite / FTS5 / 可插拔 | 内存 | 文件 / 向量 | 服务端 |
| **零依赖检索** | ✅ BM25 + ColBERT 纯 Python | ❌ | ❌ | ❌ |
| **对抗拒绝** | ✅ 「知不知」能力 | ❌ | ❌ | ❌ |

---

## 安装

```bash
# 核心（检索零外部依赖）
pip install memvault

# 含 RAG 支持（LlamaIndex + Ollama bge-m3）
pip install "memvault[rag]"

# 含 LLM 支持（OpenAI 兼容 API）
pip install "memvault[llm]"

# 完整安装
pip install "memvault[all]"
```

### 依赖说明

- **核心**: Python 3.10+、PyYAML
- **嵌入模型（可选）**: Ollama + [bge-m3](https://huggingface.co/BAAI/bge-m3)
- **LLM（可选）**: 任意 OpenAI / Anthropic / DeepSeek 兼容 API（用于压缩摘要和 Sleep Loop）
- **纯 Python 内置**: BM25 bigram、ColBERT MaxSim、RRF 融合——核心检索零额外依赖

---

## 配置

### 代码配置

```python
from memvault import MemoryVault, MemoryVaultConfig
from memvault.storage import SQLiteStorage

vault = MemoryVault(
    storage=SQLiteStorage("memories.db", pool_size=3, auto_vacuum=True),
    config=MemoryVaultConfig(
        half_life_days=14.0,     # 14 天半衰期
        char_limit=16000,        # 16K 字符上限
        weight_floor=0.3,        # 权重下限（不会被清零）
        weight_ceiling=3.0,      # 权重上限（防止垄断）
    ),
)
```

### YAML 配置

```yaml
# memvault.yaml
memory:
  half_life_days: 7.0
  char_limit: 16000
  weight_floor: 0.3
  grace_period_hours: 1.0

storage:
  backend: sqlite
  path: ./memvault.db

retrieval:
  dense: true
  sparse: true
  colbert: true
  top_k: 10
  relevance_threshold: 0.3

consolidation:
  enabled: true
  interval_hours: 24
  min_cluster_size: 2
  similarity_threshold: 0.75
```

```python
from memvault import MemoryVault, load_config

vault = MemoryVault(config=load_config("memvault.yaml"))
```

### 环境变量（最高优先级）

```bash
export MEMVAULT_HALF_LIFE_DAYS=30
export MEMVAULT_CHAR_LIMIT=32000
export MEMVAULT_LLM_PROVIDER=deepseek
export MEMVAULT_LLM_API_KEY=sk-xxx
export MEMVAULT_LLM_BASE_URL=https://api.deepseek.com/v1
```

---

## API 参考

### `MemoryVault` — 主入口

| 方法 | 说明 |
|------|------|
| `remember(text, target="memory", importance=None)` | 添加一条记忆 |
| `remember_batch(texts, target="memory")` | 批量添加记忆（比逐条快 10-50 倍） |
| `recall(query, top_k=10)` | 三路混合检索 → `List[SearchResult]` |
| `recall_with_kg(query, top_k=10)` | 检索 + 相关知识图谱规则 |
| `context()` | 格式化记忆上下文，可直接注入 LLM system prompt |
| `consolidate(dry_run=False)` | 手动触发 Sleep Loop → `ConsolidationResult` |
| `compress()` | 首因+近因压缩记忆 |
| `undo_compress()` | 撤销上次压缩 |
| `forget(min_weight=0.2)` | 清理低权重记忆 |
| `stats()` | 记忆统计 → `MemoryStats` |
| `rebuild_index()` | 重建全部检索索引 |
| `start_auto_consolidate()` | 启动后台自动巩固定时器 |

### `SearchResult` — 检索结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `entry` | `MemoryEntry` | 命中的记忆条目 |
| `score` | `float` | 综合分数（ColBERT × 权重 × 抑制因子） |
| `dense_score` | `float` | Dense 路径分数 |
| `sparse_score` | `float` | Sparse 路径分数 |
| `colbert_score` | `float` | ColBERT MaxSim 分数 |
| `source` | `str` | 活跃检索路径标签（如 `"D+S+C"`） |

### `ConsolidationResult` — 巩固结果

| 字段 | 类型 | 说明 |
|------|------|------|
| `rules_extracted` | `int` | 提取的新规则数 |
| `rules` | `List[str]` | 提取的规则文本列表 |
| `clusters_found` | `int` | 发现的相似记忆簇数 |
| `kg_nodes_added` | `int` | 新增知识图谱节点数 |
| `contradictions` | `int` | 检测到的矛盾数 |
| `elapsed_seconds` | `float` | 巩固耗时 |
| `log` | `str` | 详细执行日志 |

---

## 项目结构

```
memvault/
├── memvault/                 # 核心库
│   ├── __init__.py           #   MemoryVault 顶层 API
│   ├── config.py             #   全局配置（dataclass + YAML + 环境变量）
│   ├── types.py              #   MemoryEntry、SearchResult、ConsolidationResult 等
│   ├── cli.py                #   命令行入口
│   ├── core/                 #   核心引擎
│   │   ├── engine.py         #     MemoryStore — 存储协调层
│   │   ├── decay.py          #     权重计算（纯函数）
│   │   ├── consolidator.py   #     Sleep Loop 巩固引擎
│   │   ├── compressor.py     #     首因+近因压缩
│   │   └── formatter.py      #     LLM 上下文格式化
│   ├── retrieval/            #   三路混合检索
│   │   ├── hybrid.py         #     HybridRetriever — 编排层
│   │   ├── dense.py          #     bge-m3 语义（LlamaIndex / 内存余弦回退）
│   │   ├── sparse.py         #     BM25 字符 bigram（纯 Python）
│   │   ├── colbert.py        #     Token MaxSim 重排序（纯 Python）
│   │   └── fusion.py         #     RRF 倒数秩融合
│   ├── storage/              #   可插拔存储后端
│   │   ├── base.py           #     AbstractStorage 抽象接口
│   │   ├── sqlite.py         #     SQLite + FTS5 + 连接池
│   │   ├── file.py           #     文件存储（兼容旧版）
│   │   └── memory.py         #     内存存储（测试用）
│   ├── embedding/            #   嵌入服务
│   │   ├── base.py           #     AbstractEmbedder 抽象接口
│   │   ├── ollama.py         #     Ollama + bge-m3
│   │   └── mock.py           #     MockEmbedder + TopicMockEmbedder（测试用）
│   └── llm/                  #   LLM 集成
│       ├── client.py         #     统一客户端（OpenAI / Anthropic / DeepSeek）
│       └── mock.py           #     MockLLM（测试用）
├── tests/                    # 完整测试套件（127 项）
│   ├── test_decay.py         #   权重计算
│   ├── test_engine.py        #   MemoryStore CRUD + 批量
│   ├── test_storage.py       #   后端一致性（SQLite vs Memory）
│   ├── test_retrieval.py     #   BM25 + ColBERT + RRF 单元测试
│   ├── test_compression.py   #   压缩 + 备份 + 撤销
│   ├── test_consolidation.py #   Sleep Loop 单元测试
│   ├── test_integration.py   #   端到端检索管道
│   ├── test_benchmark.py     #   性能基准测试
│   ├── test_stress.py        #   百万 Token 压力测试
│   ├── test_sleep_loop_quality.py    #   真实 bge-m3 + LLM 质量测试
│   └── test_sleep_loop_longevity.py  #   L1/L2/L3 长寿验证
├── examples/
│   └── 01_basic.py           #   最小可用示例
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 运行测试

```bash
# 全部测试（跳过需要真实服务的质量测试）
pytest tests/ -v

# 特定测试套件
pytest tests/test_decay.py -v                 # 权重计算
pytest tests/test_integration.py -v           # 检索管道
pytest tests/test_sleep_loop_longevity.py -v  # 长寿验证 L1/L2/L3

# 压力测试
pytest tests/test_stress.py -v -k "Small"     # 快速（~5s）
pytest tests/test_stress.py -v -k "Million"   # 全量（~200s）

# 含覆盖率
pytest tests/ --cov=memvault --cov-report=html
```

---

## 灵感来源

- [SnowHaruna/memvault](https://github.com/SnowHaruna/memvault) — 原始 memcore 引擎
- **Ebbinghaus** (1885) — *Über das Gedächtnis* — 遗忘曲线
- **Buzsáki** (1989, 2015) — 海马尖波涟漪与睡眠依赖型记忆巩固
- **Anderson & Schooler** (1991) — *Reflections of the Environment in Memory* — 多源强度模型
- **BGE-M3** ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216)) — 单模型多语言多粒度嵌入

---

## 路线图

- [x] 三路混合检索（Dense + Sparse + ColBERT）
- [x] Sleep Loop 巩固 + 知识图谱
- [x] 百万 Token 压力测试验证
- [x] 对抗查询拒绝（「知不知」能力）
- [x] L1/L2/L3 长寿验证
- [ ] 持久增量 Sleep Loop（避免重复扫描全部记忆）
- [ ] 多租户 Vault 隔离
- [ ] Web 可视化界面（实时衰减曲线）
- [ ] gRPC / REST API 服务端
- [ ] 客户端 SDK（TypeScript、Rust）

---


## Acknowledgements

**独立包封装：[@GwynCat](https://github.com/GwynCat)** 🐱 — 将 MemVault 认知架构封装为 `pip install memvault` 独立 Python 包，引入 SQLite/WAL/FTS5 存储后端、BM25+ColBERT 零依赖检索、127 项测试套件。

**原作：SnowHaruna (榛名雪) & 小雪 (Hermes Agent)** — MemVault 三层记忆架构、Ebbinghaus 衰减模型、Sleep Loop 巩固。

## License

MIT — 详见 [LICENSE](LICENSE)。

---

<p align="center">
  <em>「知道什么该忘，比知道什么该存更难。」</em>
</p>
