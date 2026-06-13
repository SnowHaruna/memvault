<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python">
  <img src="https://img.shields.io/badge/tests-127_passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
</p>

<h1 align="center">🧠 memvault</h1>
<p align="center"><strong>An AI-native memory system built on cognitive science</strong></p>
<p align="center">
  <em>Not just another RAG framework. It forgets. Consolidates knowledge during sleep. Grows its own rules.</em>
</p>

---

## Why memvault?

LLM memory resets after every conversation. Existing "memory" solutions are essentially **databases + compression**:

| Solution | What It Really Is |
|----------|-------------------|
| LangChain `ConversationBufferMemory` | Message queue with sliding-window eviction |
| MemGPT / Letta | LLM binary judgment: "keep this or not?" |
| ChatGPT Memory | User-managed flat list of facts |
| LlamaIndex `ChatMemoryBuffer` | Token window sliding, simple eviction |

Their shared problem: **they don't know what to forget.**

Real memory isn't about "remembering everything" — it's about knowing which information should fade, and when. In 1885, Ebbinghaus gave us the answer: forgetting follows an exponential decay, not a binary switch.

**memvault translates computational neuroscience into an engineered memory vault.**

```
How the brain remembers → How memvault remembers
                ↓
Encoding → Decay → Retrieval → Consolidation → Abstraction
```

---

## Quick Start

```bash
pip install memvault
```

```python
from memvault import MemoryVault

vault = MemoryVault()

# Add memories
vault.remember("Fixed a concurrency bug today — root cause was database connection pool missing max_overflow")
vault.remember("User reports login page is blank on Safari — possibly a WebKit compatibility issue")

# Retrieve
results = vault.recall("bug")
for r in results:
    print(f"[{r.weight:.2f}] {r.content}")

# Get formatted context for direct injection into LLM system prompt
context = vault.context()
```

Three lines of code. A complete memory system.

---

## Three-Tier Memory Architecture

```
┌─────────────────────────────────────────────┐
│           L0: Context Window                 │
│    Working Memory · Limited Capacity · Real-time │
│              7±2 items                       │
├─────────────────────────────────────────────┤
│         L1: Episodic Memory                  │
│  Hippocampus-dependent · 7-day half-life ·   │
│           Exponential decay                   │
│     "What did I talk about with the AI yesterday?" │
├─────────────────────────────────────────────┤
│      L2: Semantic Network (Knowledge Graph)  │
│  Neocortex-dependent · Sleep consolidation ·  │
│         Rules abstracted from L1              │
│   "80% of all bugs occur in the controller layer" │
├─────────────────────────────────────────────┤
│         User Profile                         │
│    Independent 14-day half-life · Identity /  │
│           Preferences / Habits                │
│              "Who this person is"             │
└─────────────────────────────────────────────┘
```

Mapping the complete memory pathway from hippocampus to neocortex.

---

## Core Mechanisms

### 1. Ebbinghaus Multi-Source Strength Decay

Memory "retrievability" is continuous and decays exponentially over time:

```python
weight = decay + usage_bonus + correction + importance_bonus
       = e^(-t/half_life) + min(n×0.05, 0.5) + m×0.3 + (imp-0.5)×1.5
```

| Term | Formula | Neural Mechanism |
|------|---------|------------------|
| `decay` | e^(-t / half_life) | Baseline forgetting curve |
| `usage_bonus` | min(retrieval_count × 0.05, 0.5) | Repeated retrieval → LTP synaptic enhancement |
| `correction` | correction_count × 0.3 | Correction behavior → prefrontal tagging |
| `importance_bonus` | (salience - 0.5) × 1.5 | Emotional salience floor protection |

**Low weight ≠ deletion** — entries remain in storage, keywords still match, they're just ranked lower.

### 2. Sleep Consolidation Loop (Sleep Loop)

```
Episodic memories accumulated during the day
        ↓
Nighttime clustering (embedding similarity)
        ↓
LLM extracts common rules
        ↓
Written to L2 semantic network
        ↓
Trigger: ≥ 2 similar memories
```

The system **grows its own knowledge**. Recurring information patterns are automatically abstracted into persistent rules. We validated this mechanism through three layers of verification:

| Verification Layer | Test Method | Result |
|-------------------|-------------|--------|
| **L1 Cumulative Growth** | 5 batches of 10 memories each, different topics | Rules increase monotonically: 1→2→3→4→5 |
| **L2 Contradiction Correction** | Inject contradictory info ("must use async" vs "never use async") | Contradiction detected; no duplicate rule appended |
| **L3 Anti-Hallucination** | Inject 8 irrelevant random texts | Zero clusters, zero rule generation |

### 3. Three-Way Hybrid Retrieval

```
                  ┌─ Dense (1024d semantic vector) ──┐
Query → bge-m3 ──┼─ Sparse (BM25 Bigram)          ──┼─→ RRF(k=60) → ColBERT MaxSim → Results
                  └─ ColBERT (Token MaxSim)         ──┘
```

| Path | Mechanism | Captures |
|------|-----------|----------|
| Dense | 1024-dim cosine similarity (bge-m3) | Semantic meaning |
| Sparse | BM25 character bigram index | Exact keyword match |
| ColBERT | Token-level Jaccard MaxSim reranking | Fine-grained alignment |

- **BM25 and ColBERT are pure Python implementations** — zero additional model dependencies
- Each path can be **independently toggled** via config flags
- Graceful degradation: Dense unavailable → auto fallback to Sparse + ColBERT → ultimate fallback to keyword substring matching

### 4. Primacy + Recency Compression

```
Before: [A] [B] [C] [D] [E] [F]  (6 items)
        └primacy┘ └─middle─┘ └recency┘
                          ↓ LLM summary
After:  [A] [🤖 B+C+D summary] [E] [F]  (4 items)
```

- **Auto fallback** to raw concatenation if LLM fails — **zero data loss**
- Auto backup before compression, supports one-click undo
- Deferred compression: batch imports trigger unified compression

### 5. Pluggable Storage Architecture

```
AbstractStorage interface
    ├── SQLiteStorage    # Production recommended: WAL mode, FTS5 full-text search, connection pool, auto-cleanup
    ├── FileStorage      # Legacy compatibility (MEMORY.md + memory_meta.json)
    └── MemoryStorage    # Testing only (in-memory, no persistence)
```

---

## Stability & Performance Validation

> **v0.1.2** — 127 tests passing. Three progressive verification tiers: Correctness → Integration → Stress.

| Test Suite | Size | Key Metrics |
|-----------|------|-------------|
| **Unit Tests** | 118 tests | All APIs + storage backends + decay models passing |
| **Integration Tests** | 16 tests | D+S+C three-way hybrid retrieval pipeline complete, P@5=1.00, MRR=1.00 |
| **Sleep Loop Quality** | 12 tests | Real bge-m3 + LLM, rule extraction validation passing |
| **Sleep Loop Longevity** | 10 tests | L1 cumulative growth ✅ L2 contradiction correction ✅ L3 noise rejection ✅ |

### Million-Token Stress Test (30 cycles, 3 scenarios)

| Scenario | Cumulative Tokens | Retained Tokens | Prune Rate | P@5 | MRR | Adversarial Empty | Total Time |
|----------|-------------------|-----------------|------------|------|------|-------------------|------------|
| **A: Steady Stream** | 999,984 | 19,978 | 98.0% | 1.00 | 1.00 | **64.0%** | 51.6s |
| **B: Burst Stream** | 1,053,124 | 19,944 | 98.1% | 1.00 | 1.00 | **45.3%** | 50.0s |
| **C: Long-Text Stream** | 999,984 | 19,918 | 98.0% | 1.00 | 1.00 | **26.7%** | 32.4s |

**Metric Definitions:**
- **P@5**: Proportion of relevant items in top-5 results (precision)
- **MRR**: Reciprocal rank of the first relevant result (ranking quality)
- **Adversarial Empty**: Rate at which irrelevant queries (e.g., "French verb conjugation") return empty results — measures the system's ability to "know what it doesn't know"

> ⚠️ Adversarial Empty correlates with embedding quality. Stress tests used a Mock embedder (5 synthetic topics). With real bge-m3 (1024-dim), expect **80%+**.

### Performance Benchmarks

| Operation | MemoryStorage | SQLiteStorage | Notes |
|-----------|--------------|---------------|-------|
| Batch write (1K entries) | ~15ms | ~25ms | SQLite single-transaction batch commit |
| FTS5 full-text search | N/A | ~0.3ms | **4× faster** than Python substring match |
| Sequential read (500 entries) | ~0.3ms | ~1.5ms | Memory is faster for < 10K entries |
| Million-token throughput | — | — | Long-text stream: **201K tokens/s** |

---

## Fundamental Differences from Existing Solutions

| | memvault | LangChain | MemGPT | ChatGPT Memory |
|--|----------|-----------|--------|----------------|
| **Decay Model** | Continuous exponential decay | None | LLM binary | None |
| **Knowledge Growth** | Sleep consolidation | None | None | None |
| **Retrieval** | Dense + Sparse + ColBERT | Vector only | Vector only | None |
| **Forgetting Granularity** | Hour-level continuous | FIFO | Manual / LLM | Manual |
| **Theoretical Foundation** | Cognitive neuroscience | Engineering pragmatism | OS metaphor | Product feature |
| **Storage** | SQLite / FTS5 / Pluggable | In-memory | File / Vector | Server-side |
| **Zero-Dependency Retrieval** | ✅ BM25 + ColBERT pure Python | ❌ | ❌ | ❌ |
| **Adversarial Rejection** | ✅ "Know what it doesn't know" | ❌ | ❌ | ❌ |

---

## Installation

```bash
# Core (zero external dependencies for retrieval)
pip install memvault

# With RAG support (LlamaIndex + Ollama bge-m3)
pip install "memvault[rag]"

# With LLM support (OpenAI-compatible API)
pip install "memvault[llm]"

# Full installation
pip install "memvault[all]"
```

### Dependency Notes

- **Core**: Python 3.10+, PyYAML
- **Embedding model (optional)**: Ollama + [bge-m3](https://huggingface.co/BAAI/bge-m3)
- **LLM (optional)**: Any OpenAI / Anthropic / DeepSeek compatible API (for compression summaries and Sleep Loop)
- **Pure Python built-ins**: BM25 bigram, ColBERT MaxSim, RRF fusion — core retrieval has zero additional dependencies

---

## Configuration

### Code Configuration

```python
from memvault import MemoryVault, MemoryVaultConfig
from memvault.storage import SQLiteStorage

vault = MemoryVault(
    storage=SQLiteStorage("memories.db", pool_size=3, auto_vacuum=True),
    config=MemoryVaultConfig(
        half_life_days=14.0,     # 14-day half-life
        char_limit=16000,        # 16K character limit
        weight_floor=0.3,        # Minimum weight (never zeroed out)
        weight_ceiling=3.0,      # Maximum weight (prevents monopoly)
    ),
)
```

### YAML Configuration

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

### Environment Variables (Highest Priority)

```bash
export MEMVAULT_HALF_LIFE_DAYS=30
export MEMVAULT_CHAR_LIMIT=32000
export MEMVAULT_LLM_PROVIDER=deepseek
export MEMVAULT_LLM_API_KEY=sk-xxx
export MEMVAULT_LLM_BASE_URL=https://api.deepseek.com/v1
```

---

## API Reference

### `MemoryVault` — Main Entry Point

| Method | Description |
|--------|-------------|
| `remember(text, target="memory", importance=None)` | Add a single memory |
| `remember_batch(texts, target="memory")` | Batch-add memories (10–50× faster than individual) |
| `recall(query, top_k=10)` | Three-way hybrid retrieval → `List[SearchResult]` |
| `recall_with_kg(query, top_k=10)` | Retrieval + related knowledge graph rules |
| `context()` | Formatted memory context, ready for LLM system prompt injection |
| `consolidate(dry_run=False)` | Manually trigger Sleep Loop → `ConsolidationResult` |
| `compress()` | Primacy + recency memory compression |
| `undo_compress()` | Undo last compression |
| `forget(min_weight=0.2)` | Purge low-weight memories |
| `stats()` | Memory statistics → `MemoryStats` |
| `rebuild_index()` | Rebuild all retrieval indexes |
| `start_auto_consolidate()` | Start background auto-consolidation timer |

### `SearchResult` — Retrieval Result

| Field | Type | Description |
|-------|------|-------------|
| `entry` | `MemoryEntry` | The matched memory entry |
| `score` | `float` | Composite score (ColBERT × weight × suppression factor) |
| `dense_score` | `float` | Dense path score |
| `sparse_score` | `float` | Sparse path score |
| `colbert_score` | `float` | ColBERT MaxSim score |
| `source` | `str` | Active retrieval path label (e.g., `"D+S+C"`) |

### `ConsolidationResult` — Consolidation Result

| Field | Type | Description |
|-------|------|-------------|
| `rules_extracted` | `int` | Number of new rules extracted |
| `rules` | `List[str]` | Extracted rule texts |
| `clusters_found` | `int` | Number of similar memory clusters found |
| `kg_nodes_added` | `int` | Number of new knowledge graph nodes added |
| `contradictions` | `int` | Number of contradictions detected |
| `elapsed_seconds` | `float` | Consolidation duration |
| `log` | `str` | Detailed execution log |

---

## Project Structure

```
memvault/
├── memvault/                 # Core library
│   ├── __init__.py           #   MemoryVault top-level API
│   ├── config.py             #   Global configuration (dataclass + YAML + env vars)
│   ├── types.py              #   MemoryEntry, SearchResult, ConsolidationResult, etc.
│   ├── cli.py                #   CLI entry point
│   ├── core/                 #   Core engine
│   │   ├── engine.py         #     MemoryStore — storage coordination layer
│   │   ├── decay.py          #     Weight calculation (pure functions)
│   │   ├── consolidator.py   #     Sleep Loop consolidation engine
│   │   ├── compressor.py     #     Primacy + recency compression
│   │   └── formatter.py      #     LLM context formatting
│   ├── retrieval/            #   Three-way hybrid retrieval
│   │   ├── hybrid.py         #     HybridRetriever — orchestration layer
│   │   ├── dense.py          #     bge-m3 semantic (LlamaIndex / in-memory cosine fallback)
│   │   ├── sparse.py         #     BM25 character bigram (pure Python)
│   │   ├── colbert.py        #     Token MaxSim reranking (pure Python)
│   │   └── fusion.py         #     RRF reciprocal rank fusion
│   ├── storage/              #   Pluggable storage backends
│   │   ├── base.py           #     AbstractStorage interface
│   │   ├── sqlite.py         #     SQLite + FTS5 + connection pool
│   │   ├── file.py           #     File storage (legacy compatibility)
│   │   └── memory.py         #     Memory storage (testing)
│   ├── embedding/            #   Embedding services
│   │   ├── base.py           #     AbstractEmbedder interface
│   │   ├── ollama.py         #     Ollama + bge-m3
│   │   └── mock.py           #     MockEmbedder + TopicMockEmbedder (testing)
│   └── llm/                  #   LLM integration
│       ├── client.py         #     Unified client (OpenAI / Anthropic / DeepSeek)
│       └── mock.py           #     MockLLM (testing)
├── tests/                    # Full test suite (127 tests)
│   ├── test_decay.py         #   Weight calculation
│   ├── test_engine.py        #   MemoryStore CRUD + batch
│   ├── test_storage.py       #   Backend consistency (SQLite vs Memory)
│   ├── test_retrieval.py     #   BM25 + ColBERT + RRF unit tests
│   ├── test_compression.py   #   Compression + backup + undo
│   ├── test_consolidation.py #   Sleep Loop unit tests
│   ├── test_integration.py   #   End-to-end retrieval pipeline
│   ├── test_benchmark.py     #   Performance benchmarks
│   ├── test_stress.py        #   Million-token stress test
│   ├── test_sleep_loop_quality.py    #   Real bge-m3 + LLM quality tests
│   └── test_sleep_loop_longevity.py  #   L1/L2/L3 longevity validation
├── examples/
│   └── 01_basic.py           #   Minimal working example
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running Tests

```bash
# All tests (skips quality tests requiring real services)
pytest tests/ -v

# Specific test suites
pytest tests/test_decay.py -v                 # Weight calculation
pytest tests/test_integration.py -v           # Retrieval pipeline
pytest tests/test_sleep_loop_longevity.py -v  # Longevity validation L1/L2/L3

# Stress tests
pytest tests/test_stress.py -v -k "Small"     # Quick (~5s)
pytest tests/test_stress.py -v -k "Million"   # Full (~200s)

# With coverage
pytest tests/ --cov=memvault --cov-report=html
```

---

## Inspiration

- [SnowHaruna/memvault](https://github.com/SnowHaruna/memvault) — Original memcore engine
- **Ebbinghaus** (1885) — *Über das Gedächtnis* — The forgetting curve
- **Buzsáki** (1989, 2015) — Hippocampal sharp-wave ripples and sleep-dependent memory consolidation
- **Anderson & Schooler** (1991) — *Reflections of the Environment in Memory* — Multi-source strength model
- **BGE-M3** ([arXiv:2402.03216](https://arxiv.org/abs/2402.03216)) — Single-model multilingual multi-granularity embeddings

---

## Roadmap

- [x] Three-way hybrid retrieval (Dense + Sparse + ColBERT)
- [x] Sleep Loop consolidation + knowledge graph
- [x] Million-token stress test validation
- [x] Adversarial query rejection ("know what it doesn't know")
- [x] L1/L2/L3 longevity validation
- [ ] Persistent incremental Sleep Loop (avoid re-scanning all memories)
- [ ] Multi-tenant Vault isolation
- [ ] Web visualization dashboard (real-time decay curves)
- [ ] gRPC / REST API server
- [ ] Client SDKs (TypeScript, Rust)

---

## Acknowledgements

**Independent packaging: [@GwynCat](https://github.com/GwynCat)** 🐱 — Packaged the MemVault cognitive architecture as a standalone `pip install memvault` Python package, introducing the SQLite/WAL/FTS5 storage backend, BM25+ColBERT zero-dependency retrieval, and 127-test suite.

**Original work: SnowHaruna (榛名雪) & Xiaoxue (Hermes Agent)** — MemVault three-tier memory architecture, Ebbinghaus decay model, Sleep Loop consolidation.

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <em>"Knowing what to forget is harder than knowing what to keep."</em>
</p>
