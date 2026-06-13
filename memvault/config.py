"""
memvault 全局配置系统

配置层级（低优先级 → 高优先级）：
  库默认值 → MemoryVaultConfig(dataclass) → memvault.yaml → 环境变量 (MEMVAULT_*)

用法：
  from memvault.config import MemoryVaultConfig, load_config

  # 方式 1: 纯代码
  config = MemoryVaultConfig(half_life_days=14.0, char_limit=16000)

  # 方式 2: 从 YAML 加载
  config = load_config("memvault.yaml")

  # 方式 3: 混合（YAML + 代码覆盖）
  config = load_config("memvault.yaml", half_life_days=30.0)

  # 方式 4: 环境变量自动覆盖
  # 设置 MEMVAULT_HALF_LIFE_DAYS=30 后，上述任一方式都会被覆盖
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple


@dataclass
class StorageConfig:
    """存储后端配置。"""
    backend: Literal["sqlite", "file", "memory"] = "sqlite"
    path: str = "./memvault.db"


@dataclass
class MemoryConfig:
    """记忆衰减参数。"""
    half_life_days: float = 7.0
    emotional_half_life_days: float = 14.0
    low_importance_half_life_days: float = 3.0
    char_limit: int = 16000
    user_char_limit: int = 8000
    weight_floor: float = 0.3
    weight_ceiling: float = 3.0
    grace_period_hours: float = 1.0
    usage_bonus_per_retrieval: float = 0.05
    max_usage_bonus: float = 0.5
    correction_bonus: float = 0.3
    importance_default: float = 0.5


@dataclass
class RetrievalConfig:
    """检索参数。"""
    dense: bool = True
    sparse: bool = True
    colbert: bool = True
    top_k: int = 10
    fetch_k_multiplier: int = 3
    rrf_k: int = 60
    relevance_threshold: float = 0.3
    tau_decay: float = 3.0
    inhibition_half_life_hours: float = 4.0
    pattern_separation_threshold: float = 0.05
    pattern_separation_discount: float = 0.6


@dataclass
class ConsolidationConfig:
    """Sleep Loop 巩固参数。"""
    enabled: bool = True
    interval_hours: int = 24
    min_cluster_size: int = 2
    similarity_threshold: float = 0.75
    max_confidence_rules: int = 20


@dataclass
class LLMConfig:
    """LLM 集成配置。"""
    provider: Literal["openai", "anthropic", "deepseek"] = "deepseek"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024
    timeout_seconds: int = 30


@dataclass
class EmbeddingConfig:
    """嵌入服务配置。"""
    provider: Literal["ollama", "remote", "mock"] = "ollama"
    model: str = "bge-m3"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 2.0
    batch_size: int = 16
    # Remote embedding options
    remote_url: str = ""
    remote_api_key: str = ""
    remote_model: str = "text-embedding-3-small"


@dataclass
class CompressionConfig:
    """首因+近因压缩参数。"""
    threshold: int = 8              # 触发压缩的条目数
    group_size: int = 3             # 每组压缩条数
    primacy_count: int = 2          # 首因保留条数
    recency_count: int = 2          # 近因保留条数
    max_compressed_slots: int = 5   # 最多保留的摘要数
    backup_enabled: bool = True     # 压缩前自动备份


@dataclass
class MemoryVaultConfig:
    """memvault 顶层配置。

    所有子配置通过嵌套 dataclass 组织。支持：
      - 环境变量覆盖: MEMVAULT_HALF_LIFE_DAYS=14
      - YAML 文件加载: load_config("memvault.yaml")
      - 代码直接构造: MemoryVaultConfig(half_life_days=14.0)
    """

    storage: StorageConfig = field(default_factory=StorageConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    def __post_init__(self):
        """应用环境变量覆盖（最高优先级）。"""
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """将 MEMVAULT_* 环境变量映射到对应字段。

        命名规则: MEMVAULT_{SECTION}_{FIELD} = 值
        例如:
          MEMVAULT_HALF_LIFE_DAYS=30
          MEMVAULT_CHAR_LIMIT=32000
          MEMVAULT_LLM_PROVIDER=openai
          MEMVAULT_LLM_MODEL=gpt-4o-mini
          MEMVAULT_EMBEDDING_BASE_URL=http://localhost:11434
          MEMVAULT_STORAGE_PATH=./my_vault.db
        """
        prefix = "MEMVAULT_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # 去掉前缀，转为小写
            field_name = key[len(prefix):].lower()
            raw_val = value.strip()

            if not raw_val:
                continue

            # 尝试匹配子配置
            matched = False

            # 遍历子配置
            for section_name in ("storage", "memory", "retrieval", "consolidation",
                                  "llm", "embedding", "compression"):
                section = getattr(self, section_name)
                # 尝试直接匹配顶层字段
                prefix_section = f"{section_name}_"
                if field_name.startswith(prefix_section):
                    sub_field = field_name[len(prefix_section):]
                    if hasattr(section, sub_field):
                        setattr(section, sub_field, _cast_env_value(raw_val, getattr(section, sub_field)))
                        matched = True
                        break

            if matched:
                continue

            # 尝试匹配平级字段（向后兼容）
            for section_name in ("storage", "memory", "retrieval", "consolidation",
                                  "llm", "embedding", "compression"):
                section = getattr(self, section_name)
                if field_name in _dataclass_fields(section):
                    setattr(section, field_name, _cast_env_value(raw_val, getattr(section, field_name)))
                    matched = True
                    break

    @property
    def half_life_days(self) -> float:
        """便捷访问：默认半衰期。"""
        return self.memory.half_life_days

    @property
    def char_limit(self) -> int:
        """便捷访问：MEMORY.md 字符上限。"""
        return self.memory.char_limit

    @property
    def user_char_limit(self) -> int:
        """便捷访问：USER.md 字符上限。"""
        return self.memory.user_char_limit

    @property
    def weight_floor(self) -> float:
        return self.memory.weight_floor

    @property
    def weight_ceiling(self) -> float:
        return self.memory.weight_ceiling

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于保存到 YAML）。"""
        result = {}
        for section_name in ("storage", "memory", "retrieval", "consolidation",
                              "llm", "embedding", "compression"):
            section = getattr(self, section_name)
            section_dict = {}
            for f in fields(section):
                val = getattr(section, f.name)
                # 隐藏敏感字段
                if f.name in ("api_key", "remote_api_key"):
                    val = "***" if val else ""
                section_dict[f.name] = val
            result[section_name] = section_dict
        return result

    def to_yaml(self, path: Optional[str] = None) -> str:
        """导出为 YAML 字符串。若提供 path 则写入文件。"""
        import yaml

        data = self.to_dict()
        yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, indent=2)

        if path:
            Path(path).write_text(yaml_str, encoding="utf-8")

        return yaml_str


# ── 辅助函数 ──

def _dataclass_fields(obj: Any) -> set:
    """获取 dataclass 实例的所有字段名。"""
    return {f.name for f in fields(obj)}


def _cast_env_value(value: str, current: Any) -> Any:
    """将环境变量字符串值转换为与当前值匹配的类型。"""
    if isinstance(current, bool):
        return value.lower() in ("true", "1", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_config(
    yaml_path: Optional[str] = None,
    **overrides,
) -> MemoryVaultConfig:
    """从 YAML 文件加载配置，并用 kwargs 覆盖。

    配置优先级：
      1. 环境变量 MEMVAULT_* （最高）
      2. **overrides kwargs
      3. memvault.yaml 文件
      4. 库默认值 （最低）

    Args:
        yaml_path: YAML 配置文件路径（可选）
        **overrides: 覆盖项，如 half_life_days=14.0

    Returns:
        MemoryVaultConfig 实例

    Example:
        # 从文件加载
        config = load_config("memvault.yaml")

        # 从文件加载 + 覆盖半衰期
        config = load_config("memvault.yaml", half_life_days=30.0)

        # 纯默认
        config = load_config()
    """
    config = MemoryVaultConfig()

    # Layer 3: YAML 文件
    if yaml_path and Path(yaml_path).exists():
        import yaml
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            _apply_yaml_data(config, yaml_data)
        except Exception:
            pass  # YAML 损坏时静默回退到默认值

    # Layer 2: kwargs 覆盖
    for key, value in overrides.items():
        _apply_override(config, key, value)

    # Layer 1: 环境变量（在 __post_init__ 中已应用）

    return config


def _apply_yaml_data(config: MemoryVaultConfig, data: Dict[str, Any]):
    """将 YAML 数据应用到配置。"""
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            continue
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            for key, value in section_data.items():
                if hasattr(section, key):
                    # 跳过占位符/空值
                    if isinstance(value, str) and value.startswith("${"):
                        continue
                    setattr(section, key, value)


def _apply_override(config: MemoryVaultConfig, key: str, value: Any):
    """应用单个覆盖项（支持嵌套配置）。"""
    # 先检查是否是子配置名
    if hasattr(config, key) and not isinstance(getattr(config, key), (int, float, str, bool)):
        return  # 跳过嵌套对象本身

    # 尝试匹配到子配置的字段
    for section_name in ("storage", "memory", "retrieval", "consolidation",
                          "llm", "embedding", "compression"):
        section = getattr(config, section_name)
        if hasattr(section, key):
            setattr(section, key, value)
            return
