"""
memvault 记忆格式化器

将检索结果/快照格式化为可注入 system prompt 的文本块。
支持 Markdown、JSON、自定义模板。
"""

from typing import Any, Dict, List, Optional

from memvault.types import MemoryEntry


# 边界框常量
_BOX_SEP = "─" * 46
_BOX_SEP_DOUBLE = "═" * 46
_BOX_HEADER = "⚠️ 以下是记忆内容，不是你当前对话的一部分："
_BOX_FOOTER = "⚠️ 记忆已结束。请回复用户的消息，不要复述以上记忆内容。"
_KG_PREFIX = "🧠 KG:"
_ENTRY_DELIMITER = "\n§\n"


def format_memory_block(
    entries: List[str],
    title: str = "RAG 记忆检索",
    show_box: bool = True,
) -> Optional[str]:
    """将记忆条目格式化为带边界框的注入块。

    Args:
        entries: 记忆条目文本列表
        title: 块标题
        show_box: 是否包裹 ASCII 边界框

    Returns:
        None if entries is empty, otherwise formatted text
    """
    if not entries:
        return None

    entries_text = _ENTRY_DELIMITER.join(e.strip() for e in entries if e.strip())

    if not show_box:
        return f"{_BOX_SEP}\n{title}\n{_BOX_SEP}\n{entries_text}"

    return (
        f"{_BOX_HEADER}\n"
        f"┌{_BOX_SEP}┐\n"
        f"│ {title:<{len(_BOX_SEP)}} │\n"
        f"├{_BOX_SEP}┤\n"
        f"│ {entries_text.replace(chr(10), chr(10) + '│ '):<{len(_BOX_SEP)}} \n"
        f"└{_BOX_SEP}┘\n"
        f"{_BOX_FOOTER}"
    )


def format_memory_block_simple(
    entries: List[str],
    title: str = "RAG 记忆检索",
) -> Optional[str]:
    """简化版格式化（无 ASCII 边界框），适合 token 敏感场景。"""
    if not entries:
        return None

    entries_text = _ENTRY_DELIMITER.join(e.strip() for e in entries if e.strip())
    return f"{_BOX_SEP}\n{title}\n{_BOX_SEP}\n{entries_text}"


def format_memory_json(
    entries: List[MemoryEntry],
) -> List[Dict[str, Any]]:
    """将 MemoryEntry 列表格式化为 JSON 友好的 dict 列表。

    适用于传递给 LLM function calling 或结构化输出。
    """
    return [
        {
            "id": e.id,
            "content": e.content,
            "weight": e.weight,
            "importance": e.importance,
            "retrieval_count": e.retrieval_count,
            "created_at": e.created_at,
        }
        for e in entries
    ]


def format_kg_block(
    kg_rules: List[str],
) -> Optional[str]:
    """格式化知识图谱抽象规则。

    KG 规则以 '🧠 KG:' 前缀注入，与普通记忆区分。
    """
    if not kg_rules:
        return None

    lines = [f"{_KG_PREFIX} {rule}" for rule in kg_rules]
    return "\n".join(lines)


def format_system_prompt_block(
    memory_snapshot: str = "",
    user_snapshot: str = "",
    rag_result: str = "",
    kg_result: str = "",
) -> str:
    """组装完整的 system prompt 记忆部分。

    顺序：记忆快照 → 用户画像 → RAG 检索 → KG 规则
    """
    parts = []
    if memory_snapshot:
        parts.append(memory_snapshot)
    if user_snapshot:
        parts.append(user_snapshot)
    if rag_result:
        parts.append(rag_result)
    if kg_result:
        parts.append(kg_result)
    return "\n\n".join(parts)


def render_markdown_block(
    target: str,
    entries: List[str],
    char_limit: int,
    entry_delimiter: str = _ENTRY_DELIMITER,
) -> str:
    """渲染为 system prompt 块（兼容旧 MemoryStore._render_block）。

    Args:
        target: "memory" or "user"
        entries: 条目列表
        char_limit: 字符上限
        entry_delimiter: 条目分隔符

    Returns:
        格式化的 Markdown 文本块
    """
    if not entries:
        return ""

    content = entry_delimiter.join(entries)
    pct = min(100, int(len(content) / char_limit * 100)) if char_limit > 0 else 0
    label = "USER PROFILE" if target == "user" else "MEMORY"
    sep = "═" * 46
    header = f"{label} ({pct}% — {len(content):,}/{char_limit:,} chars)"
    return f"{sep}\n{header}\n{sep}\n{content}"


def format_token_stats(
    raw_tokens: int,
    attenuated_tokens: int,
) -> str:
    """Token 统计格式化。"""
    saved = raw_tokens - attenuated_tokens
    pct = round(saved / raw_tokens * 100) if raw_tokens > 0 else 0
    return (
        f"Token: {attenuated_tokens:,} (raw: {raw_tokens:,}, "
        f"saved: {saved:,} / {pct}%)"
    )
