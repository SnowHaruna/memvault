"""
memvault 首因+近因压缩器

首因效应（Primacy）+ 近因效应（Recency）压缩策略：
  - 保留最早的 N 条（首因）
  - 保留最新的 M 条（近因）
  - 中间条目送 LLM 摘要或回退为原文拼接

关键设计：
  - LLM 失败自动回退 → 零数据丢失
  - 压缩前自动备份 → 支持撤销
  - 延迟压缩支持 → 批量导入后统一执行
  - 线程安全 → threading.Lock 防止竞争
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from memvault.types import MemoryEntry


# LLM 摘要函数类型
SummarizerFn = Callable[[List[str]], Tuple[Optional[str], Optional[str]]]
# 返回 (summary_text, error_message)


class MemoryCompressor:
    """首因+近因记忆压缩器。

    用法：
        compressor = MemoryCompressor(
            summarizer=my_llm_summarize,
            config=CompressionConfig(),
        )
        result = compressor.compress(entries)
    """

    def __init__(
        self,
        summarizer: Optional[SummarizerFn] = None,
        primacy_count: int = 2,
        recency_count: int = 2,
        group_size: int = 3,
        max_compressed_slots: int = 5,
        threshold: int = 8,
        backup_enabled: bool = True,
        backup_dir: Optional[str] = None,
    ):
        """
        Args:
            summarizer: LLM 摘要函数，签名 (batch: List[str]) -> (summary, error)
                        为 None 时使用内置回退（截断合并）
            primacy_count: 首因保留条数
            recency_count: 近因保留条数
            group_size: 每组压缩条数
            max_compressed_slots: 最多保留的摘要数
            threshold: 触发压缩的最小条目数
            backup_enabled: 压缩前是否自动备份
            backup_dir: 备份文件目录（默认当前目录）
        """
        self._summarizer = summarizer
        self.primacy_count = primacy_count
        self.recency_count = recency_count
        self.group_size = group_size
        self.max_compressed_slots = max_compressed_slots
        self.threshold = threshold
        self.backup_enabled = backup_enabled
        self._backup_dir = Path(backup_dir) if backup_dir else Path(".")
        self._lock = threading.Lock()
        self._last_backup: Optional[Dict] = None

    @property
    def summarizer(self) -> Optional[SummarizerFn]:
        return self._summarizer

    @summarizer.setter
    def summarizer(self, fn: Optional[SummarizerFn]):
        self._summarizer = fn

    def compress(
        self,
        entries: List[str],
        save_backup: bool = True,
    ) -> Dict[str, Any]:
        """执行一次压缩。

        Args:
            entries: 待压缩的条目列表
            save_backup: 是否保存备份（用于撤销）

        Returns:
            {
                "success": bool,
                "before": int,          # 压缩前条目数
                "after": int,           # 压缩后条目数
                "summaries": int,       # LLM 摘要数
                "preserved": int,       # 保留原文条数
                "errors": int,          # 摘要失败批次数
                "last_error": str,      # 最后一次错误
                "llm_used": bool,       # 是否实际使用了 LLM
                "backup_id": str|None,  # 备份标识
                "message": str,         # 人类可读摘要
            }
        """
        if not self._lock.acquire(blocking=False):
            return {"success": False, "error": "压缩正在进行中，请稍后再试。"}

        try:
            return self._compress_impl(entries, save_backup)
        finally:
            self._lock.release()

    def _compress_impl(
        self,
        entries: List[str],
        save_backup: bool,
    ) -> Dict[str, Any]:
        """压缩实现（需在锁内调用）。"""
        if len(entries) <= self.threshold:
            return {
                "success": True,
                "message": f"仅 {len(entries)} 条 (≤ {self.threshold})，无需压缩。",
                "before": len(entries),
                "after": len(entries),
                "summaries": 0,
                "preserved": 0,
                "errors": 0,
                "llm_used": False,
            }

        # 备份
        backup_id = None
        if save_backup and self.backup_enabled:
            backup_id = self._save_backup(entries)

        primacy = self.primacy_count
        recency = self.recency_count
        group_size = self.group_size
        max_slots = self.max_compressed_slots

        keep_start = entries[:primacy]
        keep_end = entries[-recency:] if recency > 0 else []

        # 防止首尾重叠
        keep_start_set = set(keep_start)
        keep_end = [e for e in keep_end if e not in keep_start_set]
        middle = entries[primacy:len(entries) - recency] if recency > 0 else entries[primacy:]

        summaries: List[str] = []
        preserved: List[str] = []
        errors = 0
        last_error = ""
        total_batches = 0
        success_batches = 0
        llm_used = False

        for i in range(0, len(middle), group_size):
            batch = middle[i:i + group_size]
            if len(batch) < 2:
                preserved.extend(batch)
                continue

            total_batches += 1

            if self._summarizer:
                summary, err = self._summarizer(batch)
                if summary:
                    summaries.append("🤖 " + summary)
                    success_batches += 1
                    llm_used = True
                else:
                    preserved.extend(batch)
                    errors += 1
                    if err:
                        last_error = err
            else:
                # 无 summarizer：回退为截断合并
                merged = self._fallback_merge(batch)
                if merged:
                    summaries.append(merged)
                else:
                    preserved.extend(batch)

        # 限制摘要数量
        if len(summaries) > max_slots:
            preserved.extend(summaries[max_slots:])
            summaries = summaries[:max_slots]

        new_entries = list(keep_start) + summaries + preserved + list(keep_end)

        msg = f"压缩完成：{len(entries)}→{len(new_entries)} 条"
        if success_batches > 0:
            msg += f"（{success_batches} 条 LLM 摘要）"
        if errors > 0:
            msg += f"，{errors} 批失败（已保留原文）"
        if not llm_used:
            msg += "，使用截断合并回退"

        return {
            "success": True,
            "before": len(entries),
            "after": len(new_entries),
            "entries": new_entries,
            "summaries": success_batches,
            "preserved": len(preserved),
            "errors": errors,
            "last_error": last_error[:300] if last_error else "",
            "llm_used": llm_used,
            "backup_id": backup_id,
            "message": msg,
        }

    def undo(self, entries: List[str]) -> Dict[str, Any]:
        """撤销最近一次压缩。

        Args:
            entries: 当前条目列表（将被写入备份中的版本覆盖）

        Returns:
            {"success": bool, "restored": int, "message": str}
        """
        if not self._last_backup:
            return {"success": False, "error": "没有可用的压缩备份。"}

        try:
            backup_entries = self._last_backup.get("entries", [])
            if not backup_entries:
                return {"success": False, "error": "备份文件为空。"}

            entries.clear()
            entries.extend(backup_entries)
            self._last_backup = None

            return {
                "success": True,
                "restored": len(backup_entries),
                "message": f"已恢复到压缩前：{len(backup_entries)} 条记忆。",
            }
        except Exception as e:
            return {"success": False, "error": f"撤销失败: {e}"}

    def has_backup(self) -> bool:
        """是否有可用的压缩备份。"""
        return self._last_backup is not None

    # ── 内部 ──

    def _save_backup(self, entries: List[str]) -> Optional[str]:
        """保存压缩前备份。"""
        backup = {
            "entries": list(entries),
            "count": len(entries),
            "timestamp": time.time(),
        }
        self._last_backup = backup

        # 也写入磁盘
        try:
            self._backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = self._backup_dir / "compression_backup.json"
            backup_file.write_text(
                json.dumps(backup, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return str(backup_file)
        except OSError:
            return None

    @staticmethod
    def _fallback_merge(batch: List[str]) -> Optional[str]:
        """无 LLM 时的回退：取每条的首句，用分号连接。"""
        first_sentences = []
        for e in batch:
            # 取第一个句子终结符之前的内容
            for sep in ("。", "！", "？", "\n", ".", "!", "?"):
                idx = e.find(sep)
                if idx > 0:
                    s = e[:idx].strip()
                    if s and len(s) > 4:
                        first_sentences.append(s)
                    break
            else:
                # 没有找到分隔符，用全文（截断到 50 字）
                s = e[:50].strip()
                if s and len(s) > 4:
                    first_sentences.append(s)

        if not first_sentences:
            return None

        return "；".join(first_sentences) + "。"


# ── 便捷工厂 ──


def create_llm_summarizer(
    llm_client: Any,
    system_prompt: Optional[str] = None,
) -> SummarizerFn:
    """从 LLM 客户端创建摘要函数。

    Args:
        llm_client: LLMClient 实例（实现 call(prompt) 方法）
        system_prompt: 自定义系统提示

    Returns:
        SummarizerFn 签名函数
    """
    if system_prompt is None:
        system_prompt = (
            "你是一个认知科学知识压缩系统。请将以下一组相关记忆片段"
            "压缩为一条简洁的摘要（不超过 80 字）。"
            "保留关键事实，去除冗余描述。"
            "只输出摘要文本，不要添加任何前缀或解释。"
        )

    def summarize(batch: List[str]) -> Tuple[Optional[str], Optional[str]]:
        prompt = system_prompt + "\n\n" + "\n".join(f"- {e[:200]}" for e in batch)
        try:
            result = llm_client.call(prompt)
            # LLM 客户端可能返回 LLMResponse 对象或纯字符串
            if hasattr(result, 'text'):
                text = result.text
            elif hasattr(result, 'strip'):
                text = result
            else:
                return None, f"Unexpected LLM response type: {type(result)}"
            text = text.strip().strip('"').strip("'")
            if not text or text.upper() == "NONE":
                return None, "LLM returned empty"
            return text, None
        except Exception as e:
            return None, str(e)

    return summarize
