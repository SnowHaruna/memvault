"""
memvault 检索层 — Dense + Sparse + ColBERT 三路混合检索

  base.py    — AbstractRetriever 接口
  dense.py   — bge-m3 稠密向量检索
  sparse.py  — BM25 bigram 稀疏检索
  colbert.py — Token MaxSim 重排序
  fusion.py  — RRF 融合
  hybrid.py  — 三路混合检索编排层
"""

from memvault.retrieval.base import AbstractRetriever
from memvault.retrieval.dense import DenseRetriever
from memvault.retrieval.sparse import BM25SparseRetriever, char_bigrams
from memvault.retrieval.colbert import ColbertMaxSimReranker
from memvault.retrieval.fusion import rrf_fuse, weighted_fuse
from memvault.retrieval.hybrid import HybridRetriever

__all__ = [
    "AbstractRetriever",
    "DenseRetriever",
    "BM25SparseRetriever",
    "ColbertMaxSimReranker",
    "HybridRetriever",
    "rrf_fuse",
    "weighted_fuse",
    "char_bigrams",
]
