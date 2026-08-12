"""
RAG System Package for Srijan 2026 Chatbot

This package provides a modular RAG (Retrieval-Augmented Generation) system
with semantic search, query classification, and multi-level context retrieval.
"""

from .rag_service import RAGService
from .models import QueryType, QueryClassification, EventMetadata
from .semantic_labeler import SemanticLabeler
from .utils import CategoryNormalizer

__version__ = "3.5.0"

__all__ = [
    "RAGService",
    "QueryType",
    "QueryClassification",
    "EventMetadata",
    "SemanticLabeler",
    "CategoryNormalizer",
]