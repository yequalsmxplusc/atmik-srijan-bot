"""
RAG Engine - Main Entry Point
Uses the modular rag_system package

"""

from rag_service import RAGService
from config import settings

# Create the service instance
rag_service = RAGService(settings)