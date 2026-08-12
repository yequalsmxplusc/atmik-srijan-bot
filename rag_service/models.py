"""
Data models for the RAG system
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries the system can handle"""
    SINGLE_EVENT = "single_event"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    GENERAL = "general"
    PROCEDURAL = "procedural"


class QueryClassification(BaseModel):
    """Classification result for a user query"""
    query_type: QueryType
    category_filter: Optional[str] = None
    event_names: List[str] = Field(default_factory=list)
    semantic_intent: Optional[str] = None
    requires_full_scan: bool = False


class EventMetadata(BaseModel):
    """Metadata for a single event"""
    name: str
    category: str
    concerned_club: Optional[str] = None  # NEW
    participation_mode: Optional[str] = None  # NEW: Team/Individual
    conduct_mode: Optional[str] = None  # NEW: Online/Offline/Hybrid
    team_size: Optional[str] = None
    prizes: Optional[str] = None
    dates: Optional[str] = None
    link: Optional[str] = None
    coordinators: List[str] = Field(default_factory=list)
    poster: Optional[str] = None
    drive_link: Optional[str] = None
    status: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[str] = None
    semantic_labels: List[str] = Field(default_factory=list)