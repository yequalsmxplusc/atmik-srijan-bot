from fastapi import FastAPI, Security, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from cache import LRUCache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from datetime import datetime
import time
import hashlib

from config import settings
from rag_engine import rag_service

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    history: List[Message] = Field(default_factory=list)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    link: Optional[str] = None
    query_type: Optional[str] = None
    response_time_ms: Optional[float] = None
    poster: Optional[str] = None
    drive_link: Optional[str] = None
    status: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    total_events: int
    categories: List[str]
    uptime_seconds: float

class MetricsResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_misses: int
    avg_response_time_ms: float
    query_type_distribution: Dict[str, int]

class MetricsTracker:
    def __init__(self):
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.response_times = []
        self.query_types = {}
        self.start_time = time.time()
    
    def record_query(self, response_time_ms: float, cache_hit: bool, query_type: Optional[str] = None):
        self.total_queries += 1
        self.response_times.append(response_time_ms)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        if cache_hit: self.cache_hits += 1
        else: self.cache_misses += 1
        if query_type:
            self.query_types[query_type] = self.query_types.get(query_type, 0) + 1
    
    def get_metrics(self) -> MetricsResponse:
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return MetricsResponse(
            total_queries=self.total_queries,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            avg_response_time_ms=round(avg_time, 2),
            query_type_distribution=self.query_types
        )
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time

cache = LRUCache(max_size=1000, ttl_seconds=1800) 
metrics = MetricsTracker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Srijan 2026 ChatBot...")
    settings.validate()
    try:
        rag_service.load_and_index_data()
        print("RAG service initialized successfully")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize RAG service: {e}")
        raise
    yield
    print("Shutting down Srijan 2026 ChatBot...")
    cache.clear()
    print("Cache cleared")


def get_real_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

limiter = Limiter(key_func=get_real_ip)

app = FastAPI(
    title="Srijan 2026 Chatbot API",
    description="Chatbot API for FETSU presents Srijan 2026",
    version="2.1.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY_NAME = settings.SERVER_API_KEY_NAME
API_KEY = settings.SERVER_API_KEY
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials"
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    return response

@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model=settings.MODEL_ID,
        total_events=len(rag_service.events),
        categories=list(rag_service.category_index.keys()),
        uptime_seconds=round(metrics.get_uptime(), 2)
    )

@app.head("/health")
def simple_health():
    return {"status": "ok"}

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    return metrics.get_metrics()

@app.post("/chat", response_model=ChatResponse, dependencies=[Security(get_api_key)])
@limiter.limit("15/minute; 60/hour; 150/day") 
async def chat_endpoint(request: Request, chat_req: ChatRequest):
    start_time = time.time()
    
    query_clean = chat_req.query.strip()
    if not query_clean:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    cached_response = cache.get(query_clean, chat_req.history)
    if cached_response:
        response_time_ms = (time.time() - start_time) * 1000
        metrics.record_query(
            response_time_ms=response_time_ms,
            cache_hit=True,
            query_type=cached_response.get("metadata", {}).get("query_type")
        )
        
        # Return cached response 
        return ChatResponse(
            answer=cached_response["answer"],
            link=cached_response.get("link"),
            poster=cached_response.get("poster"),
            drive_link=cached_response.get("drive_link"),
            status=cached_response.get("status"),
            query_type=cached_response.get("metadata", {}).get("query_type"),
            response_time_ms=round(response_time_ms, 2)
        )
    
    # Get Fresh Response
    try:
        result = await run_in_threadpool(rag_service.get_answer, query_clean, chat_req.history)
        
        # Cache valid responses
        if "trouble connecting" not in result["answer"]:
            cache.set(query_clean, chat_req.history, result)
        
        response_time_ms = (time.time() - start_time) * 1000
        
        metrics.record_query(
            response_time_ms=response_time_ms,
            cache_hit=False,
            query_type=result.get("metadata", {}).get("query_type")
        )
        
        return ChatResponse(
            answer=result["answer"],
            link=result.get("link"),
            poster=result.get("poster"),
            drive_link=result.get("drive_link"),
            status=result.get("status"),
            query_type=result.get("metadata", {}).get("query_type"),
            response_time_ms=round(response_time_ms, 2)
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/clear-cache", dependencies=[Security(get_api_key)])
def clear_cache():
    cache.clear()
    return {"status": "success", "message": "Cache cleared"}

# Force reload endpoint
@app.post("/admin/refresh-data", dependencies=[Security(get_api_key)])
async def refresh_data():
    try:
        rag_service.load_and_index_data()
        cache.clear()
        return {
            "status": "success", 
            "message": "Data refreshed from Google Sheets",
            "events_loaded": len(rag_service.events)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events")
def list_events():
    return {
        "total": len(rag_service.events),
        "by_category": {
            category: [
                {
                    "name": rag_service.events[name].name,
                    "dates": rag_service.events[name].dates,
                    "prizes": rag_service.events[name].prizes,
                }
                for name in event_names
            ]
            for category, event_names in rag_service.category_index.items()
        }
    }

@app.get("/events/{event_name}")
def get_event(event_name: str):
    event = rag_service.events.get(event_name.lower())
    if not event:
        raise HTTPException(status_code=404, detail=f"Event '{event_name}' not found")
    return event.dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )