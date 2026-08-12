import hashlib
import time
from typing import Dict, Any, Optional, List
from collections import OrderedDict


class LRUCache:
    """
    Cache with LRU eviction and TTL
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 1800):
        """
        Args:
            max_size: Maximum number of entries (default 1000)
            ttl_seconds: Time to live in seconds (default 30 min)
        """
        self.cache = OrderedDict()  # Maintains insertion order
        self.max_size = max_size
        self.ttl = ttl_seconds
    
    def _get_key(self, query: str, history: List) -> str:
        """Generate cache key from query and history"""
        history_str = "|".join([f"{m.role}:{m.content}" for m in history[-2:]])
        combined = f"{query}|{history_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, history: List) -> Optional[Dict]:
        """
        Get cached response
        
        Returns None if:
        - Not in cache
        - Expired (past TTL)
        """
        key = self._get_key(query, history)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() - entry["timestamp"] >= self.ttl:
            del self.cache[key]
            return None
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        
        return entry["response"]
    
    def set(self, query: str, history: List, response: Dict):
        """
        Store response in cache
        
        If cache is full, removes least recently used entry
        """
        key = self._get_key(query, history)
        
        # If key exists, remove it (will re-add at end)
        if key in self.cache:
            del self.cache[key]
        
        # If cache is full, remove oldest (least recently used)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove first item (oldest)
        
        # Add new entry at end (most recently used)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache)
        
        # Count expired entries
        now = time.time()
        expired = sum(1 for entry in self.cache.values() 
                     if now - entry["timestamp"] >= self.ttl)
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired,
            "expired_entries": expired,
            "max_size": self.max_size,
            "utilization": f"{(total_entries / self.max_size) * 100:.1f}%"
        }