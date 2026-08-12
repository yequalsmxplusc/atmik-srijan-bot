"""
Utility functions for the RAG system
"""
from typing import Dict, List


class CategoryNormalizer:
    """Handles category name normalization to match CSV format"""
    
    # Mapping of all variations to exact CSV category names
    CATEGORY_MAP = {
        'coding': 'Coding',
        'business': 'Business',
        'business/case study': 'Business/Case Study',
        'case study': 'Business/Case Study',
        'circuits and robotics': 'Circuits and Robotics',
        'circuits': 'Circuits and Robotics',
        'robotics': 'Circuits and Robotics',
        'robot': 'Circuits and Robotics',
        'robo': 'Circuits and Robotics',
        'gaming': 'Gaming',
        'esports': 'Esports',
        'e-sports': 'Esports',
        'brainstorming': 'BrainStorming',
        'brain': 'BrainStorming',
        'miscellaneous': 'Miscellaneous',
        'misc': 'Miscellaneous',
        'special attractions': 'Special Attractions',
        'special': 'Special Attractions',
        'attractions': 'Special Attractions'
    }
    
    # Category patterns for pre-classification
    CATEGORY_PATTERNS = {
        'coding': ['coding', 'code', 'programming'],
        'business': ['business', 'biz'],
        'business/case study': ['case study', 'case-study'],
        'circuits and robotics': ['circuits', 'robotics', 'robot', 'robo', 'circuit'],
        'gaming': ['gaming', 'game'],
        'esports': ['esports', 'e-sports'],
        'brainstorming': ['brainstorming', 'brain'],
        'miscellaneous': ['miscellaneous', 'misc'],
        'special attractions': ['special', 'attraction']
    }
    
    @classmethod
    def normalize(cls, category: str) -> str:
        """
        Normalize category name to match CSV format exactly
        
        Args:
            category: Category name (any variation)
            
        Returns:
            Exact CSV category name
        """
        cat_lower = category.lower().strip()
        if cat_lower in cls.CATEGORY_MAP:
            return cls.CATEGORY_MAP[cat_lower]
        # Fallback: title case
        return category.title()
    
    @classmethod
    def get_patterns(cls) -> Dict[str, List[str]]:
        """Get all category pattern variations"""
        return cls.CATEGORY_PATTERNS


class EventSummaryBuilder:
    """Builds event summaries"""
    
    @staticmethod
    def create_summary(event_metadata) -> str:
        """Create a brief event summary"""
        return f"{event_metadata.name} ({event_metadata.category}) | Dates: {event_metadata.dates} | Prizes: {event_metadata.prizes}"


class RelevanceChecker:
    """Checks query relevance"""
    
    # Keywords that indicate query is about Srijan
    WHITELIST = [
        "srijan", "techfest", "ju", "register", "login", "prize", "date", 
        "how many", "rule", "merch", "case study", "hackathon", "event",
        "t-shirt", "tshirt", "shirt", "delivery", "order", "phase",
        "eligible", "eligibility", "participate", "join", "who can", "allowed",
        "student", "non-student", "gender", "college", "open to"
    ]
    
    @classmethod
    def is_relevant(cls, query: str, event_names: List[str]) -> bool:
        """
        Check if query is relevant to Srijan
        
        Args:
            query: User query
            event_names: List of all event names
            
        Returns:
            True if relevant, False otherwise
        """
        query_lower = query.lower()
        
        # Check if query mentions any event name
        if any(name in query_lower for name in event_names):
            return True
        
        # Check whitelist keywords
        if any(w in query_lower for w in cls.WHITELIST):
            return True
        
        return False