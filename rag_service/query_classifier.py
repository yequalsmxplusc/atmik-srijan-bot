"""
Query classification with pre-classification patterns
"""
import re
import json
from typing import Optional
from langchain_core.output_parsers import StrOutputParser

from .models import QueryType, QueryClassification
from .utils import CategoryNormalizer
from .prompts import Prompts


class QueryClassifier:
    """Classifies user queries with pattern-based pre-classification"""
    
    def __init__(self, llm_factory):
        """
        Args:
            llm_factory: Function that returns a ChatGroq instance
        """
        self.llm_factory = llm_factory
        self.classifier_prompt = Prompts.get_classifier_prompt()
    
    def classify(self, query: str) -> QueryClassification:
        """
        Classify query with pre-classification rules for reliability
        
        Args:
            query: User query string
            
        Returns:
            QueryClassification object
        """
        query_lower = query.lower()
        
        # === PRE-CLASSIFICATION: Pattern-based rules for high-confidence cases ===
        
        # Rule 1: "How many events" → Total aggregation
        if re.search(r'\bhow\s+many\s+events?\b', query_lower) and not any(
            cat in query_lower for cat in ['coding', 'business', 'robotics', 'gaming']
        ):
            return QueryClassification(
                query_type=QueryType.AGGREGATION,
                category_filter=None,
                requires_full_scan=False
            )
        
        # Rule 2: "How many [category] events"
        for category in ['coding', 'business', 'robotics', 'gaming', 'brainstorming', 'miscellaneous']:
            if re.search(rf'\bhow\s+many\s+{category}\s+events?\b', query_lower):
                return QueryClassification(
                    query_type=QueryType.AGGREGATION,
                    category_filter=CategoryNormalizer.normalize(category),
                    requires_full_scan=False
                )
        
        # Rule 3: "What are the [category] events" or "List [category] events"
        category_patterns = CategoryNormalizer.get_patterns()
        
        for category, keywords in category_patterns.items():
            for keyword in keywords:
                # Match patterns like: "what are the coding events", "list coding events"
                if re.search(rf'\b(what|list|show|tell).*\b{keyword}\s+events?\b', query_lower):
                    return QueryClassification(
                        query_type=QueryType.AGGREGATION,
                        category_filter=CategoryNormalizer.normalize(category),
                        requires_full_scan=False
                    )
        
        # Rule 4: "List all events" → Total aggregation
        if re.search(r'\blist\s+(all\s+)?events?\b', query_lower):
            return QueryClassification(
                query_type=QueryType.AGGREGATION,
                category_filter=None,
                requires_full_scan=False
            )
        
        # Rule 5: Eligibility questions → Procedural
        eligibility_keywords = ['eligible', 'eligibility', 'can i participate', 'who can', 
                               'allowed', 'open to', 'can anyone', 'restrictions', 'requirements']
        if any(keyword in query_lower for keyword in eligibility_keywords):
            return QueryClassification(
                query_type=QueryType.PROCEDURAL,
                category_filter=None,
                requires_full_scan=False
            )
        
        # Rule 6: Registration/login queries → Procedural
        procedural_keywords = ['how to register', 'register for', 'registration', 'how to login', 
                              'sign up', 'sign in', 'how to participate']
        if any(keyword in query_lower for keyword in procedural_keywords):
            return QueryClassification(
                query_type=QueryType.PROCEDURAL,
                category_filter=None,
                requires_full_scan=False
            )
        
        # Rule 7: Merchandise queries → Semantic intent (NEW!)
        merch_keywords = ['merch', 'merchandise', 't-shirt', 'tshirt', 'shirt']
        if any(keyword in query_lower for keyword in merch_keywords):
            return QueryClassification(
                query_type=QueryType.GENERAL,
                semantic_intent="merchandise",
                requires_full_scan=False
            )
        
        # === FALLBACK: Use LLM classifier ===
        return self._llm_classify(query)
    
    def _llm_classify(self, query: str) -> QueryClassification:
        """Fallback LLM-based classification"""
        try:
            llm = self.llm_factory()
            chain = self.classifier_prompt | llm | StrOutputParser()
            result = chain.invoke({"query": query})
            clean = result.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            
            # Normalize category if present
            if data.get("category_filter"):
                data["category_filter"] = CategoryNormalizer.normalize(data["category_filter"])
            
            return QueryClassification(**data)
        except Exception as e:
            print(f"Classification error: {e}")
            return QueryClassification(query_type=QueryType.GENERAL)