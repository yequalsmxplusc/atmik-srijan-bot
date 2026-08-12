import re
from datetime import datetime
import pytz
from langchain_aws import ChatBedrock  
from langchain_core.output_parsers import StrOutputParser

from .models import QueryType
from .prompts import Prompts
from .query_classifier import QueryClassifier
from .data_loader import DataLoader
from .retrieval import ContextRetriever
from .utils import RelevanceChecker
from .conversation_context import ConversationContextExtractor


class RAGService:
    
    def __init__(self, config):
        """
        Main service responsible for query processing and answer generation.
        
        Args:
            config: Configuration object with AWS credentials
        """
        self.config = config
        
        # Initialize Bedrock LLM 
        self.llm = ChatBedrock(
            model_id=config.MODEL_ID,
            region_name=config.AWS_REGION,
            credentials_profile_name=None,  
            model_kwargs={
                "temperature": 0,
                "max_tokens": 4096
            }
        )
        
        # Initialize components
        self.relevance_prompt = Prompts.get_relevance_prompt()
        self.qa_prompt = Prompts.get_qa_prompt()
        self.classifier = None
        self.retriever = None
        self.context_extractor = None
        
        # Data storage
        self.events = {}
        self.category_index = {}
        self.semantic_index = {}
        self.guides = {}
        self.general_info = ""
        self.vector_store = None
        self.vector_retriever = None
    
    def get_llm(self):
        return self.llm
    
    def load_and_index_data(self):
        """Load and index all data"""
        loader = DataLoader(
            sheet_csv_url=self.config.SHEET_CSV_URL,
            embedding_model=self.config.EMBEDDING_MODEL
        )
        
        (
            self.events,
            self.category_index,
            self.semantic_index,
            self.guides,
            self.general_info,
            self.vector_store,
            self.vector_retriever
        ) = loader.load_and_index()
        
        # Initialize components that depend on data
        self.classifier = QueryClassifier(self.get_llm)  
        self.retriever = ContextRetriever(
            events=self.events,
            category_index=self.category_index,
            semantic_index=self.semantic_index,
            guides=self.guides,
            general_info=self.general_info,
            retriever=self.vector_retriever
        )
        
        # Initialize context extractor
        self.context_extractor = ConversationContextExtractor(
            llm_factory=self.get_llm,
            events_dict=self.events
        )
    
    def get_answer(self, query: str, history: list = []) -> dict:
        """
        Get answer for user query (stateless)
        
        Args:
            query: User query string
            history: Chat history from client
            
        Returns:
            Dict with answer, link, poster, etc.
        """
        if not self.vector_store:
            return {"answer": "System initializing...", "link": None}
        
        # Extract context from client-provided history
        context_info = self.context_extractor.extract_context(query, history)
        enhanced_query = context_info["enhanced_query"]
        is_followup = context_info["is_followup"]
        
        if is_followup:
            print(f"Follow-up: {context_info.get('referenced_event', 'unknown')}")
        
        # Check relevance (skip for follow-ups)
        if not is_followup:
            if not RelevanceChecker.is_relevant(query, list(self.events.keys())):
                if not self._is_query_relevant_llm(query):
                    return {"answer": "I can only answer questions about Srijan 2026.", "link": None}
        
        # Classify query (use enhanced query)
        classification = self.classifier.classify(enhanced_query)
        print(f"Type: {classification.query_type} | Intent: {classification.semantic_intent}")
        
        # Retrieve context (use enhanced query)
        context, meta_obj = self.retriever.retrieve(enhanced_query, classification)
        
        # Generate answer
        history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in history[-2:]]) if history else ""
        curr_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M")
        
        try:
            llm = self.get_llm()
            answer = (self.qa_prompt | llm | StrOutputParser()).invoke({
                "context": context, 
                "chat_history": history_text,
                "question": query,
                "current_time": curr_time
            })
            
            # POST-PROCESSING: Fix incorrect event counts
            answer = self._fix_event_count(answer, classification)
            
            return {
                "answer": answer,
                "link": meta_obj.link if meta_obj else None,
                "poster": meta_obj.poster if meta_obj else None,
                "drive_link": meta_obj.drive_link if meta_obj else None,
                "status": meta_obj.status if meta_obj else None,
                "metadata": {
                    "query_type": classification.query_type,
                    "semantic_intent": classification.semantic_intent,
                    "is_followup": is_followup
                }
            }
        except Exception as e:
            print(f"Error: {e}")
            return {"answer": "I'm having trouble connecting. Please try again.", "link": None}
    
    def _is_query_relevant_llm(self, query: str) -> bool:
        """Fallback LLM-based relevance check."""
        try:
            llm = self.get_llm()
            res = (self.relevance_prompt | llm | StrOutputParser()).invoke({"query": query})
            return "YES" in res.strip().upper()
        except:
            return True
    
    def _fix_event_count(self, answer: str, classification) -> str:
        """Post-process answer to fix incorrect event counts"""
        if classification.query_type == QueryType.AGGREGATION and classification.category_filter:
            event_count = len(self.category_index.get(classification.category_filter.lower(), []))
            
            patterns_to_fix = [
                (r'There are (\d+) events? in the \w+ (?:category|festival)', f'There are {event_count} events'),
                (r'There are (\d+) events? in the', f'There are {event_count} events in the'),
                (r'There are (\d+) \w+ events?', f'There are {event_count} events'),
                (r'There are (\d+) events?', f'There are {event_count} events'),
                (r'Here are (\d+) \w+ events?', f'Here are {event_count} events'),
                (r'Here are (\d+) events?', f'Here are {event_count} events'),
                (r'Here are the (\d+) events?', f'Here are the {event_count} events'),
                (r'The (\d+) \w+ events? (?:are|include)', f'The {event_count} events'),
                (r'(\d+) events? in the \w+ category', f'{event_count} events in the category'),
                (r'^(\d+)\.\s+', '1. '),
            ]
            
            original = answer
            for pattern, replacement in patterns_to_fix:
                if re.search(pattern, answer, re.IGNORECASE):
                    answer = re.sub(pattern, replacement, answer, count=1, flags=re.IGNORECASE)
                    if answer != original:
                        print(f"Fixed count: {event_count} events")
                        break
        
        return answer