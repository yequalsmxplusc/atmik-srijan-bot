from typing import List, Optional, Dict, Any
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class ConversationContextExtractor:
    
    def __init__(self, llm_factory, events_dict: Dict[str, Any]):
        """
        Args:
            llm_factory: Function that returns an LLM instance
            events_dict: Dictionary of all events {name_lowercase: EventMetadata}
        """
        self.llm_factory = llm_factory
        self.events_dict = events_dict
        
        # Entity extraction prompt
        self.entity_extraction_prompt = ChatPromptTemplate.from_template(
            """Extract context from this conversation.

Known Events: {event_names}

Recent Conversation:
{history}

Current Query: {query}

Determine:
1. Is this a follow-up to previous discussion?
2. Which event (if any) is being referenced?

Respond with JSON:
{{
  "is_followup": true/false,
  "referenced_event": "event_name or null",
  "reason": "explicit|implicit|pronoun|none"
}}

Examples:
Query: "Tell me about Sherlocked" → {{"is_followup": false, "referenced_event": "sherlocked", "reason": "explicit"}}
After discussing Sherlocked...
Query: "What are the prizes?" → {{"is_followup": true, "referenced_event": "sherlocked", "reason": "implicit"}}
Query: "Tell me more" → {{"is_followup": true, "referenced_event": "sherlocked", "reason": "pronoun"}}
"""
        )
    
    def extract_context(self, query: str, history: List[Any]) -> Dict[str, Any]:
        """
        Extract conversation context from client-provided history
        
        Args:
            query: Current user query
            history: Chat history from client (last N messages)
            
        Returns:
            Dict with:
                - is_followup: bool
                - referenced_event: str or None
                - enhanced_query: Query with context if needed
        """
        # If no history, it's not a follow-up
        if not history or len(history) == 0:
            return {
                "is_followup": False,
                "referenced_event": None,
                "enhanced_query": query
            }
        
        # Format recent history (last 4 messages)
        history_text = ""
        for msg in history[-4:]:
            history_text += f"{msg.role}: {msg.content}\n"
        
        # Sample of event names (first 50 to keep prompt small)
        event_names_sample = ", ".join(list(self.events_dict.keys())[:50])
        
        try:
            llm = self.llm_factory()
            chain = self.entity_extraction_prompt | llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "history": history_text,
                "event_names": event_names_sample
            })
            
            # Parse JSON response
            clean = result.replace("```json", "").replace("```", "").strip()
            extracted = json.loads(clean)
            
            is_followup = extracted.get("is_followup", False)
            referenced_event = extracted.get("referenced_event")
            
            # Build enhanced query if needed
            enhanced_query = query
            if is_followup and referenced_event:
                event_meta = self.events_dict.get(referenced_event)
                if event_meta:
                    enhanced_query = f"{query} {event_meta.name}"
                    print(f"🔗 Context: Following up on {event_meta.name}")
                else:
                    enhanced_query = f"{query} {referenced_event}"
            
            return {
                "is_followup": is_followup,
                "referenced_event": referenced_event,
                "enhanced_query": enhanced_query,
                "extraction_reason": extracted.get("reason", "none")
            }
            
        except Exception as e:
            print(f"Context extraction failed: {e}")
            # Fallback: simple check
            return self._simple_followup_check(query, history)
    
    def _simple_followup_check(self, query: str, history: List[Any]) -> Dict[str, Any]:
        """
        Fallback: Simple heuristic-based check if LLM extraction fails
        """
        query_lower = query.lower()
        followup_words = ['it', 'that', 'this', 'more', 'else', 'also', 'the event']
        
        is_vague = (
            len(query.split()) < 8 and
            any(word in query_lower for word in followup_words)
        )
        
        if is_vague and history:
            # Try to find event in recent history
            for msg in reversed(history[-3:]):
                for ename, meta in self.events_dict.items():
                    if ename in msg.content.lower():
                        return {
                            "is_followup": True,
                            "referenced_event": ename,
                            "enhanced_query": f"{query} {meta.name}",
                            "extraction_reason": "fallback"
                        }
        
        return {
            "is_followup": False,
            "referenced_event": None,
            "enhanced_query": query,
            "extraction_reason": "fallback"
        }