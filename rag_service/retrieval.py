"""
Context retrieval module - FIXED
Fixed: Link extraction + Srijan info detection
"""
import re
from typing import Dict, List, Optional, Tuple
from .models import QueryType, QueryClassification, EventMetadata


class ContextRetriever:
    """Retrieves relevant context based on query classification"""
    
    def __init__(
        self,
        events: Dict[str, EventMetadata],
        category_index: Dict[str, List[str]],
        semantic_index: Dict[str, List[str]],
        guides: Dict[str, str],
        general_info: str,
        retriever  # FAISS retriever
    ):
        """
        Args:
            events: Dict mapping event name (lowercase) to EventMetadata
            category_index: Dict mapping category (lowercase) to list of event names
            semantic_index: Dict mapping semantic intent to list of event names
            guides: Dict mapping guide type to guide content
            general_info: General information about Srijan
            retriever: FAISS vector retriever
        """
        self.events = events
        self.category_index = category_index
        self.semantic_index = semantic_index
        self.guides = guides
        self.general_info = general_info
        self.retriever = retriever
    
    def _extract_link_from_guide(self, guide_content: str) -> Optional[str]:
        """
        Extract link from guide content
        
        Format in markdown:
        # Guide: Title
        - **Link:** /signup
        """
        match = re.search(r'\*\*Link:\*\*\s*([/\w-]+)', guide_content)
        if match:
            return match.group(1)
        return None
    
    def retrieve(
        self,
        query: str,
        classification: QueryClassification
    ) -> Tuple[str, Optional[EventMetadata]]:
        """
        Retrieve relevant context based on query classification
        
        Priority order:
        0. General Srijan Info
        0.5. Status-based queries
        1. Semantic Intent
        2. Category Aggregation
        3. Procedural
        4. Direct Event Lookup
        5. Vector Search (Fallback)
        """
        query_lower = query.lower()

        # === PRIORITY 0: GENERAL SRIJAN INFO ===
        # FIXED: More specific detection to avoid false positives
        srijan_info_patterns = [
            'what is srijan', 
            'about srijan', 
            'tell me about srijan',
            'what srijan all about'
        ]
        # Check if asking about Srijan itself (not events, not merchandise, not specific topics)
        is_general_srijan_query = (
            any(pattern in query_lower for pattern in srijan_info_patterns) and
            'event' not in query_lower and
            'merch' not in query_lower and
            'shirt' not in query_lower
        )
        
        if is_general_srijan_query and self.general_info:
            print(f"🎯 General Srijan Info Query")
            return f"*** ABOUT SRIJAN 2026 ***\n\n{self.general_info}", None

        # === PRIORITY 0.5: STATUS-BASED QUERIES ===
        status_keywords = {
            'open': ['events open', 'open events', 'which events are open', 'open right now'],
            'closed': ['events closed', 'closed events', 'which events are closed'],
            'coming soon': ['coming soon', 'upcoming events', 'future events']
        }
        
        for status, patterns in status_keywords.items():
            if any(pattern in query_lower for pattern in patterns):
                print(f"🎯 Status Query: {status}")
                
                filtered_events = []
                for ename, meta in self.events.items():
                    if meta.status and status.lower() in meta.status.lower():
                        filtered_events.append(meta)
                
                if not filtered_events:
                    return f"There are currently no events with status '{status.title()}'.", None
                
                context = f"*** EVENTS WITH STATUS: {status.upper()} ***\n\n"
                context += f"Total: {len(filtered_events)} events\n\n"
                
                for meta in filtered_events:
                    coord_str = ", ".join(meta.coordinators[:2]) if meta.coordinators else "TBD"
                    context += f"""**{meta.name}** ({meta.category})
• Status: {meta.status}
• Mode: {meta.conduct_mode or 'N/A'} | Participation: {meta.participation_mode or 'N/A'}
• Dates: {meta.dates}
• Team Size: {meta.team_size}
• Description: {meta.description[:150]}...
• Coordinators: {coord_str}

"""
                return context, filtered_events[0] if filtered_events else None

        # === PRIORITY 1: SEMANTIC INTENT ===
        if classification.semantic_intent:
            result = self._retrieve_semantic(query, classification.semantic_intent)
            if result:
                return result

        # === PRIORITY 2: CATEGORY AGGREGATION ===
        if classification.query_type == QueryType.AGGREGATION:
            return self._retrieve_aggregation(classification.category_filter)

        # === PRIORITY 3: PROCEDURAL ===
        if classification.query_type == QueryType.PROCEDURAL:
            result = self._retrieve_procedural(query_lower)
            if result:
                return result

        # === PRIORITY 4: DIRECT EVENT LOOKUP ===
        result = self._retrieve_direct(query_lower)
        if result:
            return result

        # === PRIORITY 5: VECTOR SEARCH (FALLBACK) ===
        return self._retrieve_vector(query)
    
    def _retrieve_semantic(
        self,
        query: str,
        intent: str
    ) -> Optional[Tuple[str, Optional[EventMetadata]]]:
        """Retrieve based on semantic intent"""
        
        # Special handling for merchandise
        if intent == "merchandise":
            for guide_content in self.guides.values():
                if "merchandise" in guide_content.lower() or "shirt" in guide_content.lower():
                    print(f"🎯 Merchandise Query Detected")
                    # FIXED: Extract link from guide
                    link = self._extract_link_from_guide(guide_content)
                    meta = EventMetadata(
                        name="Merchandise",
                        category="Guide",
                        link=link or "/merchandise"
                    )
                    return f"*** SRIJAN 2026 MERCHANDISE ***\n\n{guide_content}", meta
            
            # Fallback to vector search
            docs = self.retriever.invoke(query)
            merch_context = "\n\n".join(
                doc.page_content for doc in docs if "merch" in doc.page_content.lower()
            )
            if merch_context:
                return merch_context, None
        
        # Regular semantic intent
        event_names = self.semantic_index.get(intent, [])
        
        if event_names:
            print(f"🎯 Semantic Match: '{intent}' → {len(event_names)} events")
            context = f"*** EVENTS MATCHING '{intent.upper().replace('_', ' ')}' ***\n\n"
            
            matched_events = []
            for ename in event_names:
                meta = self.events.get(ename)
                if meta:
                    matched_events.append(meta)
                    coord_str = ", ".join(meta.coordinators[:2]) if meta.coordinators else "TBD"
                    
                    context += f"""**{meta.name}** ({meta.category})
• Organized by: {meta.concerned_club or 'N/A'}
• Mode: {meta.conduct_mode or 'N/A'} | Participation: {meta.participation_mode or 'N/A'}
• Dates: {meta.dates}
• Team Size: {meta.team_size}
• Prizes: {meta.prizes}
• Description: {meta.description[:200]}{'...' if len(meta.description) > 200 else ''}
• Coordinators: {coord_str}

"""
            
            return context, matched_events[0] if matched_events else None
        
        return None
    
    def _retrieve_aggregation(
        self,
        category_filter: Optional[str]
    ) -> Tuple[str, Optional[EventMetadata]]:
        """Retrieve aggregated event lists"""
        
        if category_filter:
            cat_key = category_filter.lower()
            event_names = self.category_index.get(cat_key, [])
            
            if not event_names:
                return f"No events found in category: {category_filter}", None
            
            context = f"*** EVENTS IN {category_filter.upper()} CATEGORY ***\n\n"
            for name in event_names:
                meta = self.events.get(name)
                if meta: 
                    context += f"""**{meta.name}**
• Organized by: {meta.concerned_club or 'N/A'}
• Mode: {meta.conduct_mode or 'N/A'} | Participation: {meta.participation_mode or 'N/A'}
• Dates: {meta.dates}
• Team Size: {meta.team_size}
• Prizes: {meta.prizes}
• Description: {meta.description[:150]}...

"""
            return context, None
        else:
            total = len(self.events)
            context = f"*** TOTAL SRIJAN 2026 EVENTS: {total} ***\n\nBreakdown by Category:\n\n"
            
            for cat, elist in sorted(self.category_index.items()):
                context += f"**{cat.capitalize()}**: {len(elist)} events\n"
                for ename in elist[:3]:
                    meta = self.events.get(ename)
                    if meta:
                        context += f"  • {meta.name}\n"
                if len(elist) > 3:
                    context += f"  • ...and {len(elist) - 3} more\n"
                context += "\n"
            
            return context, None
    
    def _retrieve_procedural(
        self,
        query_lower: str
    ) -> Optional[Tuple[str, EventMetadata]]:
        """Retrieve procedural guides with link extraction"""
        
        # Check for registration queries
        if "register" in query_lower or "registration" in query_lower:
            specific_event_meta = None
            for ename, meta in self.events.items():
                if ename in query_lower:
                    specific_event_meta = meta
                    break
            
            reg_guide = self.guides.get("register", "")
            
            if specific_event_meta:
                coord_str = ", ".join(specific_event_meta.coordinators) if specific_event_meta.coordinators else "TBD"
                context = f"""*** HOW TO REGISTER FOR {specific_event_meta.name.upper()} ***

{reg_guide}

**Event Details:**
• Event: {specific_event_meta.name}
• Category: {specific_event_meta.category}
• Participation: {specific_event_meta.participation_mode or 'N/A'}
• Team Size: {specific_event_meta.team_size}
• Coordinators: {coord_str}
• Link: {specific_event_meta.link}

For any queries about this specific event, you can contact the coordinators listed above.
"""
                return context, specific_event_meta
            else:
                # FIXED: Extract link from guide
                link = self._extract_link_from_guide(reg_guide)
                meta = EventMetadata(name="Registration", category="Guide", link=link or "/events")
                return reg_guide, meta
        
        # Check for eligibility questions
        eligibility_keywords = ['eligible', 'eligibility', 'can i participate', 'who can', 'allowed', 
                               'open to', 'can anyone', 'restrictions', 'requirements']
        if any(keyword in query_lower for keyword in eligibility_keywords):
            context = """*** ELIGIBILITY INFORMATION ***

All Srijan 2026 events are open to EVERYONE:
• Students and non-students welcome
• Any college or university
• Any gender, background, or identity
• No restrictions unless explicitly stated in specific event rules

You are welcome to participate in any event you're interested in!

Visit the events page to browse all 52 events and register for the ones you like.
"""
            meta = EventMetadata(name="Eligibility", category="Guide", link="/events")
            return context, meta
        
        # Check for login/signin queries
        if "login" in query_lower or "sign in" in query_lower:
            login_guide = self.guides.get("auth", "")
            # FIXED: Extract link from guide
            link = self._extract_link_from_guide(login_guide)
            meta = EventMetadata(name="Login", category="Guide", link=link or "/login")
            return login_guide, meta
        
        # Check for signup queries
        if "sign up" in query_lower or "signup" in query_lower:
            auth_guide = self.guides.get("auth", "")
            # FIXED: Extract link from guide
            link = self._extract_link_from_guide(auth_guide)
            meta = EventMetadata(name="Sign Up", category="Guide", link=link or "/signup")
            return auth_guide, meta
        
        return None
    
    def _retrieve_direct(
        self,
        query_lower: str
    ) -> Optional[Tuple[str, EventMetadata]]:
        """Direct event name lookup"""
        
        for ename, meta in self.events.items():
            if ename in query_lower:
                print(f"🎯 Direct Hit: {meta.name}")
                coord_str = ", ".join(meta.coordinators)
                context = f"""*** EVENT DETAILS ***
Event: {meta.name}
Category: {meta.category}
Organized by: {meta.concerned_club or 'N/A'}
Mode of Conduct: {meta.conduct_mode or 'N/A'}
Participation Type: {meta.participation_mode or 'N/A'}
Type: {', '.join(meta.semantic_labels) if meta.semantic_labels else 'General'}
Status: {meta.status}
Description: {meta.description}
Dates: {meta.dates}
Team Size: {meta.team_size}
Prizes: {meta.prizes}
Format: {meta.format}
Coordinators: {coord_str}
"""
                return context, meta
        
        return None
    
    def _retrieve_vector(self, query: str) -> Tuple[str, Optional[EventMetadata]]:
        """Vector search fallback"""
        
        docs = self.retriever.invoke(query)
        matched_meta = None
        
        if docs and docs[0].metadata.get("type") == "event":
            ename = docs[0].metadata.get("name", "").lower()
            if ename in self.events: 
                matched_meta = self.events[ename]
        
        vector_context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return vector_context, matched_meta