"""
Data loading and indexing module
"""
import re
import io
import pandas as pd
import requests
from typing import Dict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .models import EventMetadata
from .semantic_labeler import SemanticLabeler
from .utils import CategoryNormalizer, EventSummaryBuilder


class DataLoader:
    """Loads and indexes event data from Google Sheets and markdown files"""
    
    def __init__(self, sheet_csv_url: str, embedding_model: str):
        """
        Args:
            sheet_csv_url: URL to Google Sheets CSV export
            embedding_model: Name of HuggingFace embedding model
        """
        self.sheet_csv_url = sheet_csv_url
        self.embedding_model = embedding_model
    
    def load_and_index(self) -> tuple:
        """
        Load events and guides, build indexes
        
        Returns:
            Tuple of (events_dict, category_index, semantic_index, guides_dict, vector_store, retriever)
        """
        print("Starting Semantic Data Ingestion...")
        documents = []
        events = {}
        category_index = {}
        semantic_index = {}
        guides = {}
        general_info = ""

        # Load events from Google Sheets
        events, category_index, semantic_index, event_documents = self._load_events()
        documents.extend(event_documents)
        
        # Load guides from markdown
        guides, general_info, guide_documents = self._load_guides()
        documents.extend(guide_documents)
        
        # Build vector index
        vector_store = None
        retriever = None
        if documents:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            vector_store = FAISS.from_documents(documents, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            print("Semantic Indexing Complete.")
        
        return events, category_index, semantic_index, guides, general_info, vector_store, retriever
    
    def _load_events(self) -> tuple:
        """Load events from Google Sheets CSV"""
        events = {}
        category_index = {}
        semantic_index = {}
        documents = []
        
        try:
            print("   ↳ Fetching Events from Google Sheets...")
            response = requests.get(self.sheet_csv_url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text)).fillna("")
            
            for _, row in df.iterrows():
                name = str(row.get('Event Name', '')).strip()
                if not name:
                    continue

                # Handle category (may be null)
                cat = row.get('Category')
                if pd.isna(cat) or str(cat).strip() == '':
                    cat = 'Miscellaneous'  # Default for null categories
                else:
                    cat = str(cat).strip()
                    
                norm_cat = CategoryNormalizer.normalize(cat)
                desc_text = str(row.get('Description', '')).strip()
                tags = row.get('Tags')
                tags = str(tags).strip() if pd.notna(tags) else ""
                
                # Semantic labeling
                semantic_labels = SemanticLabeler.label_event(name, desc_text, tags, norm_cat)
                
                # Create metadata
                meta = EventMetadata(
                    name=name,
                    category=norm_cat,
                    concerned_club=str(row.get('Concerned Club/Society/Department', '')).strip() or None,
                    participation_mode=str(row.get('Mode of Participation of Event', '')).strip() or None,
                    conduct_mode=str(row.get('Mode of Conduct', '')).strip() or None,
                    team_size=str(row.get('Team Size', 'TBD')),
                    prizes=str(row.get('Prizes', 'TBD')),
                    dates=str(row.get('Dates', 'TBD')),
                    link=str(row.get('Link', '')),
                    coordinators=[c.strip() for c in str(row.get('Coordinators', '')).split(',') if c.strip()],
                    poster=str(row.get('Poster', '')) if pd.notna(row.get('Poster')) else None,
                    drive_link=str(row.get('Drive Link', '')),
                    status=str(row.get('Status', 'CLOSED')),
                    format=str(row.get('Format', '')),
                    description=desc_text,
                    tags=tags if pd.notna(tags) and tags else None,
                    semantic_labels=semantic_labels
                )
                
                # Build indexes
                events[name.lower()] = meta
                
                # Category index
                if norm_cat.lower() not in category_index:
                    category_index[norm_cat.lower()] = []
                category_index[norm_cat.lower()].append(name.lower())
                
                # Semantic index
                for label in semantic_labels:
                    if label not in semantic_index:
                        semantic_index[label] = []
                    semantic_index[label].append(name.lower())

                # Vector store documents
                coord_str = ", ".join(meta.coordinators) if meta.coordinators else "TBD"
                content = f"""Event: {name}
Category: {norm_cat}
Organized by: {meta.concerned_club or 'N/A'}
Participation: {meta.participation_mode or 'N/A'}
Mode: {meta.conduct_mode or 'N/A'}
Semantic Type: {', '.join(semantic_labels) if semantic_labels else 'general'}
Tags: {tags if tags else 'N/A'}
Status: {meta.status}
Dates: {meta.dates}
Team Size: {meta.team_size}
Prizes: {meta.prizes}
Description: {desc_text}
Format: {meta.format}
Rules: {row.get('Rules','')}
Coordinators: {coord_str}
Link: {meta.link}
"""
                documents.append(Document(page_content=content, metadata={"type": "event", "name": name}))
                
                # Summary document
                summary = EventSummaryBuilder.create_summary(meta)
                searchable_summary = f"{summary} | Type: {', '.join(semantic_labels)} | {desc_text[:100]}"
                documents.append(Document(page_content=searchable_summary, metadata={"type": "event_summary", "name": name}))
                
            print(f"   Loaded {len(events)} events.")
            print(f"   Semantic Index: {dict((k, len(v)) for k, v in semantic_index.items())}")

        except Exception as e:
            print(f"   Sheet Error: {e}")
        
        return events, category_index, semantic_index, documents
    
    def _load_guides(self) -> tuple:
        """Load guides from markdown file"""
        guides = {}
        general_info = ""
        documents = []
        
        try:
            with open("techfest_static_info.md", "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # Extract general info
            if "# F.E.T.S.U. presents Srijan 2026 General Info" in raw_text:
                gen_info = raw_text.split("# F.E.T.S.U. presents Srijan 2026 General Info")[1].split("#")[0].strip()
                general_info = gen_info
                documents.append(Document(page_content=f"General Info:\n{gen_info}", metadata={"type": "general"}))
            elif "# Srijan 2026 General Info" in raw_text:  # Fallback for old format
                gen_info = raw_text.split("# Srijan 2026 General Info")[1].split("#")[0].strip()
                general_info = gen_info
                documents.append(Document(page_content=f"General Info:\n{gen_info}", metadata={"type": "general"}))

            # Extract guides
            pattern = r'\n#\s+(Guide|Merchandise):\s*(.*?)\n(.*?)(?=\n#|\Z)'
            matches = re.findall(pattern, "\n"+raw_text, re.DOTALL)
            for doc_type, title, content in matches:
                full_text = f"{doc_type}: {title}\n{content.strip()}"
                documents.append(Document(page_content=full_text, metadata={"type": doc_type.lower()}))
                
                # Store in guides dictionary
                if "sign" in title.lower() or "login" in title.lower(): 
                    guides["auth"] = full_text
                elif "participate" in title.lower(): 
                    guides["register"] = full_text
                elif "merchandise" in doc_type.lower() or "shirt" in title.lower():
                    guides["merchandise"] = full_text
                    
            print(f"   Loaded {len(matches)} guides.")
        except FileNotFoundError:
            print("   'techfest_static_info.md' not found.")
        
        return guides, general_info, documents