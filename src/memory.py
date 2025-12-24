import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json


@dataclass
class MemoryNote:
    """Represents a single memory note (inspired by A-MEM)"""
    id: str
    topic: str
    summary: str
    source_url: str = ""
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    links: List[str] = field(default_factory=list)  # Links to related memories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding embedding)"""
        return {
            "id": self.id,
            "topic": self.topic,
            "summary": self.summary,
            "source_url": self.source_url,
            "keywords": self.keywords,
            "tags": self.tags,
            "created_at": self.created_at,
            "links": self.links
        }


class MemoryStore:
    """
    A-MEM inspired memory system with:
    - Vector-based retrieval
    - Memory linking
    - Persistence
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine sim)
        self.notes: Dict[str, MemoryNote] = {}
        self.note_ids: List[str] = []
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID based on content hash"""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def add_note(
        self,
        topic: str,
        summary: str,
        source_url: str = "",
        keywords: List[str] = None,
        tags: List[str] = None
    ) -> Tuple[str, bool]:
        """
        Add a note to memory. Direct and simple as requested.
        """
        # Generate ID based on content hash
        content = f"{topic}{summary}"
        note_id = self._generate_id(content)
        
        # Exact duplicate check
        if note_id in self.notes:
            return note_id, False
            
        # Create embedding
        text = f"{topic}. {summary}"
        embedding = self._embed_text(text)
        
        # 1. Semantic duplicate check (prevents "same kind" duplicates)
        if len(self.notes) > 0:
            similar = self.retrieve_similar(text, k=1)
            # If similarity > 0.96, it's basically the same information
            if similar and similar[0]['similarity_score'] > 0.96:
                return similar[0]['id'], False
        
        # Create note
        note = MemoryNote(
            id=note_id,
            topic=topic,
            summary=summary,
            source_url=source_url,
            keywords=keywords or [],
            tags=tags or []
        )
        note.embedding = embedding
        
        # Add to index and storage
        self.index.add(embedding.reshape(1, -1))
        self.note_ids.append(note_id)
        self.notes[note_id] = note
        
        return note_id, True
    
    def _generate_links(self, new_note: MemoryNote, k: int = 3, threshold: float = 0.7):
        """
        Generate links between new note and existing similar notes
        (Simplified version of A-MEM's link generation)
        """
        if len(self.notes) <= 1:
            return
        
        # Find similar notes
        similar_notes = self.retrieve_similar(
            f"{new_note.topic}. {new_note.summary}",
            k=k + 1  # +1 to exclude itself
        )
        
        # Link to similar notes above threshold
        for note_dict in similar_notes:
            if note_dict['id'] == new_note.id:
                continue
            
            # Simple linking based on keyword/tag overlap
            existing_note = self.notes[note_dict['id']]
            
            # Check similarity (would use LLM in full A-MEM implementation)
            keyword_overlap = set(new_note.keywords) & set(existing_note.keywords)
            tag_overlap = set(new_note.tags) & set(existing_note.tags)
            
            if len(keyword_overlap) > 0 or len(tag_overlap) > 0:
                # Create bidirectional link
                if existing_note.id not in new_note.links:
                    new_note.links.append(existing_note.id)
                if new_note.id not in existing_note.links:
                    existing_note.links.append(new_note.id)
    
    def retrieve_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar notes based on query
        Returns list of note dictionaries
        """
        if len(self.notes) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._embed_text(query)
        
        # Search
        k = min(k, len(self.notes))
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Get notes
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.note_ids):
                note_id = self.note_ids[idx]
                note = self.notes[note_id]
                note_dict = note.to_dict()
                note_dict['similarity_score'] = float(score)
                results.append(note_dict)
        
        return results
    
    def save_to_file(self, filepath: str):
        """Save memory to disk"""
        data = {
            "notes": [note.to_dict() for note in self.notes.values()],
            "note_ids": self.note_ids
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load memory from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing
        self.notes.clear()
        self.note_ids.clear()
        self.index.reset()
        
        # Reload notes
        for note_dict in data['notes']:
            note = MemoryNote(**note_dict)
            
            # Regenerate embedding
            text = f"{note.topic}. {note.summary}"
            embedding = self._embed_text(text)
            note.embedding = embedding
            
            # Add to index
            self.index.add(embedding.reshape(1, -1))
            self.note_ids.append(note.id)
            self.notes[note.id] = note