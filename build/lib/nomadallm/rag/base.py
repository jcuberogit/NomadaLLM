"""
NomadaLLM RAG Base Interface

Abstract base class for RAG providers.
Developers implement this to connect their vector DB.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGContext:
    """A single piece of retrieved context."""
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass  
class RAGResult:
    """Result from a RAG query."""
    contexts: List[RAGContext]
    query: str
    total_results: int
    latency_ms: float = 0.0
    
    def to_prompt_context(self, max_contexts: int = 5) -> str:
        """Format contexts for LLM prompt injection."""
        if not self.contexts:
            return ""
        
        lines = ["### Relevant Context:"]
        for i, ctx in enumerate(self.contexts[:max_contexts], 1):
            source = f" (Source: {ctx.source})" if ctx.source else ""
            lines.append(f"{i}. {ctx.text}{source}")
        
        return "\n".join(lines)


class RAGProvider(ABC):
    """Abstract base class for RAG providers.
    
    Developers implement this interface to connect their vector database
    to NomadaLLM. The SDK does NOT include any vector DB dependencies.
    
    Example implementation for Pinecone:
    
        class MyPineconeRAG(RAGProvider):
            def __init__(self, api_key: str, index: str):
                import pinecone
                pinecone.init(api_key=api_key)
                self.index = pinecone.Index(index)
                
            async def query(self, text: str, top_k: int = 5) -> RAGResult:
                embedding = self._embed(text)
                results = self.index.query(embedding, top_k=top_k)
                contexts = [
                    RAGContext(text=r.metadata["text"], score=r.score)
                    for r in results.matches
                ]
                return RAGResult(contexts=contexts, query=text, total_results=len(contexts))
    """
    
    @abstractmethod
    async def query(
        self,
        text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        """Query the vector database for relevant contexts.
        
        Args:
            text: The query text to search for
            top_k: Maximum number of results to return
            filter: Optional metadata filters
            
        Returns:
            RAGResult with retrieved contexts
        """
        pass
    
    @abstractmethod
    async def index(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """Index texts into the vector database.
        
        Args:
            texts: List of texts to index
            metadata: Optional metadata for each text
            ids: Optional IDs for each text
            
        Returns:
            Number of texts indexed
        """
        pass
    
    def get_embedding_model(self) -> str:
        """Return the embedding model used by this provider."""
        return "unknown"
    
    def get_provider_name(self) -> str:
        """Return the name of this RAG provider."""
        return self.__class__.__name__


class NoOpRAGProvider(RAGProvider):
    """No-op RAG provider for when RAG is disabled."""
    
    async def query(
        self,
        text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        return RAGResult(contexts=[], query=text, total_results=0)
    
    async def index(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        return 0
