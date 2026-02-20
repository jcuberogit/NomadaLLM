"""
NomadaLLM RAG Interface

Extensible interface for third-party RAG providers.
NomadaLLM does NOT include a vector database - developers plug in their own.

Supported providers (developer-provided):
- Weaviate
- Pinecone
- Chroma
- Qdrant
- Milvus
- Custom implementations

Usage:
    from nomadallm import NomadaLLM
    from nomadallm.rag import RAGProvider, PineconeRAG
    
    # Developer implements their RAG
    rag = PineconeRAG(api_key="...", index="my-index")
    
    # Attach to NomadaLLM
    llm = NomadaLLM()
    llm.attach_rag(rag)
    
    # Now prompts automatically include context
    response = await llm.prompt("What's our refund policy?")
"""

from nomadallm.rag.base import RAGProvider, RAGContext, RAGResult
from nomadallm.rag.adapters import (
    PineconeAdapter,
    WeaviateAdapter,
    ChromaAdapter,
)

__all__ = [
    "RAGProvider",
    "RAGContext",
    "RAGResult",
    "PineconeAdapter",
    "WeaviateAdapter", 
    "ChromaAdapter",
]
