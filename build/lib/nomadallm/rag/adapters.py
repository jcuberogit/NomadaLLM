"""
NomadaLLM RAG Adapters

Pre-built adapter templates for popular vector databases.
Developers copy and customize these for their setup.

IMPORTANT: These adapters require external dependencies that are NOT
included in NomadaLLM. Developers must install them separately:

    pip install pinecone-client  # For Pinecone
    pip install weaviate-client  # For Weaviate
    pip install chromadb         # For Chroma
"""

from typing import Any, Dict, List, Optional
from nomadallm.rag.base import RAGProvider, RAGContext, RAGResult


class PineconeAdapter(RAGProvider):
    """Adapter for Pinecone vector database.
    
    Usage:
        pip install pinecone-client
        
        from nomadallm.rag import PineconeAdapter
        
        rag = PineconeAdapter(
            api_key="your-api-key",
            index_name="your-index",
            environment="us-west1-gcp"
        )
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = "us-west1-gcp",
        embedding_model: str = "text-embedding-ada-002"
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self._embedding_model = embedding_model
        self._index = None
        self._embedder = None
    
    def _init_client(self):
        """Lazy init to avoid import errors if pinecone not installed."""
        if self._index is None:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.api_key)
                self._index = pc.Index(self.index_name)
            except ImportError:
                raise ImportError(
                    "Pinecone client not installed. "
                    "Install with: pip install pinecone-client"
                )
    
    async def query(
        self,
        text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        import time
        start = time.time()
        
        self._init_client()
        
        # Get embedding (implement your own or use OpenAI)
        embedding = await self._get_embedding(text)
        
        # Query Pinecone
        results = self._index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        contexts = [
            RAGContext(
                text=match.metadata.get("text", ""),
                score=match.score,
                metadata=match.metadata,
                source=match.metadata.get("source"),
                chunk_id=match.id
            )
            for match in results.matches
        ]
        
        return RAGResult(
            contexts=contexts,
            query=text,
            total_results=len(contexts),
            latency_ms=(time.time() - start) * 1000
        )
    
    async def index(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        self._init_client()
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Generate metadata if not provided
        if metadata is None:
            metadata = [{"text": t} for t in texts]
        else:
            for i, m in enumerate(metadata):
                m["text"] = texts[i]
        
        # Get embeddings
        embeddings = [await self._get_embedding(t) for t in texts]
        
        # Upsert to Pinecone
        vectors = [
            {"id": id, "values": emb, "metadata": meta}
            for id, emb, meta in zip(ids, embeddings, metadata)
        ]
        self._index.upsert(vectors=vectors)
        
        return len(texts)
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text. Override this with your embedder."""
        raise NotImplementedError(
            "Implement _get_embedding() or set self._embedder"
        )
    
    def get_embedding_model(self) -> str:
        return self._embedding_model
    
    def get_provider_name(self) -> str:
        return "Pinecone"


class WeaviateAdapter(RAGProvider):
    """Adapter for Weaviate vector database.
    
    Usage:
        pip install weaviate-client
        
        from nomadallm.rag import WeaviateAdapter
        
        rag = WeaviateAdapter(
            url="https://your-cluster.weaviate.network",
            api_key="your-api-key",
            class_name="Documents"
        )
    """
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        class_name: str = "Document"
    ):
        self.url = url
        self.api_key = api_key
        self.class_name = class_name
        self._client = None
    
    def _init_client(self):
        if self._client is None:
            try:
                import weaviate
                auth = weaviate.AuthApiKey(self.api_key) if self.api_key else None
                self._client = weaviate.Client(url=self.url, auth_client_secret=auth)
            except ImportError:
                raise ImportError(
                    "Weaviate client not installed. "
                    "Install with: pip install weaviate-client"
                )
    
    async def query(
        self,
        text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        import time
        start = time.time()
        
        self._init_client()
        
        query = (
            self._client.query
            .get(self.class_name, ["text", "source"])
            .with_near_text({"concepts": [text]})
            .with_limit(top_k)
            .with_additional(["certainty", "id"])
        )
        
        results = query.do()
        
        contexts = []
        if results.get("data", {}).get("Get", {}).get(self.class_name):
            for item in results["data"]["Get"][self.class_name]:
                contexts.append(RAGContext(
                    text=item.get("text", ""),
                    score=item.get("_additional", {}).get("certainty", 0),
                    source=item.get("source"),
                    chunk_id=item.get("_additional", {}).get("id")
                ))
        
        return RAGResult(
            contexts=contexts,
            query=text,
            total_results=len(contexts),
            latency_ms=(time.time() - start) * 1000
        )
    
    async def index(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        self._init_client()
        
        with self._client.batch as batch:
            for i, text in enumerate(texts):
                props = {"text": text}
                if metadata and i < len(metadata):
                    props.update(metadata[i])
                
                batch.add_data_object(props, self.class_name)
        
        return len(texts)
    
    def get_provider_name(self) -> str:
        return "Weaviate"


class ChromaAdapter(RAGProvider):
    """Adapter for Chroma vector database.
    
    Usage:
        pip install chromadb
        
        from nomadallm.rag import ChromaAdapter
        
        rag = ChromaAdapter(collection_name="my-docs")
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
    
    def _init_client(self):
        if self._client is None:
            try:
                import chromadb
                if self.persist_directory:
                    self._client = chromadb.PersistentClient(path=self.persist_directory)
                else:
                    self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(self.collection_name)
            except ImportError:
                raise ImportError(
                    "Chroma client not installed. "
                    "Install with: pip install chromadb"
                )
    
    async def query(
        self,
        text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> RAGResult:
        import time
        start = time.time()
        
        self._init_client()
        
        results = self._collection.query(
            query_texts=[text],
            n_results=top_k,
            where=filter
        )
        
        contexts = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                contexts.append(RAGContext(
                    text=doc,
                    score=1 - (results["distances"][0][i] if results["distances"] else 0),
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    chunk_id=results["ids"][0][i] if results["ids"] else None
                ))
        
        return RAGResult(
            contexts=contexts,
            query=text,
            total_results=len(contexts),
            latency_ms=(time.time() - start) * 1000
        )
    
    async def index(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        self._init_client()
        
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        
        self._collection.add(
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        
        return len(texts)
    
    def get_provider_name(self) -> str:
        return "Chroma"
