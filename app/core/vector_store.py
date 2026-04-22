"""
FAISS vector store for embeddings using LlamaIndex.
"""
import faiss
import os
import json
from typing import List, Dict, Any, Optional

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

class VectorStore:
    """FAISS-based vector store using LlamaIndex."""
    
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = index_path
        
        # Load or create FAISS index
        if index_path and os.path.exists(f"{index_path}.faiss"):
            self.faiss_index = faiss.read_index(f"{index_path}.faiss")
        else:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        
        # In-memory document storage for metadata (since FAISS doesn't store metadata natively)
        self.metadata_store = {}
        if index_path and os.path.exists(f"{index_path}_meta.json"):
            with open(f"{index_path}_meta.json", "r") as f:
                self.metadata_store = json.load(f)
    
    def add(self, embedding: Any, metadata: Dict[str, Any]) -> str:
        """Add embedding using LlamaIndex constructs."""
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
            
        # Extract an ID for the node
        node_id = str(metadata.get("post_id", metadata.get("person_id", len(self.metadata_store))))
        
        node = TextNode(
            id_=node_id,
            text=metadata.get("text", "image_or_face_node"),
            metadata=metadata,
            embedding=embedding
        )
        self.vector_store.add([node])
        
        # Store metadata
        self.metadata_store[node_id] = metadata
        return node_id
    
    def search(self, query_embedding: Any, k: int = 5) -> List[Dict[str, Any]]:
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
            
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=k
        )
        result = self.vector_store.query(query)
        
        results = []
        if result.ids and result.similarities:
            for node_id, similarity in zip(result.ids, result.similarities):
                if node_id in self.metadata_store:
                    results.append({
                        'score': float(similarity),
                        'metadata': self.metadata_store[node_id]
                    })
        return results
    
    def save(self, path: str):
        faiss.write_index(self.faiss_index, f"{path}.faiss")
        with open(f"{path}_meta.json", "w") as f:
            json.dump(self.metadata_store, f)