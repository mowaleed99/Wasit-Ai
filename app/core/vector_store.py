"""
FAISS vector store for embeddings using LlamaIndex for storage, direct FAISS for search.
"""
import faiss
import os
import json
from typing import List, Dict, Any, Optional

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode


class VectorStore:
    """FAISS-based vector store using LlamaIndex for add, direct FAISS for search."""
    
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index_path = index_path
        
        # Load existing FAISS index or create a new one
        if index_path and os.path.exists(f"{index_path}.faiss"):
            self.faiss_index = faiss.read_index(f"{index_path}.faiss")
        else:
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        
        # In-memory metadata store (FAISS does not store metadata natively)
        self.metadata_store = {}
        if index_path and os.path.exists(f"{index_path}_meta.json"):
            with open(f"{index_path}_meta.json", "r") as f:
                self.metadata_store = json.load(f)
    
    def add(self, embedding: Any, metadata: Dict[str, Any]) -> str:
        """Add an embedding with associated metadata to the store."""
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
            
        # Generate a node ID from metadata
        node_id = str(metadata.get("post_id", metadata.get("person_id", len(self.metadata_store))))
        
        node = TextNode(
            id_=node_id,
            text=metadata.get("text", "image_or_face_node"),
            metadata=metadata,
            embedding=embedding
        )
        self.vector_store.add([node])
        
        # Store metadata in memory
        self.metadata_store[node_id] = metadata
        return node_id
    
    def search(self, query_embedding: Any, k: int = 5) -> List[Dict[str, Any]]:
        """Search for the k nearest neighbors using FAISS directly."""
        if self.faiss_index.ntotal == 0:
            return []
        
        # Convert query to numpy array
        import numpy as np
        if hasattr(query_embedding, "tolist"):
            query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        elif isinstance(query_embedding, list):
            query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        else:
            query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search FAISS directly
        k = min(k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_vec, k)
        
        results = []
        # Build reverse mapping from FAISS ID to node_id
        id_to_node_id = {str(i): node_id for i, node_id in enumerate(self.metadata_store.keys())}
        
        for i in range(len(indices[0])):
            faiss_id = indices[0][i]
            score = float(distances[0][i])
            
            if faiss_id != -1:  # -1 means no result
                faiss_id_str = str(faiss_id)
                if faiss_id_str in id_to_node_id:
                    node_id = id_to_node_id[faiss_id_str]
                    meta = self.metadata_store.get(node_id, {})
                    results.append({
                        'post_id': meta.get('post_id', meta.get('person_id', node_id)),
                        'score': score,
                        'text': meta.get('text', ''),
                        'metadata': meta
                    })
        
        return results
    
    def save(self, path: str):
        """Persist FAISS index and metadata to disk."""
        faiss.write_index(self.faiss_index, f"{path}.faiss")
        with open(f"{path}_meta.json", "w") as f:
            json.dump(self.metadata_store, f)
    
    @property
    def count(self) -> int:
        """Return the number of vectors in the index."""
        return self.faiss_index.ntotal