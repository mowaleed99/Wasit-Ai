"""
Multimodal search combining text and image embeddings.
"""
from typing import List, Dict, Any, Optional
from app.core.vector_store import VectorStore


class MultimodalMatcher:
    """Combine text and image search results with weighted scoring."""
    
    def __init__(self, text_store: VectorStore, image_store: VectorStore, text_weight: float = 0.5):
        self.text_store = text_store
        self.image_store = image_store
        self.text_weight = text_weight
        self.image_weight = 1.0 - text_weight
    
    def search(
        self, 
        text_embedding: Optional[Any] = None, 
        image_embedding: Optional[Any] = None, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform multimodal search combining text and image similarity.
        
        Args:
            text_embedding: Optional text embedding vector
            image_embedding: Optional image embedding vector
            k: Number of top results to return
            
        Returns:
            List of results with combined scores, sorted descending
        """
        # No input provided
        if text_embedding is None and image_embedding is None:
            return []
        
        # Text-only search
        if text_embedding is not None and image_embedding is None:
            return self.text_store.search(text_embedding, k)
        
        # Image-only search
        if image_embedding is not None and text_embedding is None:
            return self.image_store.search(image_embedding, k)
        
        # Both modalities available — search and combine
        text_results = self.text_store.search(text_embedding, k * 2)
        image_results = self.image_store.search(image_embedding, k * 2)
        
        # Combine scores by post_id using raw similarity scores
        combined = {}
        
        for res in text_results:
            pid = res.get('post_id') or res.get('metadata', {}).get('post_id')
            if pid:
                combined[pid] = {
                    'post_id': pid,
                    'score': res['score'] * self.text_weight,
                    'metadata': res.get('metadata', {})
                }
        
        for res in image_results:
            pid = res.get('post_id') or res.get('metadata', {}).get('post_id')
            if pid:
                if pid in combined:
                    combined[pid]['score'] += res['score'] * self.image_weight
                    combined[pid]['metadata'].update(res.get('metadata', {}))
                else:
                    combined[pid] = {
                        'post_id': pid,
                        'score': res['score'] * self.image_weight,
                        'metadata': res.get('metadata', {})
                    }
        
        # Sort by combined score and return top k
        results_list = list(combined.values())
        results_list.sort(key=lambda x: x['score'], reverse=True)
        return results_list[:k]