"""
Core capabilities for interaction with Google Gemini models using LlamaIndex.
"""
import io
import numpy as np
from typing import List, Any
import PIL.Image

from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai

from app.config import GEMINI_API_KEY, TEXT_EMBED_MODEL, MULTIMODAL_EMBED_MODEL


class CustomGeminiEmbedder:
    """Handles text and image embedding generation using Google Gemini models."""
    
    def __init__(self):
        # Configure the raw genai SDK for image embeddings
        genai.configure(api_key=GEMINI_API_KEY)
        
        # LlamaIndex wrapper for text embeddings
        self.text_embedder = GeminiEmbedding(
            model_name=TEXT_EMBED_MODEL, 
            api_key=GEMINI_API_KEY,
            task_type="retrieval_document"
        )
        self.multimodal_model = MULTIMODAL_EMBED_MODEL

    def _normalize(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()

    async def get_text_embedding_async(self, text: str) -> List[float]:
        """Generate text embedding asynchronously using LlamaIndex."""
        try:
            embedding = await self.text_embedder.aget_text_embedding(text)
            return self._normalize(embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embedding: {str(e)}")

    def get_text_embedding(self, text: str) -> List[float]:
        """Generate text embedding synchronously using LlamaIndex."""
        try:
            embedding = self.text_embedder.get_text_embedding(text)
            return self._normalize(embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embedding: {str(e)}")

    def get_image_embedding(self, image_bytes: bytes) -> List[float]:
        """Generate image embedding using the raw genai SDK."""
        try:
            image = PIL.Image.open(io.BytesIO(image_bytes))
            result = genai.embed_content(
                model=self.multimodal_model,
                content=image,
                task_type="retrieval_document"
            )
            embedding = result['embedding'] if isinstance(result, dict) else result
            return self._normalize(embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image embedding: {str(e)}")