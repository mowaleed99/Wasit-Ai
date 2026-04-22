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
    def __init__(self):
        # Configure raw genai for image embeddings if LlamaIndex lacks direct image embedding support
        genai.configure(api_key=GEMINI_API_KEY)
        
        # LlamaIndex text embedder
        self.text_embedder = GeminiEmbedding(
            model_name=TEXT_EMBED_MODEL, 
            api_key=GEMINI_API_KEY,
            task_type="retrieval_document"
        )
        self.multimodal_model = MULTIMODAL_EMBED_MODEL

    def _normalize(self, embedding: List[float]) -> List[float]:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist()
        return (vec / norm).tolist()

    async def get_text_embedding_async(self, text: str) -> List[float]:
        """نص فقط - باستخدام LlamaIndex"""
        embedding = await self.text_embedder.aget_text_embedding(text)
        return self._normalize(embedding)

    def get_text_embedding(self, text: str) -> List[float]:
        """نص فقط - باستخدام LlamaIndex"""
        embedding = self.text_embedder.get_text_embedding(text)
        return self._normalize(embedding)

    def get_image_embedding(self, image_bytes: bytes) -> List[float]:
        """صورة فقط - باستخدام genai مباشرة (حتى يدعم LlamaIndex الصور بشكل كامل)"""
        image = PIL.Image.open(io.BytesIO(image_bytes))
        result = genai.embed_content(
            model=self.multimodal_model,
            content=image,
            task_type="retrieval_document"
        )
        embedding = result['embedding'] if isinstance(result, dict) else result
        return self._normalize(embedding)