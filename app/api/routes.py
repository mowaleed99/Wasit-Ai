"""
API routes for embeddings, vector storage, and search endpoints.
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, List
import numpy as np
import os

from app.core.embeddings import CustomGeminiEmbedder
from app.core.vector_store import VectorStore
from app.core.face_matcher import FaceRecognizer
from app.core.multimodal import MultimodalMatcher
from app.config import TEXT_EMBED_DIM, IMAGE_EMBED_DIM, FACE_EMBED_DIM, FAISS_INDEX_DIR
from app.models.schemas import (
    SearchResponse, 
    AddPostResponse, 
    HealthResponse,
    ErrorResponse
)

router = APIRouter()

# Global components — initialized lazily at startup
embedder: Optional[CustomGeminiEmbedder] = None
text_store: Optional[VectorStore] = None
image_store: Optional[VectorStore] = None
face_store: Optional[VectorStore] = None
face_recognizer: Optional[FaceRecognizer] = None
multimodal_matcher: Optional[MultimodalMatcher] = None


def get_embedder() -> CustomGeminiEmbedder:
    """Get or initialize the embedder (lazy loading fallback)."""
    global embedder
    if embedder is None:
        embedder = CustomGeminiEmbedder()
    return embedder


def get_stores():
    """Get or initialize all vector stores."""
    global text_store, image_store, face_store
    if text_store is None:
        text_store = VectorStore(TEXT_EMBED_DIM, index_path=os.path.join(FAISS_INDEX_DIR, "text"))
    if image_store is None:
        image_store = VectorStore(IMAGE_EMBED_DIM, index_path=os.path.join(FAISS_INDEX_DIR, "image"))
    if face_store is None:
        face_store = VectorStore(FACE_EMBED_DIM, index_path=os.path.join(FAISS_INDEX_DIR, "face"))
    return text_store, image_store, face_store


def get_face_recognizer() -> Optional[FaceRecognizer]:
    """Get or initialize the face recognizer."""
    global face_recognizer
    if face_recognizer is None:
        try:
            face_recognizer = FaceRecognizer()
        except Exception as e:
            print(f"Warning: Failed to initialize FaceRecognizer: {e}")
            face_recognizer = None
    return face_recognizer


def get_multimodal_matcher() -> MultimodalMatcher:
    """Get or initialize the multimodal matcher."""
    global multimodal_matcher
    if multimodal_matcher is None:
        ts, ims, _ = get_stores()
        multimodal_matcher = MultimodalMatcher(ts, ims)
    return multimodal_matcher


@router.on_event("startup")
async def startup_event():
    """Pre-load all models and vector stores at startup."""
    print("Starting up Wasit AI server...")
    try:
        get_embedder()
        print("✅ Embedder loaded")
    except Exception as e:
        print(f"❌ Failed to load embedder: {e}")
    
    try:
        get_stores()
        print("✅ Vector stores loaded")
    except Exception as e:
        print(f"❌ Failed to load vector stores: {e}")
    
    try:
        get_face_recognizer()
        print("✅ Face recognizer loaded")
    except Exception as e:
        print(f"❌ Failed to load face recognizer: {e}")
    
    try:
        get_multimodal_matcher()
        print("✅ Multimodal matcher loaded")
    except Exception as e:
        print(f"❌ Failed to load multimodal matcher: {e}")
    
    print("Wasit AI server is ready!")


# ========== Embedding Endpoints ==========

@router.post("/text-embedding", response_model=SearchResponse)
async def get_text_embedding(text: str = Form(...)):
    """Generate embedding vector for a given text."""
    try:
        emb = get_embedder()
        embedding = emb.get_text_embedding(text)
        return SearchResponse(
            status="success",
            data={"embedding": embedding}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image-embedding", response_model=SearchResponse)
async def get_image_embedding(image: UploadFile = File(...)):
    """Generate embedding vector for an uploaded image."""
    try:
        image_bytes = await image.read()
        emb = get_embedder()
        embedding = emb.get_image_embedding(image_bytes)
        return SearchResponse(
            status="success",
            data={"embedding": embedding}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Add Endpoints ==========

@router.post("/add-text", response_model=AddPostResponse)
async def add_text(text: str = Form(...), post_id: str = Form(...)):
    """Add a text post to the vector store."""
    try:
        emb = get_embedder()
        ts, _, _ = get_stores()
        embedding = emb.get_text_embedding(text)
        ts.add(np.array(embedding, dtype=np.float32), {"post_id": post_id, "text": text})
        ts.save(os.path.join(FAISS_INDEX_DIR, "text"))
        return AddPostResponse(
            status="success",
            data={"message": "Text added successfully", "post_id": post_id}
        )
    except Exception as e:
        print(f"ERROR in add-text: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-image", response_model=AddPostResponse)
async def add_image(post_id: str = Form(...), image: UploadFile = File(...)):
    """Add an image post to the vector store."""
    try:
        image_bytes = await image.read()
        emb = get_embedder()
        _, ims, _ = get_stores()
        embedding = emb.get_image_embedding(image_bytes)
        ims.add(np.array(embedding, dtype=np.float32), {"post_id": post_id})
        ims.save(os.path.join(FAISS_INDEX_DIR, "image"))
        return AddPostResponse(
            status="success",
            data={"message": "Image added successfully", "post_id": post_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-face", response_model=AddPostResponse)
async def add_face(person_id: str = Form(...), image: UploadFile = File(...)):
    """Add a face embedding to the face store."""
    recognizer = get_face_recognizer()
    if not recognizer:
        raise HTTPException(status_code=500, detail="Face recognizer is not available")
    try:
        image_bytes = await image.read()
        embedding = recognizer.get_face_embedding(image_bytes)
        if embedding is None:
            return AddPostResponse(
                status="error",
                data={"error": "No face detected in the image", "person_id": person_id}
            )
        _, _, fs = get_stores()
        fs.add(embedding, {"person_id": person_id})
        fs.save(os.path.join(FAISS_INDEX_DIR, "face"))
        return AddPostResponse(
            status="success",
            data={"message": "Face added successfully", "person_id": person_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Search Endpoints ==========

@router.post("/search-text", response_model=SearchResponse)
async def search_text(text: str = Form(...), k: int = Form(5)):
    """Search for similar text posts."""
    try:
        emb = get_embedder()
        ts, _, _ = get_stores()
        embedding = emb.get_text_embedding(text)
        results = ts.search(np.array(embedding, dtype=np.float32), k)
        return SearchResponse(
            status="success",
            data={"results": results, "count": len(results)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-image", response_model=SearchResponse)
async def search_image(k: int = Form(5), image: UploadFile = File(...)):
    """Search for similar image posts."""
    try:
        image_bytes = await image.read()
        emb = get_embedder()
        _, ims, _ = get_stores()
        embedding = emb.get_image_embedding(image_bytes)
        results = ims.search(np.array(embedding, dtype=np.float32), k)
        return SearchResponse(
            status="success",
            data={"results": results, "count": len(results)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/face-match", response_model=SearchResponse)
async def face_match(k: int = Form(5), image: UploadFile = File(...)):
    """Search for matching faces."""
    recognizer = get_face_recognizer()
    if not recognizer:
        raise HTTPException(status_code=500, detail="Face recognizer is not available")
    try:
        image_bytes = await image.read()
        embedding = recognizer.get_face_embedding(image_bytes)
        if embedding is None:
            return SearchResponse(
                status="success",
                data={"results": [], "count": 0, "message": "No face detected"}
            )
        _, _, fs = get_stores()
        results = fs.search(embedding, k)
        return SearchResponse(
            status="success",
            data={"results": results, "count": len(results)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal-search", response_model=SearchResponse)
async def multimodal_search(
    text: Optional[str] = Form(None), 
    image: Optional[UploadFile] = File(None),
    k: int = Form(5)
):
    """Combined text and image search with weighted scoring."""
    try:
        if not text and not image:
            raise HTTPException(status_code=400, detail="Must provide at least text or image")
        
        text_emb = None
        if text:
            emb = get_embedder()
            text_emb = np.array(emb.get_text_embedding(text), dtype=np.float32)
        
        image_emb = None
        if image:
            image_bytes = await image.read()
            emb = get_embedder()
            image_emb = np.array(emb.get_image_embedding(image_bytes), dtype=np.float32)
        
        matcher = get_multimodal_matcher()
        results = matcher.search(text_emb, image_emb, k)
        return SearchResponse(
            status="success",
            data={"results": results, "count": len(results)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Health Check ==========

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with component status."""
    recognizer = get_face_recognizer()
    ts, ims, fs = get_stores()
    
    return HealthResponse(
        status="ok",
        data={
            "message": "Server is running",
            "components": {
                "embedder": "loaded" if embedder else "not loaded",
                "text_store": f"loaded ({ts.count} vectors)" if ts else "not loaded",
                "image_store": f"loaded ({ims.count} vectors)" if ims else "not loaded",
                "face_store": f"loaded ({fs.count} vectors)" if fs else "not loaded",
                "face_recognizer": "loaded" if recognizer else "not loaded",
                "multimodal_matcher": "loaded" if multimodal_matcher else "not loaded"
            }
        }
    )