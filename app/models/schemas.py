"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ========== Request Schemas ==========

class TextEmbeddingRequest(BaseModel):
    """Request schema for generating text embeddings."""
    text: str = Field(..., description="The text to embed")


class AddTextRequest(BaseModel):
    """Request schema for adding a text post to the vector store."""
    text: str = Field(..., description="The text content of the post")
    post_id: str = Field(..., description="Unique identifier for the post")


class AddImageRequest(BaseModel):
    """Request schema for adding an image post (image sent as UploadFile)."""
    post_id: str = Field(..., description="Unique identifier for the post")


class AddFaceRequest(BaseModel):
    """Request schema for adding a face to the face store."""
    person_id: str = Field(..., description="Unique identifier for the person")


class SearchTextRequest(BaseModel):
    """Request schema for text-only search."""
    text: str = Field(..., description="The search query text")
    k: int = Field(5, ge=1, le=50, description="Number of results to return")


class SearchImageRequest(BaseModel):
    """Request schema for image-only search (image sent as UploadFile)."""
    k: int = Field(5, ge=1, le=50, description="Number of results to return")


class FaceMatchRequest(BaseModel):
    """Request schema for face matching (image sent as UploadFile)."""
    k: int = Field(5, ge=1, le=50, description="Number of results to return")


class MultimodalSearchRequest(BaseModel):
    """Request schema for combined text+image search."""
    text: Optional[str] = Field(None, description="Optional search query text")
    k: int = Field(5, ge=1, le=50, description="Number of results to return")


# ========== Response Schemas ==========

class SearchResult(BaseModel):
    """Schema for a single search result."""
    post_id: str = Field(..., description="Post or person identifier")
    score: float = Field(..., description="Similarity score (0-1 range)")
    text: Optional[str] = Field(None, description="Text content if available")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SearchResponse(BaseModel):
    """Standard success response for all search endpoints."""
    status: str = Field("success", description="Response status")
    data: Dict[str, Any] = Field(..., description="Response data containing results")


class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = Field("error", description="Response status")
    detail: str = Field(..., description="Error description")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field("ok", description="Server status")
    data: Dict[str, Any] = Field(..., description="Health information")


class AddPostResponse(BaseModel):
    """Response schema for add post/image/face endpoints."""
    status: str = Field("success", description="Response status")
    data: Dict[str, Any] = Field(..., description="Response data with post_id and message")