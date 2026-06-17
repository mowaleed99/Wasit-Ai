"""
Configuration management for the application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "models/gemini-embedding-001")
MULTIMODAL_EMBED_MODEL = os.getenv("MULTIMODAL_EMBED_MODEL", "models/gemini-1.5-flash")
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", 0.6))

# Vector dimensions (based on Gemini models)
TEXT_EMBED_DIM = 3072     # gemini-embedding-001 outputs 3072 dimensions
IMAGE_EMBED_DIM = 3072    # gemini-embedding-2-preview outputs 3072 dimensions
FACE_EMBED_DIM = 512      # ArcFace (InsightFace) outputs 512 dimensions

# FAISS settings
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_indexes")
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)