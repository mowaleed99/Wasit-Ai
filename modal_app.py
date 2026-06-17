"""
Modal deployment configuration for Lost & Found AI.
"""
import modal
from modal import App, Image, Secret, asgi_app, Volume

app = App("lost-found-ai")

# Persistent volume for FAISS indexes (survives container restarts)
volume = Volume.from_name("lost-found-models-volume", create_if_missing=True)

# Define the container image with all dependencies
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi",
        "uvicorn",
        "google-generativeai",
        "faiss-cpu",
        "numpy",
        "pillow",
        "python-dotenv",
        "python-multipart",
        "llama-index",
        "llama-index-vector-stores-faiss",
        "llama-index-embeddings-gemini",
        "insightface",
        "onnxruntime",
        "opencv-python-headless"
    )
    # System libraries required for OpenCV and InsightFace
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .add_local_dir("./app", remote_path="/root/app")
)


@app.function(
    image=image,
    secrets=[Secret.from_name("gemini-secret")],
    volumes={"/data": volume},
    min_containers=0,  # Set to 0 for dev (saves cost), set to 1 for production (no cold start)
    allow_concurrent_inputs=10
)
@asgi_app()
def fastapi_app():
    import os
    
    # Use persistent volume for FAISS indexes so data survives restarts
    os.environ["FAISS_INDEX_DIR"] = "/data/faiss_indexes"
    os.makedirs("/data/faiss_indexes", exist_ok=True)
    
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from app.api.routes import router

    web_app = FastAPI(title="Lost & Found AI")

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all API routes under /api prefix
    web_app.include_router(router, prefix="/api")

    # Root health check (same as local version)
    @web_app.get("/")
    async def root():
        return {"message": "Wasit AI server is running"}

    return web_app