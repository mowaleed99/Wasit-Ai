import modal
from modal import App, Image, Secret, asgi_app, Volume

app = App("lost-found-ai")

# Define the volume for persistent storage
# If it doesn't exist, modal will create it automatically when requested like this (or you can create it via CLI)
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
        "onnxruntime"
    )
    # Required for opencv/insightface
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .add_local_dir("./app", remote_path="/root/app")
)

@app.function(
    image=image,
    secrets=[Secret.from_name("gemini-secret")],
    volumes={"/data": volume},
    min_containers=1
)
@asgi_app()
def fastapi_app():
    import os
    # Ensure data directory exists
    os.makedirs("/data", exist_ok=True)
    
    # Import routes and configure app
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

    web_app.include_router(router, prefix="/api")

    return web_app
