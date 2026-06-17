from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

# Initialize FastAPI app with title and version
app = FastAPI(title="Wasit AI", version="1.0.0")

# Allow cross-origin requests from any frontend (Flutter, web, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes from the routes file under the /api prefix
app.include_router(api_router, prefix="/api")

# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Wasit AI server is running"}