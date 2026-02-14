from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, datasets, training, models, inference, workspaces
from app.config import settings
from app.db.database import init_db

app = FastAPI(
    title="ForgeLLM",
    description="Fine-tune LLMs with LoRA - SaaS Platform",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for cloud deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(workspaces.router, prefix="/api/v1/workspaces", tags=["workspaces"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["inference"])


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    init_db()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ForgeLLM"}


@app.get("/")
async def root():
    return {
        "message": "Welcome to ForgeLLM API",
        "docs": "/docs",
        "version": "0.1.0"
    }
