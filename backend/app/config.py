from pydantic_settings import BaseSettings
from typing import List
import os


def get_database_url():
    """Convert DATABASE_URL to SQLAlchemy format if needed (for Render/Heroku)"""
    url = os.environ.get("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:15432/forgellm")
    # Render/Heroku use postgres:// but SQLAlchemy needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ForgeLLM"
    DEBUG: bool = os.environ.get("DEBUG", "true").lower() == "true"
    DEMO_MODE: bool = os.environ.get("DEMO_MODE", "true").lower() == "true"
    
    # Database (using psycopg v3 driver)
    DATABASE_URL: str = get_database_url()
    
    # Redis
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:16379/0")
    
    # JWT
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS - Allow all origins in production for simplicity
    CORS_ORIGINS: List[str] = ["*"]
    
    # Storage
    UPLOAD_DIR: str = "./data/uploads"
    MODELS_DIR: str = "./data/models"
    
    # ML Config
    BASE_MODEL: str = "mistralai/Mistral-7B-v0.1"
    MAX_TOKEN_LENGTH: int = 2048
    
    class Config:
        env_file = ".env"


settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
