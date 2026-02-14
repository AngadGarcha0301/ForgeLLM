from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ForgeLLM"
    DEBUG: bool = True
    
    # Database (using psycopg v3 driver)
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:15432/forgellm"
    
    # Redis
    REDIS_URL: str = "redis://localhost:16379/0"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:10801"]
    
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
