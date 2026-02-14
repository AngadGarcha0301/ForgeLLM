"""
Pytest fixtures and configuration for ForgeLLM backend tests.
"""
import pytest
from typing import Generator, Dict, Any
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base
from app.dependencies import get_db
from app.core.security import get_password_hash, create_access_token
from app.db import models
from datetime import timedelta


# Test database - in-memory SQLite
SQLALCHEMY_DATABASE_URL = "sqlite://"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def db() -> Generator:
    """Create a fresh database for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db) -> Generator:
    """Create a test client with database override."""
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(db) -> models.User:
    """Create a test user."""
    user = models.User(
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpass123")
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_workspace(db, test_user) -> models.Workspace:
    """Create a test workspace."""
    workspace = models.Workspace(
        name="Test Workspace",
        description="A workspace for testing",
        owner_id=test_user.id
    )
    db.add(workspace)
    db.commit()
    db.refresh(workspace)
    return workspace


@pytest.fixture
def test_dataset(db, test_workspace) -> models.Dataset:
    """Create a test dataset."""
    dataset = models.Dataset(
        name="test_dataset.jsonl",
        workspace_id=test_workspace.id,
        file_path="/data/uploads/test_dataset.jsonl",
        file_size=1024,
        token_count=500,
        sample_count=10,
        format="jsonl",
        status="ready"
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@pytest.fixture
def test_training_job(db, test_workspace, test_dataset) -> models.TrainingJob:
    """Create a test training job."""
    job = models.TrainingJob(
        workspace_id=test_workspace.id,
        dataset_id=test_dataset.id,
        name="Test Training Job",
        base_model="mistralai/Mistral-7B-v0.1",
        status="pending",
        progress=0.0,
        current_step=0,
        lora_r=16,
        lora_alpha=32
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


@pytest.fixture
def test_model(db, test_workspace) -> models.Model:
    """Create a test trained model (without training job dependency)."""
    model = models.Model(
        name="test-adapter-v1",
        description="A test LoRA adapter",
        workspace_id=test_workspace.id,
        training_job_id=None,  # Can be null
        adapter_path="/data/models/test-adapter",
        base_model="mistralai/Mistral-7B-v0.1",
        metrics={"loss": 0.5, "eval_loss": 0.6},
        is_active=True
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@pytest.fixture
def auth_headers(test_user) -> Dict[str, str]:
    """Get authorization headers for test user."""
    access_token = create_access_token(
        data={"sub": str(test_user.id)},
        expires_delta=timedelta(minutes=30)
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def second_user(db) -> models.User:
    """Create a second test user (for isolation tests)."""
    user = models.User(
        email="other@example.com",
        username="otheruser",
        hashed_password=get_password_hash("otherpass123")
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def second_user_headers(second_user) -> Dict[str, str]:
    """Get authorization headers for second user."""
    access_token = create_access_token(
        data={"sub": str(second_user.id)},
        expires_delta=timedelta(minutes=30)
    )
    return {"Authorization": f"Bearer {access_token}"}
