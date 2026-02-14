"""
Tests for authentication endpoints.

Endpoints tested:
- POST /api/v1/auth/register
- POST /api/v1/auth/login
- GET /api/v1/auth/me
"""
import pytest
from fastapi.testclient import TestClient


class TestRegister:
    """Tests for user registration."""
    
    def test_register_success(self, client: TestClient):
        """Test successful user registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "securepass123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "id" in data
        assert data["is_active"] is True
        assert "created_at" in data
        # Password should not be returned
        assert "password" not in data
        assert "hashed_password" not in data
    
    def test_register_duplicate_email(self, client: TestClient, test_user):
        """Test registration with existing email fails."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",  # Same as test_user
                "username": "differentuser",
                "password": "securepass123"
            }
        )
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()
    
    def test_register_invalid_email(self, client: TestClient):
        """Test registration with invalid email format."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "invalid-email",
                "username": "newuser",
                "password": "securepass123"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_register_missing_fields(self, client: TestClient):
        """Test registration with missing required fields."""
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "test@example.com"}
        )
        
        assert response.status_code == 422
    
    def test_register_creates_default_workspace(self, client: TestClient, db):
        """Test that registration creates a default workspace."""
        from app.db import models
        
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "workspace@example.com",
                "username": "workspaceuser",
                "password": "securepass123"
            }
        )
        
        assert response.status_code == 200
        user_id = response.json()["id"]
        
        # Check workspace was created
        workspace = db.query(models.Workspace).filter(
            models.Workspace.owner_id == user_id
        ).first()
        
        assert workspace is not None
        assert workspace.name == "Default Workspace"


class TestLogin:
    """Tests for user login."""
    
    def test_login_with_email_success(self, client: TestClient, test_user):
        """Test successful login with email."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "test@example.com",
                "password": "testpass123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_with_username_success(self, client: TestClient, test_user):
        """Test successful login with username."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "testuser",
                "password": "testpass123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_wrong_password(self, client: TestClient, test_user):
        """Test login with incorrect password."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "test@example.com",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()
    
    def test_login_nonexistent_user(self, client: TestClient):
        """Test login with non-existent user."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent@example.com",
                "password": "somepassword"
            }
        )
        
        assert response.status_code == 401
    
    def test_login_missing_credentials(self, client: TestClient):
        """Test login with missing credentials."""
        response = client.post(
            "/api/v1/auth/login",
            data={}
        )
        
        assert response.status_code == 422


class TestGetCurrentUser:
    """Tests for getting current user info."""
    
    def test_get_me_success(self, client: TestClient, test_user, auth_headers):
        """Test getting current user info with valid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert data["id"] == test_user.id
    
    def test_get_me_no_token(self, client: TestClient):
        """Test getting current user info without token."""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
    
    def test_get_me_invalid_token(self, client: TestClient):
        """Test getting current user info with invalid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401
    
    def test_get_me_expired_token(self, client: TestClient, test_user):
        """Test getting current user info with expired token."""
        from app.core.security import create_access_token
        from datetime import timedelta
        
        # Create expired token
        expired_token = create_access_token(
            data={"sub": str(test_user.id)},
            expires_delta=timedelta(minutes=-10)  # Expired 10 minutes ago
        )
        
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        
        assert response.status_code == 401
