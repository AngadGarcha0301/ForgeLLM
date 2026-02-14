"""
Tests for security utilities and multi-tenant isolation.

Tests:
- Password hashing and verification
- JWT token creation and validation
- Multi-tenant data isolation
"""
import pytest
from datetime import timedelta


class TestPasswordSecurity:
    """Tests for password hashing."""
    
    def test_hash_password(self):
        """Test password hashing."""
        from app.core.security import get_password_hash
        
        password = "mysecretpassword"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 0
        # bcrypt hashes start with $2b$
        assert hashed.startswith("$2b$")
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        from app.core.security import get_password_hash, verify_password
        
        password = "mysecretpassword"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        from app.core.security import get_password_hash, verify_password
        
        password = "mysecretpassword"
        hashed = get_password_hash(password)
        
        assert verify_password("wrongpassword", hashed) is False
    
    def test_hash_is_unique(self):
        """Test that same password produces different hashes (due to salt)."""
        from app.core.security import get_password_hash
        
        password = "samepassword"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        assert hash1 != hash2  # Different salts


class TestJWTTokens:
    """Tests for JWT token handling."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        from app.core.security import create_access_token
        
        token = create_access_token(
            data={"sub": "123"},
            expires_delta=timedelta(minutes=30)
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        # JWT has 3 parts separated by dots
        assert token.count(".") == 2
    
    def test_token_contains_user_id(self):
        """Test that token payload contains user ID."""
        from app.core.security import create_access_token
        from jose import jwt
        from app.config import settings
        
        user_id = "456"
        token = create_access_token(
            data={"sub": user_id},
            expires_delta=timedelta(minutes=30)
        )
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert payload["sub"] == user_id
    
    def test_token_has_expiration(self):
        """Test that token has expiration claim."""
        from app.core.security import create_access_token
        from jose import jwt
        from app.config import settings
        
        token = create_access_token(
            data={"sub": "123"},
            expires_delta=timedelta(minutes=30)
        )
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert "exp" in payload
    
    def test_expired_token_raises(self):
        """Test that expired token raises exception."""
        from app.core.security import create_access_token
        from jose import jwt, ExpiredSignatureError
        from app.config import settings
        
        # Create token that's already expired
        token = create_access_token(
            data={"sub": "123"},
            expires_delta=timedelta(minutes=-10)
        )
        
        with pytest.raises(ExpiredSignatureError):
            jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


class TestMultiTenantIsolation:
    """Tests for multi-tenant data isolation."""
    
    def test_user_cannot_access_other_users_workspace(
        self, client, test_workspace, second_user_headers
    ):
        """Test that user cannot access another user's workspace."""
        response = client.get(
            f"/api/v1/datasets/?workspace_id={test_workspace.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_user_cannot_see_other_users_datasets(
        self, client, test_dataset, second_user_headers
    ):
        """Test that user cannot see another user's datasets."""
        response = client.get(
            f"/api/v1/datasets/{test_dataset.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_user_cannot_see_other_users_training_jobs(
        self, client, test_training_job, second_user_headers
    ):
        """Test that user cannot see another user's training jobs."""
        response = client.get(
            f"/api/v1/training/{test_training_job.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_user_cannot_see_other_users_models(
        self, client, test_model, second_user_headers
    ):
        """Test that user cannot see another user's models."""
        response = client.get(
            f"/api/v1/models/{test_model.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_user_cannot_modify_other_users_resources(
        self, client, test_model, second_user_headers
    ):
        """Test that user cannot modify another user's resources."""
        response = client.patch(
            f"/api/v1/models/{test_model.id}",
            json={"name": "Hacked!"},
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_user_cannot_delete_other_users_resources(
        self, client, test_dataset, second_user_headers
    ):
        """Test that user cannot delete another user's resources."""
        response = client.delete(
            f"/api/v1/datasets/{test_dataset.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_users_see_only_their_own_data(
        self, client, db, test_user, test_workspace, test_dataset, second_user, auth_headers, second_user_headers
    ):
        """Test that each user only sees their own data."""
        from app.db import models
        
        # Create workspace and dataset for second user
        second_workspace = models.Workspace(
            name="Second User Workspace",
            owner_id=second_user.id
        )
        db.add(second_workspace)
        db.commit()
        db.refresh(second_workspace)
        
        second_dataset = models.Dataset(
            name="second_user_dataset.jsonl",
            workspace_id=second_workspace.id,
            file_path="/data/uploads/second.jsonl",
            status="ready"
        )
        db.add(second_dataset)
        db.commit()
        
        # First user should see only their dataset
        response1 = client.get(
            f"/api/v1/datasets/?workspace_id={test_workspace.id}",
            headers=auth_headers
        )
        assert response1.status_code == 200
        datasets1 = response1.json()
        assert all(d["workspace_id"] == test_workspace.id for d in datasets1)
        
        # Second user should see only their dataset
        response2 = client.get(
            f"/api/v1/datasets/?workspace_id={second_workspace.id}",
            headers=second_user_headers
        )
        assert response2.status_code == 200
        datasets2 = response2.json()
        assert all(d["workspace_id"] == second_workspace.id for d in datasets2)
        
        # Datasets should not overlap
        dataset_ids_1 = {d["id"] for d in datasets1}
        dataset_ids_2 = {d["id"] for d in datasets2}
        assert dataset_ids_1.isdisjoint(dataset_ids_2)
