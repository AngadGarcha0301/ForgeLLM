"""
Tests for model endpoints.

Endpoints tested:
- GET /api/v1/models/
- GET /api/v1/models/{model_id}
- PATCH /api/v1/models/{model_id}
- DELETE /api/v1/models/{model_id}
"""
import pytest
from fastapi.testclient import TestClient


class TestListModels:
    """Tests for listing models."""
    
    def test_list_models_success(self, client: TestClient, test_workspace, test_model, auth_headers):
        """Test listing models in workspace."""
        response = client.get(
            f"/api/v1/models/?workspace_id={test_workspace.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(m["id"] == test_model.id for m in data)
    
    def test_list_models_empty(self, client: TestClient, db, test_user, auth_headers):
        """Test listing models in workspace with no models."""
        from app.db import models
        
        # Create empty workspace
        empty_workspace = models.Workspace(
            name="Empty Workspace",
            owner_id=test_user.id
        )
        db.add(empty_workspace)
        db.commit()
        
        response = client.get(
            f"/api/v1/models/?workspace_id={empty_workspace.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json() == []
    
    def test_list_models_wrong_workspace(self, client: TestClient, test_workspace, second_user_headers):
        """Test listing models in another user's workspace fails."""
        response = client.get(
            f"/api/v1/models/?workspace_id={test_workspace.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_list_models_unauthorized(self, client: TestClient, test_workspace):
        """Test listing models without authentication."""
        response = client.get(f"/api/v1/models/?workspace_id={test_workspace.id}")
        
        assert response.status_code == 401


class TestGetModel:
    """Tests for getting model details."""
    
    def test_get_model_success(self, client: TestClient, test_model, auth_headers):
        """Test getting model details."""
        response = client.get(
            f"/api/v1/models/{test_model.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_model.id
        assert data["name"] == test_model.name
        assert data["description"] == test_model.description
        assert data["base_model"] == test_model.base_model
        assert data["adapter_path"] == test_model.adapter_path
        assert "metrics" in data
        assert data["is_active"] is True
    
    def test_get_model_not_found(self, client: TestClient, auth_headers):
        """Test getting non-existent model."""
        response = client.get(
            "/api/v1/models/99999",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_get_model_unauthorized(self, client: TestClient, test_model):
        """Test getting model without authentication."""
        response = client.get(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 401
    
    def test_get_model_wrong_user(self, client: TestClient, test_model, second_user_headers):
        """Test getting another user's model fails."""
        response = client.get(
            f"/api/v1/models/{test_model.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404


class TestUpdateModel:
    """Tests for updating models."""
    
    def test_update_model_name(self, client: TestClient, test_model, auth_headers):
        """Test updating model name."""
        response = client.patch(
            f"/api/v1/models/{test_model.id}",
            json={"name": "Updated Model Name"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Model Name"
    
    def test_update_model_description(self, client: TestClient, test_model, auth_headers):
        """Test updating model description."""
        response = client.patch(
            f"/api/v1/models/{test_model.id}",
            json={"description": "Updated description for this model"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description for this model"
    
    def test_update_model_multiple_fields(self, client: TestClient, test_model, auth_headers):
        """Test updating multiple model fields."""
        response = client.patch(
            f"/api/v1/models/{test_model.id}",
            json={
                "name": "New Name",
                "description": "New Description"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"
        assert data["description"] == "New Description"
    
    def test_update_model_not_found(self, client: TestClient, auth_headers):
        """Test updating non-existent model."""
        response = client.patch(
            "/api/v1/models/99999",
            json={"name": "Test"},
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_update_model_unauthorized(self, client: TestClient, test_model):
        """Test updating model without authentication."""
        response = client.patch(
            f"/api/v1/models/{test_model.id}",
            json={"name": "Test"}
        )
        
        assert response.status_code == 401
    
    def test_update_model_wrong_user(self, client: TestClient, test_model, second_user_headers):
        """Test updating another user's model fails."""
        response = client.patch(
            f"/api/v1/models/{test_model.id}",
            json={"name": "Hacked!"},
            headers=second_user_headers
        )
        
        assert response.status_code == 404


class TestDeleteModel:
    """Tests for deleting models."""
    
    def test_delete_model_success(self, client: TestClient, test_model, auth_headers, db):
        """Test deleting a model."""
        model_id = test_model.id
        
        response = client.delete(
            f"/api/v1/models/{model_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()
        
        # Verify model is deleted
        from app.db import models
        deleted = db.query(models.Model).filter(
            models.Model.id == model_id
        ).first()
        assert deleted is None
    
    def test_delete_model_not_found(self, client: TestClient, auth_headers):
        """Test deleting non-existent model."""
        response = client.delete(
            "/api/v1/models/99999",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_delete_model_unauthorized(self, client: TestClient, test_model):
        """Test deleting model without authentication."""
        response = client.delete(f"/api/v1/models/{test_model.id}")
        
        assert response.status_code == 401
    
    def test_delete_model_wrong_user(self, client: TestClient, test_model, second_user_headers):
        """Test deleting another user's model fails."""
        response = client.delete(
            f"/api/v1/models/{test_model.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
