"""
Tests for dataset endpoints.

Endpoints tested:
- POST /api/v1/datasets/upload
- GET /api/v1/datasets/{dataset_id}
- GET /api/v1/datasets/
- DELETE /api/v1/datasets/{dataset_id}
"""
import pytest
from io import BytesIO
from fastapi.testclient import TestClient


class TestUploadDataset:
    """Tests for dataset upload."""
    
    def test_upload_dataset_jsonl(self, client: TestClient, test_workspace, auth_headers):
        """Test uploading a JSONL dataset."""
        content = b'{"instruction": "Hello", "input": "", "output": "Hi there!"}\n'
        content += b'{"instruction": "Bye", "input": "", "output": "Goodbye!"}\n'
        
        response = client.post(
            f"/api/v1/datasets/upload?workspace_id={test_workspace.id}",
            files={"file": ("train.jsonl", BytesIO(content), "application/jsonl")},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["workspace_id"] == test_workspace.id
        assert "id" in data
        assert data["status"] in ["processing", "ready"]
    
    def test_upload_dataset_json(self, client: TestClient, test_workspace, auth_headers):
        """Test uploading a JSON dataset."""
        content = b'[{"instruction": "Test", "input": "", "output": "Response"}]'
        
        response = client.post(
            f"/api/v1/datasets/upload?workspace_id={test_workspace.id}",
            files={"file": ("train.json", BytesIO(content), "application/json")},
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    def test_upload_dataset_unauthorized(self, client: TestClient, test_workspace):
        """Test uploading dataset without authentication."""
        content = b'{"instruction": "Test", "input": "", "output": "Response"}\n'
        
        response = client.post(
            f"/api/v1/datasets/upload?workspace_id={test_workspace.id}",
            files={"file": ("train.jsonl", BytesIO(content), "application/jsonl")}
        )
        
        assert response.status_code == 401
    
    def test_upload_dataset_wrong_workspace(self, client: TestClient, test_workspace, second_user_headers):
        """Test uploading to another user's workspace fails."""
        content = b'{"instruction": "Test", "input": "", "output": "Response"}\n'
        
        response = client.post(
            f"/api/v1/datasets/upload?workspace_id={test_workspace.id}",
            files={"file": ("train.jsonl", BytesIO(content), "application/jsonl")},
            headers=second_user_headers
        )
        
        assert response.status_code == 404
        assert "workspace" in response.json()["detail"].lower()
    
    def test_upload_dataset_nonexistent_workspace(self, client: TestClient, auth_headers):
        """Test uploading to non-existent workspace."""
        content = b'{"instruction": "Test", "input": "", "output": "Response"}\n'
        
        response = client.post(
            "/api/v1/datasets/upload?workspace_id=99999",
            files={"file": ("train.jsonl", BytesIO(content), "application/jsonl")},
            headers=auth_headers
        )
        
        assert response.status_code == 404


class TestGetDataset:
    """Tests for getting dataset details."""
    
    def test_get_dataset_success(self, client: TestClient, test_dataset, auth_headers):
        """Test getting dataset details."""
        response = client.get(
            f"/api/v1/datasets/{test_dataset.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_dataset.id
        assert data["name"] == test_dataset.name
        assert data["workspace_id"] == test_dataset.workspace_id
        assert "file_path" in data
        assert "token_count" in data
        assert "sample_count" in data
    
    def test_get_dataset_not_found(self, client: TestClient, auth_headers):
        """Test getting non-existent dataset."""
        response = client.get(
            "/api/v1/datasets/99999",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_get_dataset_unauthorized(self, client: TestClient, test_dataset):
        """Test getting dataset without authentication."""
        response = client.get(f"/api/v1/datasets/{test_dataset.id}")
        
        assert response.status_code == 401
    
    def test_get_dataset_wrong_user(self, client: TestClient, test_dataset, second_user_headers):
        """Test getting another user's dataset fails."""
        response = client.get(
            f"/api/v1/datasets/{test_dataset.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404


class TestListDatasets:
    """Tests for listing datasets."""
    
    def test_list_datasets_success(self, client: TestClient, test_workspace, test_dataset, auth_headers):
        """Test listing datasets in workspace."""
        response = client.get(
            f"/api/v1/datasets/?workspace_id={test_workspace.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(d["id"] == test_dataset.id for d in data)
    
    def test_list_datasets_empty(self, client: TestClient, db, test_user, auth_headers):
        """Test listing datasets in empty workspace."""
        from app.db import models
        
        # Create empty workspace
        empty_workspace = models.Workspace(
            name="Empty Workspace",
            owner_id=test_user.id
        )
        db.add(empty_workspace)
        db.commit()
        
        response = client.get(
            f"/api/v1/datasets/?workspace_id={empty_workspace.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json() == []
    
    def test_list_datasets_wrong_workspace(self, client: TestClient, test_workspace, second_user_headers):
        """Test listing datasets in another user's workspace fails."""
        response = client.get(
            f"/api/v1/datasets/?workspace_id={test_workspace.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404


class TestDeleteDataset:
    """Tests for deleting datasets."""
    
    def test_delete_dataset_success(self, client: TestClient, test_dataset, auth_headers, db):
        """Test deleting a dataset."""
        dataset_id = test_dataset.id
        
        response = client.delete(
            f"/api/v1/datasets/{dataset_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        # Verify dataset is deleted
        from app.db import models
        deleted = db.query(models.Dataset).filter(
            models.Dataset.id == dataset_id
        ).first()
        assert deleted is None
    
    def test_delete_dataset_not_found(self, client: TestClient, auth_headers):
        """Test deleting non-existent dataset."""
        response = client.delete(
            "/api/v1/datasets/99999",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_delete_dataset_wrong_user(self, client: TestClient, test_dataset, second_user_headers):
        """Test deleting another user's dataset fails."""
        response = client.delete(
            f"/api/v1/datasets/{test_dataset.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
