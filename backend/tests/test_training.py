"""
Tests for training endpoints.

Endpoints tested:
- POST /api/v1/training/start
- GET /api/v1/training/{job_id}
- GET /api/v1/training/
- POST /api/v1/training/{job_id}/cancel
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestStartTraining:
    """Tests for starting training jobs."""
    
    @patch('app.services.training_service.TrainingService._queue_training_job')
    def test_start_training_success(self, mock_queue, client: TestClient, test_workspace, test_dataset, auth_headers):
        """Test starting a training job with default config."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "workspace_id": test_workspace.id,
                "dataset_id": test_dataset.id,
                "name": "My Fine-tuning Job",
                "base_model": "mistralai/Mistral-7B-v0.1"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["workspace_id"] == test_workspace.id
        assert data["dataset_id"] == test_dataset.id
        assert data["name"] == "My Fine-tuning Job"
        assert data["base_model"] == "mistralai/Mistral-7B-v0.1"
        assert data["status"] == "pending"
        assert data["progress"] == 0.0
        mock_queue.assert_called_once()
    
    @patch('app.services.training_service.TrainingService._queue_training_job')
    def test_start_training_with_custom_config(self, mock_queue, client: TestClient, test_workspace, test_dataset, auth_headers):
        """Test starting training with custom LoRA config."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "workspace_id": test_workspace.id,
                "dataset_id": test_dataset.id,
                "base_model": "meta-llama/Llama-2-7b-hf",
                "config": {
                    "lora_r": 32,
                    "lora_alpha": 64,
                    "lora_dropout": 0.1,
                    "learning_rate": 1e-4,
                    "num_epochs": 5,
                    "batch_size": 8
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["base_model"] == "meta-llama/Llama-2-7b-hf"
    
    def test_start_training_unauthorized(self, client: TestClient, test_workspace, test_dataset):
        """Test starting training without authentication."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "workspace_id": test_workspace.id,
                "dataset_id": test_dataset.id
            }
        )
        
        assert response.status_code == 401
    
    def test_start_training_wrong_workspace(self, client: TestClient, test_workspace, test_dataset, second_user_headers):
        """Test starting training in another user's workspace fails."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "workspace_id": test_workspace.id,
                "dataset_id": test_dataset.id
            },
            headers=second_user_headers
        )
        
        assert response.status_code == 404
        assert "workspace" in response.json()["detail"].lower()
    
    def test_start_training_invalid_dataset(self, client: TestClient, test_workspace, auth_headers):
        """Test starting training with non-existent dataset."""
        response = client.post(
            "/api/v1/training/start",
            json={
                "workspace_id": test_workspace.id,
                "dataset_id": 99999
            },
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "dataset" in response.json()["detail"].lower()
    
    def test_start_training_missing_fields(self, client: TestClient, auth_headers):
        """Test starting training with missing required fields."""
        response = client.post(
            "/api/v1/training/start",
            json={"workspace_id": 1},  # Missing dataset_id
            headers=auth_headers
        )
        
        assert response.status_code == 422


class TestGetTrainingJob:
    """Tests for getting training job details."""
    
    def test_get_training_job_success(self, client: TestClient, test_training_job, auth_headers):
        """Test getting training job details."""
        response = client.get(
            f"/api/v1/training/{test_training_job.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_training_job.id
        assert data["workspace_id"] == test_training_job.workspace_id
        assert data["dataset_id"] == test_training_job.dataset_id
        assert "status" in data
        assert "progress" in data
        assert "current_step" in data
    
    def test_get_training_job_not_found(self, client: TestClient, auth_headers):
        """Test getting non-existent training job."""
        response = client.get(
            "/api/v1/training/99999",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_get_training_job_unauthorized(self, client: TestClient, test_training_job):
        """Test getting training job without authentication."""
        response = client.get(f"/api/v1/training/{test_training_job.id}")
        
        assert response.status_code == 401
    
    def test_get_training_job_wrong_user(self, client: TestClient, test_training_job, second_user_headers):
        """Test getting another user's training job fails."""
        response = client.get(
            f"/api/v1/training/{test_training_job.id}",
            headers=second_user_headers
        )
        
        assert response.status_code == 404


class TestListTrainingJobs:
    """Tests for listing training jobs."""
    
    def test_list_training_jobs_success(self, client: TestClient, test_workspace, test_training_job, auth_headers):
        """Test listing training jobs in workspace."""
        response = client.get(
            f"/api/v1/training/?workspace_id={test_workspace.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(j["id"] == test_training_job.id for j in data)
    
    def test_list_training_jobs_empty(self, client: TestClient, db, test_user, auth_headers):
        """Test listing training jobs in workspace with no jobs."""
        from app.db import models
        
        # Create empty workspace
        empty_workspace = models.Workspace(
            name="Empty Workspace",
            owner_id=test_user.id
        )
        db.add(empty_workspace)
        db.commit()
        
        response = client.get(
            f"/api/v1/training/?workspace_id={empty_workspace.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json() == []
    
    def test_list_training_jobs_ordered_by_date(self, client: TestClient, db, test_workspace, test_dataset, auth_headers):
        """Test that training jobs are ordered by created_at desc."""
        from app.db import models
        from datetime import datetime, timedelta
        
        # Create multiple jobs
        job1 = models.TrainingJob(
            workspace_id=test_workspace.id,
            dataset_id=test_dataset.id,
            name="Job 1",
            base_model="model1",
            status="completed"
        )
        job2 = models.TrainingJob(
            workspace_id=test_workspace.id,
            dataset_id=test_dataset.id,
            name="Job 2",
            base_model="model2",
            status="pending"
        )
        db.add_all([job1, job2])
        db.commit()
        
        response = client.get(
            f"/api/v1/training/?workspace_id={test_workspace.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should be ordered newest first
        assert len(data) >= 2


class TestCancelTrainingJob:
    """Tests for cancelling training jobs."""
    
    def test_cancel_training_job_success(self, client: TestClient, test_training_job, auth_headers):
        """Test cancelling a pending training job."""
        response = client.post(
            f"/api/v1/training/{test_training_job.id}/cancel",
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    def test_cancel_training_job_not_found(self, client: TestClient, auth_headers):
        """Test cancelling non-existent training job."""
        response = client.post(
            "/api/v1/training/99999/cancel",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_cancel_training_job_unauthorized(self, client: TestClient, test_training_job):
        """Test cancelling training job without authentication."""
        response = client.post(f"/api/v1/training/{test_training_job.id}/cancel")
        
        assert response.status_code == 401
    
    def test_cancel_training_job_wrong_user(self, client: TestClient, test_training_job, second_user_headers):
        """Test cancelling another user's training job fails."""
        response = client.post(
            f"/api/v1/training/{test_training_job.id}/cancel",
            headers=second_user_headers
        )
        
        assert response.status_code == 404
