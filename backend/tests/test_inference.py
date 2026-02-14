"""
Tests for inference endpoints.

Endpoints tested:
- POST /api/v1/inference/predict
- POST /api/v1/inference/batch
"""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


class TestPredict:
    """Tests for single prediction endpoint."""
    
    @patch('app.api.inference.InferenceService')
    def test_predict_success(self, mock_service, client: TestClient, test_model, auth_headers):
        """Test successful prediction."""
        # Mock the inference service
        mock_instance = mock_service.return_value
        mock_instance.generate = AsyncMock(return_value={
            "generated_text": "Hello! How can I help you today?",
            "tokens_used": 42
        })
        
        response = client.post(
            "/api/v1/inference/predict",
            json={
                "model_id": test_model.id,
                "prompt": "Hello, how are you?",
                "max_tokens": 100,
                "temperature": 0.7
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == test_model.id
        assert data["prompt"] == "Hello, how are you?"
        assert "generated_text" in data
        assert "tokens_used" in data
    
    def test_predict_model_not_found(self, client: TestClient, auth_headers):
        """Test prediction with non-existent model."""
        response = client.post(
            "/api/v1/inference/predict",
            json={
                "model_id": 99999,
                "prompt": "Hello"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "model" in response.json()["detail"].lower()
    
    def test_predict_unauthorized(self, client: TestClient, test_model):
        """Test prediction without authentication."""
        response = client.post(
            "/api/v1/inference/predict",
            json={
                "model_id": test_model.id,
                "prompt": "Hello"
            }
        )
        
        assert response.status_code == 401
    
    def test_predict_wrong_user(self, client: TestClient, test_model, second_user_headers):
        """Test prediction with another user's model fails."""
        response = client.post(
            "/api/v1/inference/predict",
            json={
                "model_id": test_model.id,
                "prompt": "Hello"
            },
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_predict_missing_prompt(self, client: TestClient, test_model, auth_headers):
        """Test prediction with missing prompt."""
        response = client.post(
            "/api/v1/inference/predict",
            json={"model_id": test_model.id},
            headers=auth_headers
        )
        
        assert response.status_code == 422
    
    def test_predict_with_all_params(self, client: TestClient, test_model, auth_headers):
        """Test prediction with all optional parameters."""
        with patch('app.api.inference.InferenceService') as mock_service:
            mock_instance = mock_service.return_value
            mock_instance.generate = AsyncMock(return_value={
                "generated_text": "Response",
                "tokens_used": 10
            })
            
            response = client.post(
                "/api/v1/inference/predict",
                json={
                    "model_id": test_model.id,
                    "prompt": "Test prompt",
                    "max_tokens": 512,
                    "temperature": 0.5,
                    "top_p": 0.85
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
    
    @patch('app.api.inference.InferenceService')
    def test_predict_service_error(self, mock_service, client: TestClient, test_model, auth_headers):
        """Test prediction when inference service fails."""
        mock_instance = mock_service.return_value
        mock_instance.generate = AsyncMock(side_effect=Exception("GPU out of memory"))
        
        response = client.post(
            "/api/v1/inference/predict",
            json={
                "model_id": test_model.id,
                "prompt": "Hello"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 500
        assert "inference failed" in response.json()["detail"].lower()


class TestBatchPredict:
    """Tests for batch prediction endpoint."""
    
    @patch('app.api.inference.InferenceService')
    def test_batch_predict_success(self, mock_service, client: TestClient, test_model, auth_headers):
        """Test successful batch prediction."""
        mock_instance = mock_service.return_value
        mock_instance.generate = AsyncMock(side_effect=[
            {"generated_text": "Response 1", "tokens_used": 10},
            {"generated_text": "Response 2", "tokens_used": 15},
            {"generated_text": "Response 3", "tokens_used": 12}
        ])
        
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model_id": test_model.id,
                "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
                "max_tokens": 100,
                "temperature": 0.7
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == test_model.id
        assert "results" in data
        assert len(data["results"]) == 3
    
    def test_batch_predict_model_not_found(self, client: TestClient, auth_headers):
        """Test batch prediction with non-existent model."""
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model_id": 99999,
                "prompts": ["Hello", "World"]
            },
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_batch_predict_unauthorized(self, client: TestClient, test_model):
        """Test batch prediction without authentication."""
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model_id": test_model.id,
                "prompts": ["Hello"]
            }
        )
        
        assert response.status_code == 401
    
    def test_batch_predict_wrong_user(self, client: TestClient, test_model, second_user_headers):
        """Test batch prediction with another user's model fails."""
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model_id": test_model.id,
                "prompts": ["Hello"]
            },
            headers=second_user_headers
        )
        
        assert response.status_code == 404
    
    def test_batch_predict_empty_prompts(self, client: TestClient, test_model, auth_headers):
        """Test batch prediction with empty prompts list."""
        with patch('app.api.inference.InferenceService') as mock_service:
            mock_instance = mock_service.return_value
            
            response = client.post(
                "/api/v1/inference/batch",
                json={
                    "model_id": test_model.id,
                    "prompts": []
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["results"] == []
    
    @patch('app.api.inference.InferenceService')
    def test_batch_predict_partial_failure(self, mock_service, client: TestClient, test_model, auth_headers):
        """Test batch prediction where some prompts fail."""
        mock_instance = mock_service.return_value
        
        # First call succeeds, second fails, third succeeds
        mock_instance.generate = AsyncMock(side_effect=[
            {"generated_text": "Response 1", "tokens_used": 10},
            Exception("Token limit exceeded"),
            {"generated_text": "Response 3", "tokens_used": 12}
        ])
        
        response = client.post(
            "/api/v1/inference/batch",
            json={
                "model_id": test_model.id,
                "prompts": ["Prompt 1", "Very long prompt...", "Prompt 3"]
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        # Check that error is captured for failed prompt
        assert any("error" in r for r in data["results"])
