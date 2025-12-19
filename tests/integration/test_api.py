"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestMCTSSearchEndpoint:
    """Test MCTS search endpoint."""

    def test_mcts_search_simple(self, client):
        """Test MCTS search with simple MDP."""
        request_data = {
            "mdp": {
                "states": [
                    {"id": "s1", "data": {}, "is_terminal": False},
                    {"id": "s2", "data": {}, "is_terminal": True},
                    {"id": "s3", "data": {}, "is_terminal": True},
                ],
                "actions": [
                    {"id": "a1", "name": "action1", "parameters": {}},
                    {"id": "a2", "name": "action2", "parameters": {}},
                ],
                "transitions": [
                    {
                        "from_state_id": "s1",
                        "action_id": "a1",
                        "to_state_id": "s2",
                        "reward": 10.0,
                        "probability": 1.0,
                    },
                    {
                        "from_state_id": "s1",
                        "action_id": "a2",
                        "to_state_id": "s3",
                        "reward": 5.0,
                        "probability": 1.0,
                    },
                ],
                "initial_state_id": "s1",
                "gamma": 0.95,
            },
            "num_simulations": 50,
            "exploration_weight": 1.41,
        }

        response = client.post("/api/v1/mcts/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "best_action" in data
        assert "search_tree" in data
        assert "simulations_run" in data
        assert data["simulations_run"] == 50
        
        if data["best_action"]:
            assert data["best_action"]["id"] in ["a1", "a2"]

    def test_mcts_search_invalid_initial_state(self, client):
        """Test MCTS search with invalid initial state."""
        request_data = {
            "mdp": {
                "states": [
                    {"id": "s1", "data": {}, "is_terminal": False},
                ],
                "actions": [
                    {"id": "a1", "name": "action1", "parameters": {}},
                ],
                "transitions": [],
                "initial_state_id": "s999",  # Invalid state
                "gamma": 0.95,
            },
            "num_simulations": 10,
        }

        response = client.post("/api/v1/mcts/search", json=request_data)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_mcts_search_invalid_transition(self, client):
        """Test MCTS search with invalid transition."""
        request_data = {
            "mdp": {
                "states": [
                    {"id": "s1", "data": {}, "is_terminal": False},
                ],
                "actions": [
                    {"id": "a1", "name": "action1", "parameters": {}},
                ],
                "transitions": [
                    {
                        "from_state_id": "s1",
                        "action_id": "a1",
                        "to_state_id": "s999",  # Invalid state
                        "reward": 10.0,
                    }
                ],
                "initial_state_id": "s1",
                "gamma": 0.95,
            },
            "num_simulations": 10,
        }

        response = client.post("/api/v1/mcts/search", json=request_data)
        assert response.status_code == 400

    def test_mcts_search_validation(self, client):
        """Test request validation."""
        # Missing required fields
        request_data = {
            "mdp": {
                "states": [],
                "actions": [],
                "transitions": [],
            }
        }

        response = client.post("/api/v1/mcts/search", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_mcts_search_terminal_only(self, client):
        """Test MCTS search with only terminal state."""
        request_data = {
            "mdp": {
                "states": [
                    {"id": "s1", "data": {}, "is_terminal": True},
                ],
                "actions": [],
                "transitions": [],
                "initial_state_id": "s1",
                "gamma": 0.95,
            },
            "num_simulations": 10,
        }

        response = client.post("/api/v1/mcts/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Terminal state should have no best action
        assert data["best_action"] is None
