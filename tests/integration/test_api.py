"""Integration tests for the FastAPI adapter.

These tests exercise the full HTTP stack from request parsing through domain
logic to JSON serialisation, using HTTPX's async test client.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.adapters.api.dependencies import _build_optimizer
from src.adapters.api.main import app


@pytest.fixture(autouse=True)
def reset_optimizer_cache() -> None:
    """Clear the lru_cache between tests to allow fresh optimizer instances."""
    _build_optimizer.cache_clear()


@pytest.fixture
def client() -> TestClient:
    """Return a synchronous HTTPX test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Shared request payload builder
# ---------------------------------------------------------------------------


def _make_payload(
    goal_type: str = "partial",
    max_steps: int = 20,
    budget_limit: float = 200.0,
    include_bench_craft: bool = True,
    include_chaos_orb: bool = False,
) -> dict:
    """Build a minimal valid request payload.

    Args:
        goal_type: Goal evaluation mode.
        max_steps: Hard step limit.
        budget_limit: Budget constraint.
        include_bench_craft: Whether to add the deterministic bench craft action.
        include_chaos_orb: Whether to add the stochastic chaos orb action.

    Returns:
        Dictionary suitable for JSON serialisation.
    """
    actions = []
    if include_bench_craft:
        actions.append(
            {
                "action_id": "bench_life_regen",
                "action_type": "bench",
                "name": "Bench: Life Regeneration",
                "cost": 2.0,
                "effects": {
                    "deterministic": True,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "transformation_type": "add_specific_mod",
                            "specific_modifier_id": "life_regen",
                            "specific_modifier_tier": 1,
                            "specific_modifier_value": 50.0,
                        }
                    ],
                },
                "prerequisites": {
                    "required_state": "has_empty_mod_slot",
                },
            }
        )
    if include_chaos_orb:
        actions.append(
            {
                "action_id": "chaos_orb",
                "action_type": "currency",
                "name": "Chaos Orb",
                "cost": 1.5,
                "effects": {
                    "deterministic": False,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "transformation_type": "reroll_all",
                            "modifier_pool": [
                                "life_regen",
                                "crit_chance",
                                "fire_res",
                                "cold_res",
                                "flat_life",
                                "mana",
                            ],
                            "min_mod_count": 3,
                            "max_mod_count": 6,
                        }
                    ],
                },
            }
        )
    return {
        "goal": {
            "goal_type": goal_type,
            "target_modifiers": [
                {"modifier_id": "life_regen", "min_tier": 1, "weight": 1.0}
            ],
            "base_item": {
                "item_type": "chest_armour",
                "item_level": 86,
                "influence": "hunter",
            },
            "constraints": {
                "max_steps": max_steps,
                "budget_limit": budget_limit,
            },
        },
        "actions": actions,
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/v1/optimize  (all strategies)
# ---------------------------------------------------------------------------


class TestOptimizeAll:
    def test_optimize_all_returns_200(self, client: TestClient) -> None:
        payload = _make_payload()
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 200

    def test_optimize_all_response_has_all_strategies(
        self, client: TestClient
    ) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize", json=payload).json()
        assert "deterministic" in data
        assert "cheapest" in data
        assert "balanced" in data

    def test_each_strategy_has_required_fields(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize", json=payload).json()
        for key in ("deterministic", "cheapest", "balanced"):
            result = data[key]
            assert "success" in result
            assert "strategy" in result
            assert "best_path" in result
            assert "summary" in result
            assert "metadata" in result

    def test_strategy_label_matches_key(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize", json=payload).json()
        for key in ("deterministic", "cheapest", "balanced"):
            assert data[key]["strategy"] == key

    def test_empty_actions_returns_422(self, client: TestClient) -> None:
        payload = _make_payload(include_bench_craft=False)
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 422

    def test_invalid_goal_type_returns_422(self, client: TestClient) -> None:
        payload = _make_payload()
        payload["goal"]["goal_type"] = "super_exact"
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 422

    def test_invalid_action_type_returns_422(self, client: TestClient) -> None:
        payload = _make_payload()
        payload["actions"][0]["action_type"] = "magic"
        response = client.post("/api/v1/optimize", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/optimize/deterministic
# ---------------------------------------------------------------------------


class TestOptimizeDeterministic:
    def test_returns_200(self, client: TestClient) -> None:
        payload = _make_payload()
        response = client.post("/api/v1/optimize/deterministic", json=payload)
        assert response.status_code == 200

    def test_strategy_is_deterministic(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize/deterministic", json=payload).json()
        assert data["strategy"] == "deterministic"

    def test_success_field_is_boolean(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize/deterministic", json=payload).json()
        assert isinstance(data["success"], bool)


# ---------------------------------------------------------------------------
# POST /api/v1/optimize/cheapest
# ---------------------------------------------------------------------------


class TestOptimizeCheapest:
    def test_returns_200(self, client: TestClient) -> None:
        payload = _make_payload()
        response = client.post("/api/v1/optimize/cheapest", json=payload)
        assert response.status_code == 200

    def test_strategy_is_cheapest(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize/cheapest", json=payload).json()
        assert data["strategy"] == "cheapest"


# ---------------------------------------------------------------------------
# POST /api/v1/optimize/balanced
# ---------------------------------------------------------------------------


class TestOptimizeBalanced:
    def test_returns_200(self, client: TestClient) -> None:
        payload = _make_payload()
        response = client.post("/api/v1/optimize/balanced", json=payload)
        assert response.status_code == 200

    def test_strategy_is_balanced(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize/balanced", json=payload).json()
        assert data["strategy"] == "balanced"


# ---------------------------------------------------------------------------
# Summary and metadata field validation
# ---------------------------------------------------------------------------


class TestResponseStructure:
    def test_summary_fields_present(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize/balanced", json=payload).json()
        summary = data["summary"]
        assert "total_steps" in summary
        assert "total_cost" in summary
        assert "expected_final_reward" in summary
        assert "success_probability" in summary
        assert "alternative_paths" in summary

    def test_metadata_fields_present(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize/balanced", json=payload).json()
        meta = data["metadata"]
        assert "iterations" in meta
        assert "computation_time" in meta
        assert "tree_depth" in meta
        assert "nodes_explored" in meta

    def test_path_step_fields_present(self, client: TestClient) -> None:
        payload = _make_payload()
        data = client.post("/api/v1/optimize/deterministic", json=payload).json()
        if data["best_path"]:
            step = data["best_path"][0]
            assert "step" in step
            assert "action_id" in step
            assert "action_name" in step
            assert "estimated_cost" in step
            assert "expected_reward" in step
            assert "confidence" in step

    def test_metadata_iteration_count_matches_request(
        self, client: TestClient
    ) -> None:
        payload = _make_payload()
        payload["iterations"] = 100
        data = client.post("/api/v1/optimize/balanced", json=payload).json()
        assert data["metadata"]["iterations"] == 100
