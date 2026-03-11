"""Unit tests for the MCTS search engine."""

from __future__ import annotations

import pytest

from src.core.mcts.search import SearchResult
from src.core.mdp.entities import CraftingStrategy
from src.core.optimizer import CraftingOptimizer


class TestMCTSSearchStructure:
    """Tests for MCTS search algorithm correctness."""

    def test_search_returns_search_result(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=50, seed=42)
        result = optimizer.optimize(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
            strategy=CraftingStrategy.DETERMINISTIC,
        )
        assert isinstance(result, SearchResult)

    def test_search_result_has_best_path(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=50, seed=42)
        result = optimizer.optimize(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
            strategy=CraftingStrategy.DETERMINISTIC,
        )
        # Bench craft always adds the modifier → best path should have 1 step.
        assert len(result.best_path) >= 1

    def test_search_metadata_populated(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=100, seed=0)
        result = optimizer.optimize(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
            strategy=CraftingStrategy.BALANCED,
        )
        assert result.metadata.iterations == 100
        assert result.metadata.computation_time >= 0.0
        assert result.metadata.nodes_explored >= 1

    def test_strategy_label_in_result(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=50, seed=1)
        for strategy in CraftingStrategy:
            result = optimizer.optimize(
                initial_state=empty_state,
                goal=single_mod_goal,
                available_actions=[bench_craft_action],
                strategy=strategy,
            )
            assert result.strategy == strategy.value

    def test_success_true_for_deterministic_bench_craft(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        # With enough iterations the bench craft path should succeed.
        optimizer = CraftingOptimizer(iterations=200, seed=42)
        result = optimizer.optimize(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
            strategy=CraftingStrategy.DETERMINISTIC,
        )
        assert result.success is True

    def test_path_step_fields_populated(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=100, seed=7)
        result = optimizer.optimize(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
            strategy=CraftingStrategy.DETERMINISTIC,
        )
        if result.best_path:
            step = result.best_path[0]
            assert step.step == 1
            assert step.action_id == "bench_life_regen"
            assert step.action_name == "Bench: Life Regeneration"
            assert step.estimated_cost == pytest.approx(2.0)
            assert 0.0 <= step.expected_reward <= 1.0
            assert 0.0 <= step.confidence <= 1.0

    def test_summary_total_cost_equals_sum_of_steps(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=100, seed=3)
        result = optimizer.optimize(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
            strategy=CraftingStrategy.CHEAPEST,
        )
        expected_cost = sum(s.estimated_cost for s in result.best_path)
        assert result.summary.total_cost == pytest.approx(expected_cost)


class TestOptimizeAllStrategies:
    """Tests for the optimize_all_strategies convenience method."""

    def test_returns_all_three_strategies(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=50, seed=5)
        results = optimizer.optimize_all_strategies(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
        )
        assert set(results.keys()) == {"deterministic", "cheapest", "balanced"}

    def test_each_result_has_correct_strategy_label(
        self,
        empty_state,
        single_mod_goal,
        bench_craft_action,
    ) -> None:
        optimizer = CraftingOptimizer(iterations=50, seed=9)
        results = optimizer.optimize_all_strategies(
            initial_state=empty_state,
            goal=single_mod_goal,
            available_actions=[bench_craft_action],
        )
        for name, result in results.items():
            assert result.strategy == name
