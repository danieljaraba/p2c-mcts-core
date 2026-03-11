"""Unit tests for the MCTS node."""

from __future__ import annotations

import math

import pytest

from src.core.mcts.node import MCTSNode
from src.core.mdp.entities import (
    ActionEffects,
    ActionType,
    BaseItem,
    CraftingAction,
    ItemState,
    Outcome,
)


@pytest.fixture
def empty_state() -> ItemState:
    return ItemState(base_item=BaseItem("helmet", 86), modifiers=frozenset())


@pytest.fixture
def dummy_action() -> CraftingAction:
    return CraftingAction(
        action_id="chaos_orb",
        action_type=ActionType.CURRENCY,
        name="Chaos Orb",
        cost=1.5,
        effects=ActionEffects(
            deterministic=False,
            outcomes=(Outcome(probability=1.0, transformation_type="reroll_all"),),
        ),
    )


class TestMCTSNodeProperties:
    """Tests for MCTSNode properties and state."""

    def test_initial_visits_and_value(self, empty_state: ItemState) -> None:
        node = MCTSNode(state=empty_state)
        assert node.visits == 0
        assert node.total_value == 0.0

    def test_mean_value_unvisited_returns_zero(self, empty_state: ItemState) -> None:
        node = MCTSNode(state=empty_state)
        assert node.mean_value == 0.0

    def test_mean_value_after_updates(self, empty_state: ItemState) -> None:
        node = MCTSNode(state=empty_state)
        node.visits = 4
        node.total_value = 3.2
        assert node.mean_value == pytest.approx(0.8)

    def test_is_root_true_for_no_parent(self, empty_state: ItemState) -> None:
        node = MCTSNode(state=empty_state)
        assert node.is_root is True

    def test_is_root_false_for_child(
        self, empty_state: ItemState, dummy_action: CraftingAction
    ) -> None:
        parent = MCTSNode(state=empty_state)
        child = MCTSNode(state=empty_state, parent=parent, action=dummy_action)
        assert child.is_root is False


class TestUCB1Score:
    """Tests for the UCB1 scoring method."""

    def test_unvisited_node_returns_infinity(self, empty_state: ItemState) -> None:
        node = MCTSNode(state=empty_state)
        assert node.ucb1_score(1.414) == float("inf")

    def test_ucb1_score_without_parent_returns_mean_value(
        self, empty_state: ItemState
    ) -> None:
        node = MCTSNode(state=empty_state)
        node.visits = 5
        node.total_value = 3.0
        # No parent → exploitation only.
        assert node.ucb1_score(1.414) == pytest.approx(0.6)

    def test_ucb1_score_with_parent(
        self, empty_state: ItemState, dummy_action: CraftingAction
    ) -> None:
        parent = MCTSNode(state=empty_state)
        parent.visits = 10
        child = MCTSNode(state=empty_state, parent=parent, action=dummy_action)
        child.visits = 2
        child.total_value = 1.4

        exploitation = 1.4 / 2
        exploration = 1.414 * math.sqrt(math.log(10) / 2)
        expected = exploitation + exploration
        assert child.ucb1_score(1.414) == pytest.approx(expected, rel=1e-4)

    def test_higher_exploration_constant_raises_score(
        self, empty_state: ItemState, dummy_action: CraftingAction
    ) -> None:
        parent = MCTSNode(state=empty_state)
        parent.visits = 10
        child = MCTSNode(state=empty_state, parent=parent, action=dummy_action)
        child.visits = 2
        child.total_value = 1.0

        score_low_c = child.ucb1_score(0.5)
        score_high_c = child.ucb1_score(2.0)
        assert score_high_c > score_low_c


class TestExpansionHelpers:
    """Tests for exploration tracking helpers."""

    def test_explored_action_ids_empty_for_no_children(
        self, empty_state: ItemState
    ) -> None:
        node = MCTSNode(state=empty_state)
        assert node.explored_action_ids() == frozenset()

    def test_explored_action_ids_includes_child_actions(
        self, empty_state: ItemState, dummy_action: CraftingAction
    ) -> None:
        parent = MCTSNode(state=empty_state)
        child = MCTSNode(
            state=empty_state, parent=parent, action=dummy_action
        )
        parent.children.append(child)
        assert "chaos_orb" in parent.explored_action_ids()

    def test_is_fully_expanded_false_when_unexplored_actions_remain(
        self, empty_state: ItemState
    ) -> None:
        node = MCTSNode(state=empty_state)
        assert node.is_fully_expanded(frozenset({"chaos_orb", "bench"})) is False

    def test_is_fully_expanded_true_when_all_actions_explored(
        self, empty_state: ItemState, dummy_action: CraftingAction
    ) -> None:
        parent = MCTSNode(state=empty_state)
        child = MCTSNode(
            state=empty_state, parent=parent, action=dummy_action
        )
        parent.children.append(child)
        assert parent.is_fully_expanded(frozenset({"chaos_orb"})) is True


class TestMCTSNodeRepr:
    """Tests for MCTSNode string representation."""

    def test_repr_contains_action_id(
        self, empty_state: ItemState, dummy_action: CraftingAction
    ) -> None:
        parent = MCTSNode(state=empty_state)
        child = MCTSNode(
            state=empty_state, parent=parent, action=dummy_action
        )
        assert "chaos_orb" in repr(child)

    def test_repr_root_label(self, empty_state: ItemState) -> None:
        node = MCTSNode(state=empty_state)
        assert "root" in repr(node)
