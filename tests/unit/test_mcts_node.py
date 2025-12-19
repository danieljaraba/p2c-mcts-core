"""Unit tests for MCTS node entity."""
import pytest

from src.domain.entities.mdp import Action, State
from src.domain.entities.mcts_node import MCTSNode


class TestMCTSNode:
    """Test MCTSNode entity."""

    def test_create_node(self):
        """Test creating an MCTS node."""
        state = State(id="s1")
        node = MCTSNode(state=state)
        
        assert node.state == state
        assert node.parent is None
        assert node.action is None
        assert len(node.children) == 0
        assert node.visits == 0
        assert node.value == 0.0

    def test_is_fully_expanded(self):
        """Test checking if node is fully expanded."""
        state = State(id="s1")
        action1 = Action(id="a1", name="move")
        action2 = Action(id="a2", name="jump")
        
        node = MCTSNode(state=state, untried_actions=[action1, action2])
        assert not node.is_fully_expanded()
        
        node.untried_actions = []
        assert node.is_fully_expanded()

    def test_is_terminal(self):
        """Test checking if node is terminal."""
        terminal_state = State(id="s1", is_terminal=True)
        non_terminal_state = State(id="s2", is_terminal=False)
        
        terminal_node = MCTSNode(state=terminal_state)
        non_terminal_node = MCTSNode(state=non_terminal_state)
        
        assert terminal_node.is_terminal()
        assert not non_terminal_node.is_terminal()

    def test_add_child(self):
        """Test adding a child node."""
        parent_state = State(id="s1")
        child_state = State(id="s2")
        action = Action(id="a1", name="move")
        
        parent = MCTSNode(state=parent_state)
        child = parent.add_child(child_state, action)
        
        assert len(parent.children) == 1
        assert child.parent == parent
        assert child.state == child_state
        assert child.action == action

    def test_update(self):
        """Test updating node statistics."""
        state = State(id="s1")
        node = MCTSNode(state=state)
        
        node.update(10.0)
        assert node.visits == 1
        assert node.value == 10.0
        
        node.update(5.0)
        assert node.visits == 2
        assert node.value == 15.0

    def test_best_child(self):
        """Test selecting best child using UCB1."""
        parent_state = State(id="s1")
        parent = MCTSNode(state=parent_state)
        
        # Add and visit children
        child1_state = State(id="s2")
        child1 = parent.add_child(child1_state, Action(id="a1", name="action1"))
        child1.update(10.0)
        
        child2_state = State(id="s3")
        child2 = parent.add_child(child2_state, Action(id="a2", name="action2"))
        child2.update(15.0)
        
        parent.visits = 2
        
        best = parent.best_child()
        assert best is not None
        # The best child should have higher value
        assert best == child2

    def test_get_most_visited_child(self):
        """Test getting most visited child."""
        parent_state = State(id="s1")
        parent = MCTSNode(state=parent_state)
        
        child1 = parent.add_child(State(id="s2"), Action(id="a1", name="a1"))
        child1.visits = 5
        
        child2 = parent.add_child(State(id="s3"), Action(id="a2", name="a2"))
        child2.visits = 10
        
        most_visited = parent.get_most_visited_child()
        assert most_visited == child2
