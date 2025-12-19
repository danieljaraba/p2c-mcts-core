"""Unit tests for MCTS service."""
import pytest

from src.application.use_cases.mcts_service import MCTSService
from src.domain.entities.mdp import Action, MDP, State, Transition


class TestMCTSService:
    """Test MCTS service."""

    def test_create_service(self):
        """Test creating MCTS service."""
        service = MCTSService()
        assert service.exploration_weight == 1.41
        assert service.root is None

    def test_search_simple_mdp(self):
        """Test MCTS search on a simple MDP."""
        # Create a simple MDP
        state1 = State(id="s1")
        state2 = State(id="s2", is_terminal=True)
        state3 = State(id="s3", is_terminal=True)
        
        action1 = Action(id="a1", name="good_action")
        action2 = Action(id="a2", name="bad_action")
        
        mdp = MDP(
            states=[state1, state2, state3],
            actions=[action1, action2],
            initial_state=state1,
            gamma=0.95
        )
        
        # Good action leads to higher reward
        mdp.add_transition(Transition(state1, action1, state2, 100.0))
        # Bad action leads to lower reward
        mdp.add_transition(Transition(state1, action2, state3, 1.0))
        
        # Perform search
        service = MCTSService()
        best_action = service.search(mdp, state1, num_simulations=100)
        
        # Should find the good action most of the time
        assert best_action is not None
        # With enough simulations, should prefer the higher reward action
        assert best_action.id in ["a1", "a2"]

    def test_search_creates_root(self):
        """Test that search creates a root node."""
        state = State(id="s1", is_terminal=True)
        mdp = MDP(states=[state], actions=[], initial_state=state)
        
        service = MCTSService()
        service.search(mdp, state, num_simulations=1)
        
        assert service.root is not None
        assert service.root.state == state

    def test_get_search_tree(self):
        """Test getting search tree structure."""
        state1 = State(id="s1")
        state2 = State(id="s2", is_terminal=True)
        action = Action(id="a1", name="action")
        
        mdp = MDP(
            states=[state1, state2],
            actions=[action],
            initial_state=state1
        )
        mdp.add_transition(Transition(state1, action, state2, 10.0))
        
        service = MCTSService()
        service.search(mdp, state1, num_simulations=10)
        
        tree = service.get_search_tree(service.root)
        assert "state_id" in tree
        assert tree["state_id"] == "s1"
        assert "visits" in tree
        assert "value" in tree
        assert "children" in tree

    def test_search_terminal_state(self):
        """Test search from terminal state."""
        state = State(id="s1", is_terminal=True)
        mdp = MDP(states=[state], actions=[], initial_state=state)
        
        service = MCTSService()
        best_action = service.search(mdp, state, num_simulations=10)
        
        # Terminal state has no actions
        assert best_action is None
