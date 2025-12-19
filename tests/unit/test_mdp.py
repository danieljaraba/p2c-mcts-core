"""Unit tests for MDP entity."""
import pytest

from src.domain.entities.mdp import Action, MDP, State, Transition


class TestState:
    """Test State entity."""

    def test_create_state(self):
        """Test creating a state."""
        state = State(id="s1", data={"value": 10}, is_terminal=False)
        assert state.id == "s1"
        assert state.data["value"] == 10
        assert not state.is_terminal

    def test_state_equality(self):
        """Test state equality comparison."""
        state1 = State(id="s1")
        state2 = State(id="s1")
        state3 = State(id="s2")
        assert state1 == state2
        assert state1 != state3

    def test_state_hash(self):
        """Test state hashing."""
        state1 = State(id="s1")
        state2 = State(id="s1")
        assert hash(state1) == hash(state2)


class TestAction:
    """Test Action entity."""

    def test_create_action(self):
        """Test creating an action."""
        action = Action(id="a1", name="move", parameters={"direction": "up"})
        assert action.id == "a1"
        assert action.name == "move"
        assert action.parameters["direction"] == "up"

    def test_action_equality(self):
        """Test action equality comparison."""
        action1 = Action(id="a1", name="move")
        action2 = Action(id="a1", name="move")
        action3 = Action(id="a2", name="jump")
        assert action1 == action2
        assert action1 != action3


class TestMDP:
    """Test MDP entity."""

    def test_create_mdp(self):
        """Test creating an MDP."""
        state1 = State(id="s1")
        state2 = State(id="s2")
        action = Action(id="a1", name="move")
        
        mdp = MDP(
            states=[state1, state2],
            actions=[action],
            initial_state=state1,
            gamma=0.95
        )
        
        assert len(mdp.states) == 2
        assert len(mdp.actions) == 1
        assert mdp.initial_state == state1
        assert mdp.gamma == 0.95

    def test_add_transition(self):
        """Test adding transitions to MDP."""
        state1 = State(id="s1")
        state2 = State(id="s2")
        action = Action(id="a1", name="move")
        
        mdp = MDP(states=[state1, state2], actions=[action], initial_state=state1)
        
        transition = Transition(
            from_state=state1,
            action=action,
            to_state=state2,
            reward=10.0
        )
        mdp.add_transition(transition)
        
        assert len(mdp.transitions) == 1
        assert mdp.transitions[0].reward == 10.0

    def test_get_available_actions(self):
        """Test getting available actions from a state."""
        state1 = State(id="s1")
        state2 = State(id="s2")
        action1 = Action(id="a1", name="move")
        action2 = Action(id="a2", name="jump")
        
        mdp = MDP(
            states=[state1, state2],
            actions=[action1, action2],
            initial_state=state1
        )
        
        mdp.add_transition(Transition(state1, action1, state2, 5.0))
        mdp.add_transition(Transition(state1, action2, state2, 10.0))
        
        actions = mdp.get_available_actions(state1)
        assert len(actions) == 2
        assert action1 in actions
        assert action2 in actions

    def test_get_next_state(self):
        """Test getting next state from action."""
        state1 = State(id="s1")
        state2 = State(id="s2")
        action = Action(id="a1", name="move")
        
        mdp = MDP(states=[state1, state2], actions=[action], initial_state=state1)
        mdp.add_transition(Transition(state1, action, state2, 10.0))
        
        next_state = mdp.get_next_state(state1, action)
        assert next_state == state2

    def test_get_reward(self):
        """Test getting reward for transition."""
        state1 = State(id="s1")
        state2 = State(id="s2")
        action = Action(id="a1", name="move")
        
        mdp = MDP(states=[state1, state2], actions=[action], initial_state=state1)
        mdp.add_transition(Transition(state1, action, state2, 15.0))
        
        reward = mdp.get_reward(state1, action, state2)
        assert reward == 15.0
