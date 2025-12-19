"""Markov Decision Process (MDP) entity."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class State:
    """Represents a state in the MDP."""

    id: str
    data: Dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False

    def __hash__(self) -> int:
        """Hash function for state comparison."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality comparison for states."""
        if not isinstance(other, State):
            return False
        return self.id == other.id


@dataclass
class Action:
    """Represents an action in the MDP."""

    id: str
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash function for action comparison."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality comparison for actions."""
        if not isinstance(other, Action):
            return False
        return self.id == other.id


@dataclass
class Transition:
    """Represents a state transition in the MDP."""

    from_state: State
    action: Action
    to_state: State
    reward: float
    probability: float = 1.0


class MDP:
    """Markov Decision Process model."""

    def __init__(
        self,
        states: List[State],
        actions: List[Action],
        initial_state: State,
        gamma: float = 0.95,
    ):
        """
        Initialize MDP.

        Args:
            states: List of all possible states
            actions: List of all possible actions
            initial_state: The starting state
            gamma: Discount factor for future rewards
        """
        self.states = states
        self.actions = actions
        self.initial_state = initial_state
        self.gamma = gamma
        self.transitions: List[Transition] = []

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the MDP."""
        self.transitions.append(transition)

    def get_available_actions(self, state: State) -> List[Action]:
        """Get available actions from a given state."""
        available_actions = []
        for transition in self.transitions:
            if transition.from_state == state:
                if transition.action not in available_actions:
                    available_actions.append(transition.action)
        return available_actions

    def get_next_state(self, state: State, action: Action) -> Optional[State]:
        """Get the next state given current state and action."""
        for transition in self.transitions:
            if transition.from_state == state and transition.action == action:
                return transition.to_state
        return None

    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """Get reward for a state transition."""
        for transition in self.transitions:
            if (
                transition.from_state == state
                and transition.action == action
                and transition.to_state == next_state
            ):
                return transition.reward
        return 0.0
