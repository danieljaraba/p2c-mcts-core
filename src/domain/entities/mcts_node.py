"""Monte Carlo Tree Search node entity."""
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

# Support both old MDP and new crafting domain
try:
    from src.domain.entities.mdp import Action as MDPAction, State as MDPState
except ImportError:
    MDPAction = None
    MDPState = None

try:
    from src.domain.entities.crafting import CraftingAction, ItemState
except ImportError:
    CraftingAction = None
    ItemState = None


@dataclass
class MCTSNode:
    """Represents a node in the MCTS tree."""

    state: Any  # Can be State (MDP) or ItemState (Crafting)
    parent: Optional["MCTSNode"] = None
    action: Any = None  # Can be Action (MDP) or CraftingAction (Crafting)
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize untried actions if not provided."""
        if not self.untried_actions:
            self.untried_actions = []

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        # Check if state has is_terminal attribute
        if hasattr(self.state, 'is_terminal'):
            return self.state.is_terminal
        return False

    def best_child(self, exploration_weight: float = 1.41) -> Optional["MCTSNode"]:
        """
        Select the best child using UCB1 formula.

        Args:
            exploration_weight: Exploration parameter (default sqrt(2))

        Returns:
            Best child node or None if no children exist
        """
        if not self.children:
            return None

        best_score = float("-inf")
        best_child = None

        for child in self.children:
            if child.visits == 0:
                # Unvisited nodes have infinite UCB1 value
                return child

            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )
            ucb1_score = exploitation + exploration

            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child

        return best_child

    def add_child(self, state: Any, action: Any) -> "MCTSNode":
        """Add a child node to this node."""
        child = MCTSNode(state=state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        """Update node statistics with simulation result."""
        self.visits += 1
        self.value += reward

    def get_most_visited_child(self) -> Optional["MCTSNode"]:
        """Get the child with the most visits."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
