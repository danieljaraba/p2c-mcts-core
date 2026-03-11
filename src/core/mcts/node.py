"""MCTS tree node representation.

Each node represents a point in the crafting decision tree.  Nodes store the
MDP state reached at that point, the action that led there, visit statistics
used by UCB1, and references to the parent and children in the tree.
"""

from __future__ import annotations

import math

from src.core.mdp.entities import CraftingAction, ItemState


class MCTSNode:
    """A single node in the MCTS search tree.

    The node is deliberately *mutable* so that the MCTS backpropagation phase
    can update visit counts and accumulated values in-place without
    constructing new objects on every iteration.

    Args:
        state: The MDP state at this node.
        parent: Parent node, or ``None`` for the root.
        action: The crafting action that produced *state* from the parent's
            state.  ``None`` for the root node.
    """

    def __init__(
        self,
        state: ItemState,
        parent: MCTSNode | None = None,
        action: CraftingAction | None = None,
    ) -> None:
        self.state: ItemState = state
        self.parent: MCTSNode | None = parent
        self.action: CraftingAction | None = action
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.total_value: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean_value(self) -> float:
        """Return the average reward across all visits to this node.

        Returns:
            Mean value, or 0.0 if the node has never been visited.
        """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def is_root(self) -> bool:
        """Return True if this node has no parent."""
        return self.parent is None

    # ------------------------------------------------------------------
    # UCB1 score
    # ------------------------------------------------------------------

    def ucb1_score(self, exploration_constant: float) -> float:
        """Compute the UCB1 score used for tree-policy selection.

        Formula:  mean_value + C * sqrt( ln(parent_visits) / visits )

        Unvisited nodes receive +∞ to guarantee they are explored first.

        Args:
            exploration_constant: The *C* parameter in UCB1.  Higher values
                favour unexplored branches; lower values favour exploitation.

        Returns:
            UCB1 score as a float, or ``inf`` for unvisited nodes.
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.mean_value
        exploitation = self.mean_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    # ------------------------------------------------------------------
    # Expansion helpers
    # ------------------------------------------------------------------

    def explored_action_ids(self) -> frozenset[str]:
        """Return the set of action IDs already represented by child nodes.

        Returns:
            Frozenset of ``action_id`` strings.
        """
        return frozenset(
            child.action.action_id
            for child in self.children
            if child.action is not None
        )

    def is_fully_expanded(self, available_action_ids: frozenset[str]) -> bool:
        """Return True when every available action has at least one child node.

        Args:
            available_action_ids: Set of action IDs applicable in this state.

        Returns:
            True if all actions have been tried at least once.
        """
        return available_action_ids <= self.explored_action_ids()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        action_id = self.action.action_id if self.action else "root"
        return (
            f"MCTSNode(action={action_id!r}, visits={self.visits}, "
            f"mean_value={self.mean_value:.4f})"
        )
