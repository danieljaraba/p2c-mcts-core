"""Input ports (interfaces) for MCTS operations."""
from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.mdp import Action, MDP, State
from src.domain.entities.mcts_node import MCTSNode


class MCTSServicePort(ABC):
    """Port for MCTS service operations."""

    @abstractmethod
    def search(
        self, mdp: MDP, initial_state: State, num_simulations: int
    ) -> Optional[Action]:
        """
        Perform MCTS search to find the best action.

        Args:
            mdp: Markov Decision Process model
            initial_state: Starting state
            num_simulations: Number of MCTS simulations to run

        Returns:
            Best action to take or None
        """
        pass

    @abstractmethod
    def get_search_tree(self, root: MCTSNode) -> dict:
        """
        Get the current search tree structure.

        Args:
            root: Root node of the search tree

        Returns:
            Dictionary representation of the tree
        """
        pass


class MDPRepositoryPort(ABC):
    """Port for MDP persistence operations."""

    @abstractmethod
    def save(self, mdp: MDP, mdp_id: str) -> None:
        """Save an MDP model."""
        pass

    @abstractmethod
    def get(self, mdp_id: str) -> Optional[MDP]:
        """Retrieve an MDP model by ID."""
        pass

    @abstractmethod
    def list_all(self) -> List[MDP]:
        """List all stored MDP models."""
        pass

    @abstractmethod
    def delete(self, mdp_id: str) -> bool:
        """Delete an MDP model."""
        pass
