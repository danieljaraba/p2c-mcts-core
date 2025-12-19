"""In-memory MDP repository implementation."""
from typing import Dict, List, Optional

from src.application.ports.input_ports import MDPRepositoryPort
from src.domain.entities.mdp import MDP


class InMemoryMDPRepository(MDPRepositoryPort):
    """In-memory implementation of MDP repository."""

    def __init__(self) -> None:
        """Initialize repository with empty storage."""
        self._storage: Dict[str, MDP] = {}

    def save(self, mdp: MDP, mdp_id: str) -> None:
        """
        Save an MDP model.

        Args:
            mdp: MDP model to save
            mdp_id: Unique identifier for the MDP
        """
        self._storage[mdp_id] = mdp

    def get(self, mdp_id: str) -> Optional[MDP]:
        """
        Retrieve an MDP model by ID.

        Args:
            mdp_id: Unique identifier for the MDP

        Returns:
            MDP model or None if not found
        """
        return self._storage.get(mdp_id)

    def list_all(self) -> List[MDP]:
        """
        List all stored MDP models.

        Returns:
            List of all MDP models
        """
        return list(self._storage.values())

    def delete(self, mdp_id: str) -> bool:
        """
        Delete an MDP model.

        Args:
            mdp_id: Unique identifier for the MDP

        Returns:
            True if deleted, False if not found
        """
        if mdp_id in self._storage:
            del self._storage[mdp_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all stored MDPs."""
        self._storage.clear()
