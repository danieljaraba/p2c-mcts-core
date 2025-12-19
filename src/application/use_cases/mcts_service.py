"""MCTS algorithm use case implementation."""
import random
from typing import List, Optional

from src.application.ports.input_ports import MCTSServicePort
from src.domain.entities.mdp import Action, MDP, State
from src.domain.entities.mcts_node import MCTSNode


class MCTSService(MCTSServicePort):
    """Implementation of Monte Carlo Tree Search algorithm."""

    # Default maximum depth for simulation to prevent infinite loops
    DEFAULT_MAX_SIMULATION_DEPTH = 100

    def __init__(self, exploration_weight: float = 1.41, max_simulation_depth: int = DEFAULT_MAX_SIMULATION_DEPTH):
        """
        Initialize MCTS service.

        Args:
            exploration_weight: UCB1 exploration parameter (default sqrt(2))
            max_simulation_depth: Maximum depth for simulation phase
        """
        self.exploration_weight = exploration_weight
        self.max_simulation_depth = max_simulation_depth
        self.root: Optional[MCTSNode] = None

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
        # Create root node
        self.root = MCTSNode(
            state=initial_state,
            untried_actions=mdp.get_available_actions(initial_state),
        )

        # Run simulations
        for _ in range(num_simulations):
            node = self._select(self.root)
            
            if not node.is_terminal():
                node = self._expand(node, mdp)
            
            reward = self._simulate(node, mdp)
            self._backpropagate(node, reward)

        # Return best action
        best_child = self.root.get_most_visited_child()
        return best_child.action if best_child else None

    def get_search_tree(self, root: MCTSNode) -> dict:
        """
        Get the current search tree structure.

        Args:
            root: Root node of the search tree

        Returns:
            Dictionary representation of the tree
        """
        return self._node_to_dict(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB1.

        Args:
            node: Current node to select from

        Returns:
            Selected leaf node
        """
        while not node.is_terminal() and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                break
            node = best_child
        return node

    def _expand(self, node: MCTSNode, mdp: MDP) -> MCTSNode:
        """
        Expansion phase: add a new child node.

        Args:
            node: Node to expand
            mdp: MDP model

        Returns:
            Newly created child node or the original node if can't expand
        """
        if not node.untried_actions:
            return node

        # Pick a random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Get next state from MDP
        next_state = mdp.get_next_state(node.state, action)
        if next_state is None:
            return node

        # Create and add child node
        child = node.add_child(next_state, action)
        child.untried_actions = mdp.get_available_actions(next_state)
        return child

    def _simulate(self, node: MCTSNode, mdp: MDP) -> float:
        """
        Simulation phase: perform random rollout.

        Args:
            node: Node to simulate from
            mdp: MDP model

        Returns:
            Cumulative reward from simulation
        """
        current_state = node.state
        total_reward = 0.0
        depth = 0

        while not current_state.is_terminal and depth < self.max_simulation_depth:
            available_actions = mdp.get_available_actions(current_state)
            if not available_actions:
                break

            # Random action selection
            action = random.choice(available_actions)
            next_state = mdp.get_next_state(current_state, action)
            
            if next_state is None:
                break

            reward = mdp.get_reward(current_state, action, next_state)
            total_reward += reward * (mdp.gamma ** depth)
            
            current_state = next_state
            depth += 1

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: update statistics.

        Args:
            node: Leaf node to backpropagate from
            reward: Reward to propagate
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def _node_to_dict(self, node: MCTSNode) -> dict:
        """Convert MCTS node to dictionary representation."""
        return {
            "state_id": node.state.id,
            "visits": node.visits,
            "value": node.value,
            "action": node.action.id if node.action else None,
            "children": [self._node_to_dict(child) for child in node.children],
        }
