"""MCTS search algorithm for Path of Exile 2 item crafting optimisation.

The :class:`MCTSSearch` class implements the four-phase Monte Carlo Tree Search
loop using the **Template Method** pattern.  The four phases (selection,
expansion, simulation, backpropagation) are fixed; the concrete behaviour of
selection and simulation is injected via the **Strategy** pattern
(:class:`~src.core.mcts.policies.SelectionPolicy` and
:class:`~src.core.mcts.policies.SimulationPolicy`).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.core.mcts.node import MCTSNode
from src.core.mcts.policies import SelectionPolicy, SimulationPolicy
from src.core.mdp.entities import CraftingAction, CraftingGoal, ItemState
from src.core.mdp.reward import RewardFunction
from src.core.mdp.transition import TransitionModel

# ---------------------------------------------------------------------------
# Result data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathStep:
    """One step in the recommended crafting sequence.

    Args:
        step: 1-based position in the sequence.
        action_id: Unique action identifier.
        action_name: Human-readable display name.
        estimated_cost: Currency cost of this individual step.
        expected_reward: Mean value observed in the subtree below this step.
        confidence: Fraction of total iterations that passed through this node.
    """

    step: int
    action_id: str
    action_name: str
    estimated_cost: float
    expected_reward: float
    confidence: float


@dataclass(frozen=True)
class SearchSummary:
    """Aggregate statistics for the best crafting path.

    Args:
        total_steps: Number of steps in the recommended path.
        total_cost: Cumulative currency cost of all steps.
        expected_final_reward: Mean reward at the end of the best path.
        success_probability: Estimated fraction of rollouts that reached the
            goal in the search tree.
        alternative_paths: Number of distinct paths explored at depth 1.
    """

    total_steps: int
    total_cost: float
    expected_final_reward: float
    success_probability: float
    alternative_paths: int


@dataclass(frozen=True)
class SearchMetadata:
    """Metadata about the MCTS run itself.

    Args:
        iterations: Total number of MCTS iterations executed.
        computation_time: Wall-clock time in seconds.
        tree_depth: Maximum depth reached in the search tree.
        nodes_explored: Total nodes created during the search.
    """

    iterations: int
    computation_time: float
    tree_depth: int
    nodes_explored: int


@dataclass
class SearchResult:
    """Complete result of a single MCTS optimisation run.

    Args:
        success: True when the best path reaches a goal state.
        best_path: Ordered sequence of recommended crafting actions.
        summary: Aggregate statistics for the best path.
        metadata: Runtime metadata of the MCTS run.
        strategy: Strategy name used for this run.
    """

    success: bool
    best_path: list[PathStep]
    summary: SearchSummary
    metadata: SearchMetadata
    strategy: str


# ---------------------------------------------------------------------------
# MCTS search engine
# ---------------------------------------------------------------------------


class MCTSSearch:
    """Monte Carlo Tree Search over the crafting MDP.

    Implements the standard four-phase MCTS loop:

    1. **Selection**      – descend the tree using the selection policy.
    2. **Expansion**      – create a child node for one unexplored action.
    3. **Simulation**     – random rollout from the expanded node.
    4. **Backpropagation** – propagate the rollout reward back to the root.

    The search terminates after *iterations* iterations or when the time limit
    is reached, whichever comes first.

    Args:
        transition_model: Domain transition function T(s, a) → s'.
        reward_function: Strategy for evaluating a state against the goal.
        selection_policy: UCB1 or other selection strategy (tree phase).
        simulation_policy: Policy for choosing actions during rollout.
        iterations: Number of MCTS iterations to run.
        max_simulation_depth: Maximum rollout depth during simulation.
        rng: Random number generator used for tie-breaking and expansion order.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        reward_function: RewardFunction,
        selection_policy: SelectionPolicy,
        simulation_policy: SimulationPolicy,
        iterations: int = 500,
        max_simulation_depth: int = 20,
        rng: random.Random | None = None,
    ) -> None:
        self._transition = transition_model
        self._reward = reward_function
        self._selection_policy = selection_policy
        self._simulation_policy = simulation_policy
        self._iterations = iterations
        self._max_simulation_depth = max_simulation_depth
        self._rng = rng if rng is not None else random.Random()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        initial_state: ItemState,
        goal: CraftingGoal,
        available_actions: list[CraftingAction],
        strategy: str = "balanced",
    ) -> SearchResult:
        """Run MCTS and return the best crafting path found.

        Args:
            initial_state: Starting item state.
            goal: Crafting goal.
            available_actions: Complete set of actions available for crafting.
            strategy: Name of the optimisation strategy (for result labelling).

        Returns:
            :class:`SearchResult` containing the best path and statistics.
        """
        start_time = time.perf_counter()
        root = MCTSNode(state=initial_state)
        nodes_explored = 1
        max_depth_reached = 0

        for _ in range(self._iterations):
            # --- 1. Selection ---
            node, depth = self._select(root, goal, available_actions)
            max_depth_reached = max(max_depth_reached, depth)

            # --- 2. Expansion ---
            if not self._is_terminal(node.state, goal):
                child = self._expand(node, available_actions)
                if child is not node:  # A new child was created.
                    node = child
                    nodes_explored += 1

            # --- 3. Simulation ---
            value = self._simulate(node.state, goal, available_actions)

            # --- 4. Backpropagation ---
            self._backpropagate(node, value)

        elapsed = time.perf_counter() - start_time

        best_path = self._extract_best_path(root)
        success = self._is_goal_met(
            best_path[-1].expected_reward if best_path else 0.0
        )

        summary = SearchSummary(
            total_steps=len(best_path),
            total_cost=sum(s.estimated_cost for s in best_path),
            expected_final_reward=(
                best_path[-1].expected_reward if best_path else 0.0
            ),
            success_probability=self._estimate_success_probability(root),
            alternative_paths=len(root.children),
        )
        metadata = SearchMetadata(
            iterations=self._iterations,
            computation_time=round(elapsed, 4),
            tree_depth=max_depth_reached,
            nodes_explored=nodes_explored,
        )
        return SearchResult(
            success=success,
            best_path=best_path,
            summary=summary,
            metadata=metadata,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Phase 1 – Selection
    # ------------------------------------------------------------------

    def _select(
        self,
        node: MCTSNode,
        goal: CraftingGoal,
        available_actions: list[CraftingAction],
    ) -> tuple[MCTSNode, int]:
        """Descend the tree until a leaf or unexpanded node is found.

        Args:
            node: Starting node (root).
            goal: Crafting goal.
            available_actions: All crafting actions.

        Returns:
            Tuple of (selected node, depth reached).
        """
        depth = 0
        while (
            not self._is_terminal(node.state, goal)
            and node.is_fully_expanded(
                self._applicable_action_ids(node.state, available_actions)
            )
            and node.children
        ):
            node = self._selection_policy.select_child(node)
            depth += 1
        return node, depth

    # ------------------------------------------------------------------
    # Phase 2 – Expansion
    # ------------------------------------------------------------------

    def _expand(
        self,
        node: MCTSNode,
        available_actions: list[CraftingAction],
    ) -> MCTSNode:
        """Create one new child node for an unexplored action.

        The unexplored action is chosen at random among those not yet tried
        from this node.  One next-state is sampled from the transition model.

        Args:
            node: Node to expand.
            available_actions: All crafting actions.

        Returns:
            The newly created child node, or *node* itself if no expansion
            is possible.
        """
        applicable = self._get_applicable_actions(node.state, available_actions)
        explored_ids = node.explored_action_ids()
        unexplored = [a for a in applicable if a.action_id not in explored_ids]

        if not unexplored:
            return node

        action = self._rng.choice(unexplored)
        next_state = self._transition.apply(node.state, action)
        child = MCTSNode(state=next_state, parent=node, action=action)
        node.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Phase 3 – Simulation (rollout)
    # ------------------------------------------------------------------

    def _simulate(
        self,
        state: ItemState,
        goal: CraftingGoal,
        available_actions: list[CraftingAction],
    ) -> float:
        """Run a random rollout from *state* and return the terminal reward.

        Args:
            state: Starting state of the rollout.
            goal: Crafting goal.
            available_actions: All crafting actions.

        Returns:
            Scalar reward from the reward function at the end of the rollout.
        """
        current = state
        for _ in range(self._max_simulation_depth):
            if self._is_terminal(current, goal):
                break
            applicable = self._get_applicable_actions(current, available_actions)
            if not applicable:
                break
            action = self._simulation_policy.select_action(
                current, applicable, goal
            )
            current = self._transition.apply(current, action)

        return self._reward.evaluate(current, goal)

    # ------------------------------------------------------------------
    # Phase 4 – Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate *value* from *node* up to the root.

        Args:
            node: Leaf node where the simulation terminated.
            value: Reward to propagate.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def _extract_best_path(self, root: MCTSNode) -> list[PathStep]:
        """Build the recommended crafting sequence from the search tree.

        Greedily follows the most-visited child from the root to a leaf,
        assembling a list of :class:`PathStep` objects.

        Args:
            root: Root node of the completed search tree.

        Returns:
            Ordered list of path steps.
        """
        path: list[PathStep] = []
        node = root
        step_number = 1
        total_visits = root.visits or 1

        while node.children:
            best_child = max(node.children, key=lambda c: c.visits)
            if best_child.action is None or best_child.visits == 0:
                break
            path.append(
                PathStep(
                    step=step_number,
                    action_id=best_child.action.action_id,
                    action_name=best_child.action.name,
                    estimated_cost=best_child.action.cost,
                    expected_reward=round(best_child.mean_value, 4),
                    confidence=round(best_child.visits / total_visits, 4),
                )
            )
            node = best_child
            step_number += 1

        return path

    # ------------------------------------------------------------------
    # Terminal state check
    # ------------------------------------------------------------------

    def _is_terminal(self, state: ItemState, goal: CraftingGoal) -> bool:
        """Return True when the search should not expand further from *state*.

        Args:
            state: State to evaluate.
            goal: Crafting goal.

        Returns:
            True if the state is a goal state, has reached the step limit, or
            the item is corrupted.
        """
        if state.is_corrupted:
            return True
        if state.step_count >= goal.max_steps:
            return True
        if goal.budget_limit is not None and (
            state.accumulated_cost >= goal.budget_limit
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_applicable_actions(
        self,
        state: ItemState,
        actions: list[CraftingAction],
    ) -> list[CraftingAction]:
        """Return the subset of *actions* applicable in *state*.

        Args:
            state: Current item state.
            actions: All crafting actions.

        Returns:
            List of applicable :class:`CraftingAction` objects.
        """
        return [a for a in actions if self._transition.can_apply(state, a)]

    def _applicable_action_ids(
        self,
        state: ItemState,
        actions: list[CraftingAction],
    ) -> frozenset[str]:
        """Return IDs of actions applicable in *state*.

        Args:
            state: Current item state.
            actions: All crafting actions.

        Returns:
            Frozenset of applicable action ID strings.
        """
        return frozenset(
            a.action_id for a in self._get_applicable_actions(state, actions)
        )

    @staticmethod
    def _is_goal_met(final_reward: float) -> bool:
        """Return True when the final reward indicates the goal was reached.

        A reward of 0.9 or above is treated as goal achievement.

        Args:
            final_reward: Reward value at the end of the best path.

        Returns:
            Boolean success indicator.
        """
        return final_reward >= 0.9

    @staticmethod
    def _estimate_success_probability(root: MCTSNode) -> float:
        """Estimate the probability of reaching the goal from the root.

        Uses the fraction of rollouts (visits) from the root whose cumulative
        value exceeds the success threshold (mean_value ≥ 0.9).

        Args:
            root: Root node of the completed search tree.

        Returns:
            Float in [0, 1].
        """
        if root.visits == 0:
            return 0.0
        return round(root.mean_value, 4)
