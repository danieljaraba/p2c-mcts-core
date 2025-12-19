"""MCTS service for Path of Exile 2 crafting optimization."""
import random
from copy import deepcopy
from typing import List, Optional

from src.domain.entities.crafting import (
    CraftingAction,
    CraftingGoal,
    ItemState,
    Modifier,
    ModifierType,
    Rarity,
    RewardSystem,
)
from src.domain.entities.mcts_node import MCTSNode
from src.infrastructure.config_loader import config_loader


class CraftingMCTSService:
    """MCTS service for crafting optimization."""

    DEFAULT_MAX_SIMULATION_DEPTH = 50

    def __init__(
        self,
        exploration_weight: float = 1.41,
        max_simulation_depth: int = DEFAULT_MAX_SIMULATION_DEPTH,
    ):
        """
        Initialize crafting MCTS service.

        Args:
            exploration_weight: UCB1 exploration parameter
            max_simulation_depth: Maximum depth for simulation
        """
        self.exploration_weight = exploration_weight
        self.max_simulation_depth = max_simulation_depth
        self.root: Optional[MCTSNode] = None
        self.goal: Optional[CraftingGoal] = None
        self.reward_system: Optional[RewardSystem] = None
        self.available_actions: List[CraftingAction] = []
        self.budget_spent: float = 0.0

    def search(
        self,
        initial_state: ItemState,
        goal: CraftingGoal,
        reward_system: RewardSystem,
        available_actions: List[CraftingAction],
        num_simulations: int,
    ) -> Optional[CraftingAction]:
        """
        Perform MCTS search to find best crafting action.

        Args:
            initial_state: Starting item state
            goal: Crafting goal
            reward_system: Reward system configuration
            available_actions: Available crafting actions
            num_simulations: Number of simulations to run

        Returns:
            Best crafting action or None
        """
        self.goal = goal
        self.reward_system = reward_system
        self.available_actions = available_actions
        self.budget_spent = 0.0

        # Create root node
        valid_actions = self._get_valid_actions(initial_state)
        self.root = MCTSNode(state=initial_state, untried_actions=valid_actions)

        # Run simulations
        for _ in range(num_simulations):
            node = self._select(self.root)

            if not node.is_terminal():
                node = self._expand(node)

            reward = self._simulate(node)
            self._backpropagate(node, reward)

        # Return best action
        best_child = self.root.get_most_visited_child()
        return best_child.action if best_child else None

    def get_search_tree(self, root: MCTSNode) -> dict:
        """Get search tree structure."""
        return self._node_to_dict(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase using UCB1."""
        while not node.is_terminal() and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                break
            node = best_child
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase."""
        if not node.untried_actions:
            return node

        # Pick random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)

        # Apply action to get next state
        next_state = self._apply_action(node.state, action)
        if next_state is None:
            return node

        # Create and add child node
        valid_actions = self._get_valid_actions(next_state)
        child = node.add_child(next_state, action)
        child.untried_actions = valid_actions
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Simulation phase with random rollout."""
        current_state = deepcopy(node.state)
        total_reward = 0.0
        depth = 0
        step_cost = 0.0

        while not current_state.is_terminal and depth < self.max_simulation_depth:
            valid_actions = self._get_valid_actions(current_state)
            if not valid_actions:
                break

            # Random action selection
            action = random.choice(valid_actions)
            next_state = self._apply_action(current_state, action)

            if next_state is None:
                break

            # Calculate reward
            reward = self._calculate_reward(current_state, action, next_state)
            total_reward += reward
            step_cost += action.cost

            # Check budget constraint for this simulation path
            # step_cost accumulates cost within this rollout simulation
            if self.goal and self.goal.constraints.get("budgetLimit"):
                if step_cost >= self.goal.constraints["budgetLimit"]:
                    next_state.is_terminal = True
                    total_reward += self.reward_system.terminalReward.get(
                        "failureValue", -10.0
                    )

            current_state = next_state
            depth += 1

        # Terminal reward if goal reached
        if self.goal and self.goal.matches_goal(current_state):
            total_reward += self.reward_system.terminalReward.get("successValue", 100.0)

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagation phase."""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def _get_valid_actions(self, state: ItemState) -> List[CraftingAction]:
        """Get valid actions for a given state."""
        valid = []

        for action in self.available_actions:
            if self._can_apply_action(state, action):
                valid.append(action)

        return valid

    def _can_apply_action(self, state: ItemState, action: CraftingAction) -> bool:
        """Check if an action can be applied to a state."""
        prereqs = action.prerequisites.get("requiredState", {})

        # Check rarity requirement
        if "rarity" in prereqs and state.rarity != Rarity(prereqs["rarity"]):
            return False

        # Check min rarity
        if "minRarity" in prereqs:
            rarity_order = [Rarity.NORMAL, Rarity.MAGIC, Rarity.RARE, Rarity.UNIQUE]
            min_rarity = Rarity(prereqs["minRarity"])
            if rarity_order.index(state.rarity) < rarity_order.index(min_rarity):
                return False

        # Check max modifiers
        if "maxModifiers" in prereqs and state.modifier_count() > prereqs["maxModifiers"]:
            return False

        # Check min modifiers
        if "minModifiers" in prereqs and state.modifier_count() < prereqs["minModifiers"]:
            return False

        return True

    def _apply_action(
        self, state: ItemState, action: CraftingAction
    ) -> Optional[ItemState]:
        """Apply a crafting action to a state and return new state."""
        new_state = deepcopy(state)

        # Get currency data
        try:
            currency_data = config_loader.get_currency_by_id(action.actionId)
            transformation = currency_data["effects"]["outcomes"][0][
                "stateTransformation"
            ]

            trans_type = transformation.get("type")

            if trans_type == "reroll_all_mods":
                # Chaos orb - reroll all mods
                new_state.modifiers = self._generate_random_modifiers(
                    new_state, 4, 6
                )
                new_state.rarity = Rarity.RARE

            elif trans_type == "add_random_mod":
                # Exalted orb - add a random mod
                if new_state.modifier_count() < 6:
                    new_mod = self._generate_random_modifier(new_state)
                    if new_mod:
                        new_state.modifiers.append(new_mod)

            elif trans_type == "upgrade_to_rare":
                # Regal/Alchemy - upgrade to rare
                new_state.rarity = Rarity.RARE
                if transformation.get("addMod"):
                    new_mod = self._generate_random_modifier(new_state)
                    if new_mod:
                        new_state.modifiers.append(new_mod)
                elif transformation.get("addMods"):
                    # Alchemy adds 4-6 mods
                    new_state.modifiers = self._generate_random_modifiers(
                        new_state, 4, 6
                    )

            elif trans_type == "remove_all_mods":
                # Scouring - remove all mods
                new_state.modifiers = []
                new_state.rarity = Rarity.NORMAL

            elif trans_type == "remove_random_mod":
                # Annulment - remove random mod
                if new_state.modifiers:
                    new_state.modifiers.pop(random.randint(0, len(new_state.modifiers) - 1))

            elif trans_type == "upgrade_to_magic":
                # Transmutation - upgrade to magic
                new_state.rarity = Rarity.MAGIC
                new_state.modifiers = self._generate_random_modifiers(new_state, 1, 2)

            elif trans_type == "reroll_magic":
                # Alteration - reroll magic item
                new_state.modifiers = self._generate_random_modifiers(new_state, 1, 2)

            return new_state

        except Exception:
            # If action not found or error, return None
            return None

    def _generate_random_modifiers(
        self, state: ItemState, min_count: int, max_count: int
    ) -> List[Modifier]:
        """Generate random modifiers for an item."""
        count = random.randint(min_count, max_count)
        available = config_loader.get_available_modifiers_for_item(
            state.itemLevel, state.itemType
        )

        if not available:
            return []

        modifiers = []
        prefix_count = 0
        suffix_count = 0
        max_prefix = 3
        max_suffix = 3

        attempts = 0
        while len(modifiers) < count and attempts < count * 10:
            attempts += 1
            mod_data = random.choice(available)

            # Check prefix/suffix limits
            mod_type = ModifierType(mod_data["type"])
            if mod_type == ModifierType.PREFIX and prefix_count >= max_prefix:
                continue
            if mod_type == ModifierType.SUFFIX and suffix_count >= max_suffix:
                continue

            # Check if already have this modifier
            if any(m.modifierId == mod_data["modifierId"] for m in modifiers):
                continue

            # Create modifier
            modifier = Modifier(
                modifierId=mod_data["modifierId"],
                name=mod_data["name"],
                tier=mod_data["tier"],
                value=(mod_data["minValue"] + mod_data["maxValue"]) / 2.0,
                modifierType=mod_type,
            )

            modifiers.append(modifier)

            if mod_type == ModifierType.PREFIX:
                prefix_count += 1
            else:
                suffix_count += 1

        return modifiers

    def _generate_random_modifier(self, state: ItemState) -> Optional[Modifier]:
        """Generate a single random modifier."""
        mods = self._generate_random_modifiers(state, 1, 1)
        return mods[0] if mods else None

    def _calculate_reward(
        self, state: ItemState, action: CraftingAction, next_state: ItemState
    ) -> float:
        """Calculate reward for a state transition."""
        if not self.goal or not self.reward_system:
            return 0.0

        # Use goal's reward calculation
        modifier_weights = self.reward_system.scoringFunction.get("modifierWeights", {})

        # Add penalty for unwanted mods
        modifier_weights["penaltyForUnwantedMods"] = self.reward_system.scoringFunction.get(
            "penaltyForUnwantedMods", -0.5
        )

        reward = self.goal.calculate_reward(next_state, modifier_weights)

        # Subtract cost
        reward -= action.cost * 0.1  # Scale cost penalty

        return reward

    def _node_to_dict(self, node: MCTSNode) -> dict:
        """Convert node to dictionary."""
        state_info = {
            "itemType": node.state.itemType if hasattr(node.state, "itemType") else "unknown",
            "rarity": node.state.rarity.value if hasattr(node.state, "rarity") else "unknown",
            "modifierCount": len(node.state.modifiers) if hasattr(node.state, "modifiers") else 0,
        }

        return {
            "state": state_info,
            "visits": node.visits,
            "value": node.value,
            "action": node.action.actionId if node.action else None,
            "children": [self._node_to_dict(child) for child in node.children],
        }
