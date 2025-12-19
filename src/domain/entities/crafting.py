"""Path of Exile 2 crafting domain entities."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Rarity(str, Enum):
    """Item rarity levels."""
    NORMAL = "normal"
    MAGIC = "magic"
    RARE = "rare"
    UNIQUE = "unique"


class ModifierType(str, Enum):
    """Modifier type (prefix or suffix)."""
    PREFIX = "prefix"
    SUFFIX = "suffix"


@dataclass
class Modifier:
    """Represents a modifier on an item."""
    modifierId: str
    name: str
    tier: int
    value: float
    modifierType: ModifierType
    
    def __hash__(self) -> int:
        """Hash for set/dict operations."""
        return hash((self.modifierId, self.tier))
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, Modifier):
            return False
        return self.modifierId == other.modifierId and self.tier == other.tier


@dataclass
class ItemState:
    """Represents the current state of an item being crafted."""
    itemType: str
    itemLevel: int
    rarity: Rarity
    modifiers: List[Modifier] = field(default_factory=list)
    influence: Optional[str] = None
    baseStats: Dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False  # True if goal is reached or budget exceeded
    
    def __hash__(self) -> int:
        """Hash for state comparison."""
        # Create a hashable representation
        mods_tuple = tuple(sorted((m.modifierId, m.tier) for m in self.modifiers))
        return hash((self.itemType, self.itemLevel, self.rarity, mods_tuple, self.influence))
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, ItemState):
            return False
        return (self.itemType == other.itemType and
                self.itemLevel == other.itemLevel and
                self.rarity == other.rarity and
                set(self.modifiers) == set(other.modifiers) and
                self.influence == other.influence)
    
    def prefix_count(self) -> int:
        """Count of prefix modifiers."""
        return sum(1 for m in self.modifiers if m.modifierType == ModifierType.PREFIX)
    
    def suffix_count(self) -> int:
        """Count of suffix modifiers."""
        return sum(1 for m in self.modifiers if m.modifierType == ModifierType.SUFFIX)
    
    def modifier_count(self) -> int:
        """Total modifier count."""
        return len(self.modifiers)
    
    def has_modifier(self, modifier_id: str) -> bool:
        """Check if item has a specific modifier."""
        return any(m.modifierId == modifier_id for m in self.modifiers)
    
    def get_modifier(self, modifier_id: str) -> Optional[Modifier]:
        """Get a specific modifier if it exists."""
        for m in self.modifiers:
            if m.modifierId == modifier_id:
                return m
        return None


@dataclass
class CraftingAction:
    """Represents a crafting action (currency or bench craft)."""
    actionId: str
    actionType: str  # "currency", "bench", "other"
    name: str
    cost: float
    deterministic: bool
    description: str = ""
    prerequisites: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """Hash for action comparison."""
        return hash(self.actionId)
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, CraftingAction):
            return False
        return self.actionId == other.actionId


@dataclass
class CraftingGoal:
    """Represents the crafting goal."""
    goalType: str  # "exact", "partial", "score-based"
    targetModifiers: List[Dict[str, Any]]  # List of {modifierId, tier, weight}
    baseItem: Dict[str, Any]  # {itemType, itemLevel, influence}
    constraints: Dict[str, Any] = field(default_factory=dict)  # {maxSteps, budgetLimit}
    
    def matches_goal(self, state: ItemState) -> bool:
        """Check if state matches the goal."""
        if self.goalType == "exact":
            # All target modifiers must be present with exact tier
            for target in self.targetModifiers:
                mod = state.get_modifier(target["modifierId"])
                if not mod or mod.tier != target.get("tier", 1):
                    return False
            return True
        
        elif self.goalType == "partial":
            # At least some target modifiers should be present
            matches = sum(1 for target in self.targetModifiers 
                         if state.has_modifier(target["modifierId"]))
            return matches >= len(self.targetModifiers) // 2
        
        # For score-based, we don't check terminal here
        return False
    
    def calculate_reward(self, state: ItemState, modifier_weights: Dict[str, float]) -> float:
        """Calculate reward based on how close state is to goal."""
        reward = 0.0
        
        for target in self.targetModifiers:
            mod_id = target["modifierId"]
            weight = target.get("weight", 1.0)
            target_tier = target.get("tier", 1)
            
            mod = state.get_modifier(mod_id)
            if mod:
                # Reward for having the modifier
                tier_bonus = 1.0 if mod.tier >= target_tier else 0.5
                reward += weight * tier_bonus * modifier_weights.get(mod_id, 1.0)
        
        # Penalty for unwanted mods (not in target list)
        target_ids = {t["modifierId"] for t in self.targetModifiers}
        unwanted = [m for m in state.modifiers if m.modifierId not in target_ids]
        penalty = len(unwanted) * modifier_weights.get("penaltyForUnwantedMods", -0.5)
        
        return reward + penalty


@dataclass
class RewardSystem:
    """Reward system configuration."""
    rewardType: str  # "heuristic", "learned", "hybrid"
    scoringFunction: Dict[str, Any]
    terminalReward: Dict[str, float] = field(default_factory=lambda: {"successValue": 100.0, "failureValue": -10.0})
