"""Configuration loader for game data."""
import json
from pathlib import Path
from typing import Any, Dict, List

from src.domain.entities.crafting import CraftingAction, Modifier, ModifierType


class ConfigLoader:
    """Loads configuration from JSON files."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize with config directory path."""
        self.config_dir = Path(config_dir)
        self._currencies: List[Dict[str, Any]] = []
        self._modifiers: List[Dict[str, Any]] = []
        self._base_items: List[Dict[str, Any]] = []
        self._load_all()
    
    def _load_all(self) -> None:
        """Load all configuration files."""
        self._load_currencies()
        self._load_modifiers()
        self._load_base_items()
    
    def _load_currencies(self) -> None:
        """Load currency items from JSON."""
        currency_file = self.config_dir / "currencies" / "currency_items.json"
        if currency_file.exists():
            with open(currency_file, "r") as f:
                data = json.load(f)
                self._currencies = data.get("currencies", [])
    
    def _load_modifiers(self) -> None:
        """Load modifiers from JSON."""
        modifier_file = self.config_dir / "modifiers" / "modifiers.json"
        if modifier_file.exists():
            with open(modifier_file, "r") as f:
                data = json.load(f)
                self._modifiers = data.get("modifiers", [])
    
    def _load_base_items(self) -> None:
        """Load base items from JSON."""
        base_items_file = self.config_dir / "base_items" / "base_items.json"
        if base_items_file.exists():
            with open(base_items_file, "r") as f:
                data = json.load(f)
                self._base_items = data.get("baseItems", [])
    
    def get_all_currencies(self) -> List[Dict[str, Any]]:
        """Get all currency items."""
        return self._currencies
    
    def get_currency_by_id(self, action_id: str) -> Dict[str, Any]:
        """Get currency by ID."""
        for currency in self._currencies:
            if currency["actionId"] == action_id:
                return currency
        raise ValueError(f"Currency {action_id} not found")
    
    def get_all_modifiers(self) -> List[Dict[str, Any]]:
        """Get all modifiers."""
        return self._modifiers
    
    def get_modifier_by_id(self, modifier_id: str) -> Dict[str, Any]:
        """Get modifier by ID."""
        for modifier in self._modifiers:
            if modifier["modifierId"] == modifier_id:
                return modifier
        raise ValueError(f"Modifier {modifier_id} not found")
    
    def get_modifier_tier(self, modifier_id: str, tier: int) -> Dict[str, Any]:
        """Get specific tier of a modifier."""
        modifier = self.get_modifier_by_id(modifier_id)
        for tier_data in modifier.get("tiers", []):
            if tier_data["tier"] == tier:
                return tier_data
        raise ValueError(f"Tier {tier} not found for modifier {modifier_id}")
    
    def get_all_base_items(self) -> List[Dict[str, Any]]:
        """Get all base items."""
        return self._base_items
    
    def get_base_item(self, item_type: str) -> Dict[str, Any]:
        """Get base item by type."""
        for item in self._base_items:
            if item["itemType"] == item_type:
                return item
        raise ValueError(f"Base item {item_type} not found")
    
    def get_available_modifiers_for_item(
        self, 
        item_level: int, 
        item_type: str
    ) -> List[Dict[str, Any]]:
        """Get modifiers available for an item based on item level."""
        available = []
        for modifier in self._modifiers:
            for tier_data in modifier.get("tiers", []):
                if tier_data["itemLevel"] <= item_level:
                    available.append({
                        "modifierId": modifier["modifierId"],
                        "name": modifier["name"],
                        "type": modifier["type"],
                        "tier": tier_data["tier"],
                        "weight": tier_data["weight"],
                        "minValue": tier_data["minValue"],
                        "maxValue": tier_data["maxValue"]
                    })
        return available
    
    def create_crafting_action(self, action_data: Dict[str, Any]) -> CraftingAction:
        """Create a CraftingAction from JSON data."""
        return CraftingAction(
            actionId=action_data["actionId"],
            actionType=action_data["actionType"],
            name=action_data["name"],
            cost=action_data["cost"],
            deterministic=action_data["effects"]["deterministic"],
            description=action_data.get("description", ""),
            prerequisites=action_data.get("prerequisites", {})
        )
    
    def create_modifier(
        self, 
        modifier_id: str, 
        tier: int
    ) -> Modifier:
        """Create a Modifier entity from configuration."""
        modifier_data = self.get_modifier_by_id(modifier_id)
        tier_data = self.get_modifier_tier(modifier_id, tier)
        
        # Use average of min and max for the value
        value = (tier_data["minValue"] + tier_data["maxValue"]) / 2.0
        
        return Modifier(
            modifierId=modifier_id,
            name=modifier_data["name"],
            tier=tier,
            value=value,
            modifierType=ModifierType(modifier_data["type"])
        )


# Global config loader instance
config_loader = ConfigLoader()
