"""Test script for Path of Exile 2 crafting MCTS."""
from src.application.use_cases.crafting_mcts_service import CraftingMCTSService
from src.domain.entities.crafting import (
    CraftingGoal,
    ItemState,
    Rarity,
    RewardSystem,
)
from src.infrastructure.config_loader import config_loader


def test_basic_crafting():
    """Test basic crafting optimization."""
    print("=" * 60)
    print("Path of Exile 2 - MCTS Crafting Test")
    print("=" * 60)

    # Create initial item state
    initial_state = ItemState(
        itemType="chest_armour",
        itemLevel=86,
        rarity=Rarity.NORMAL,
        modifiers=[],
        influence=None,
        baseStats={"armour": 100},
        is_terminal=False,
    )

    # Create crafting goal
    goal = CraftingGoal(
        goalType="partial",
        targetModifiers=[
            {"modifierId": "max_life", "tier": 3, "weight": 1.0},
            {"modifierId": "fire_resistance", "tier": 2, "weight": 0.8},
        ],
        baseItem={
            "itemType": "chest_armour",
            "itemLevel": 86,
            "influence": None,
        },
        constraints={"maxSteps": 10, "budgetLimit": 50.0},
    )

    # Create reward system
    reward_system = RewardSystem(
        rewardType="heuristic",
        scoringFunction={
            "modifierWeights": {"max_life": 1.0, "fire_resistance": 0.8},
            "penaltyForUnwantedMods": -0.5,
        },
        terminalReward={"successValue": 100.0, "failureValue": -10.0},
    )

    # Get available actions from config
    available_actions = [
        config_loader.create_crafting_action(currency)
        for currency in config_loader.get_all_currencies()
    ]

    print(f"\nInitial State:")
    print(f"  Item: {initial_state.itemType}")
    print(f"  Item Level: {initial_state.itemLevel}")
    print(f"  Rarity: {initial_state.rarity.value}")
    print(f"  Modifiers: {len(initial_state.modifiers)}")

    print(f"\nGoal:")
    print(f"  Type: {goal.goalType}")
    print(f"  Target Modifiers: {len(goal.targetModifiers)}")
    print(f"  Max Steps: {goal.constraints['maxSteps']}")
    print(f"  Budget: {goal.constraints['budgetLimit']} chaos")

    print(f"\nAvailable Actions: {len(available_actions)}")

    # Run MCTS
    print("\nRunning MCTS (100 simulations)...")
    service = CraftingMCTSService(exploration_weight=1.41, max_simulation_depth=10)

    best_action = service.search(
        initial_state=initial_state,
        goal=goal,
        reward_system=reward_system,
        available_actions=available_actions,
        num_simulations=100,
    )

    print("\nResults:")
    if best_action:
        print(f"  Best Action: {best_action.name} (ID: {best_action.actionId})")
        print(f"  Cost: {best_action.cost} chaos")
    else:
        print("  No action recommended")

    if service.root:
        print(f"\nTree Statistics:")
        print(f"  Root Visits: {service.root.visits}")
        print(f"  Root Value: {service.root.value:.2f}")
        print(f"  Children: {len(service.root.children)}")

        if service.root.children:
            print(f"\n  Top Children:")
            sorted_children = sorted(
                service.root.children, key=lambda c: c.visits, reverse=True
            )[:5]
            for i, child in enumerate(sorted_children, 1):
                avg_value = child.value / child.visits if child.visits > 0 else 0
                print(
                    f"    {i}. {child.action.name:20} | "
                    f"Visits: {child.visits:4} | "
                    f"Avg Value: {avg_value:6.2f}"
                )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_basic_crafting()
    print("\n✓ Test completed successfully!")
