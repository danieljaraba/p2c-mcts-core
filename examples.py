"""Example usage of MCTS for Path of Exile 2 crafting optimization."""
from src.application.use_cases.crafting_mcts_service import CraftingMCTSService
from src.domain.entities.crafting import (
    CraftingGoal,
    ItemState,
    Rarity,
    RewardSystem,
)
from src.infrastructure.config_loader import config_loader


def chest_armour_crafting_example():
    """Example: Craft a chest armour with life and resistance."""
    print("=" * 70)
    print("Path of Exile 2 - MCTS Crafting Example: Chest Armour")
    print("=" * 70)

    # Initial state - a normal chest armour
    initial_state = ItemState(
        itemType="chest_armour",
        itemLevel=86,
        rarity=Rarity.NORMAL,
        modifiers=[],
        influence=None,
        baseStats={"armour": 100},
    )

    # Goal - get max life and fire resistance
    goal = CraftingGoal(
        goalType="partial",
        targetModifiers=[
            {"modifierId": "max_life", "tier": 3, "weight": 1.5},
            {"modifierId": "fire_resistance", "tier": 3, "weight": 1.0},
            {"modifierId": "cold_resistance", "tier": 2, "weight": 0.8},
        ],
        baseItem={
            "itemType": "chest_armour",
            "itemLevel": 86,
            "influence": None,
        },
        constraints={"maxSteps": 15, "budgetLimit": 100.0},
    )

    # Reward system
    reward_system = RewardSystem(
        rewardType="heuristic",
        scoringFunction={
            "modifierWeights": {
                "max_life": 1.5,
                "fire_resistance": 1.0,
                "cold_resistance": 0.8,
            },
            "penaltyForUnwantedMods": -0.3,
            "progressBonus": 0.0,
        },
        terminalReward={"successValue": 200.0, "failureValue": -20.0},
    )

    # Get available crafting actions
    available_actions = [
        config_loader.create_crafting_action(currency)
        for currency in config_loader.get_all_currencies()
    ]

    print("\nCrafting Setup:")
    print(f"  Base Item: {initial_state.itemType} (iLvl {initial_state.itemLevel})")
    print(f"  Initial Rarity: {initial_state.rarity.value}")
    print(f"\nGoal:")
    print(f"  Type: {goal.goalType}")
    print(f"  Target Modifiers:")
    for target in goal.targetModifiers:
        print(f"    - {target['modifierId']} (tier {target['tier']}, weight {target['weight']})")
    print(f"  Constraints:")
    print(f"    Max Steps: {goal.constraints['maxSteps']}")
    print(f"    Budget: {goal.constraints['budgetLimit']} chaos")

    print(f"\n  Available Currency Types: {len(available_actions)}")

    # Run MCTS
    print("\nRunning MCTS optimization (500 simulations)...")
    service = CraftingMCTSService(exploration_weight=1.41, max_simulation_depth=15)

    best_action = service.search(
        initial_state=initial_state,
        goal=goal,
        reward_system=reward_system,
        available_actions=available_actions,
        num_simulations=500,
    )

    # Display results
    print("\nOptimization Results:")
    if best_action:
        print(f"  Recommended First Step: {best_action.name}")
        print(f"    Action ID: {best_action.actionId}")
        print(f"    Cost: {best_action.cost} chaos")
        print(f"    Type: {best_action.actionType}")
    else:
        print("  No optimal action found")

    if service.root:
        print(f"\n  Search Tree Statistics:")
        print(f"    Total Simulations: {service.root.visits}")
        print(f"    Total Value: {service.root.value:.2f}")
        print(f"    Average Value: {service.root.value / service.root.visits:.2f}")
        print(f"    Branches Explored: {len(service.root.children)}")

        if service.root.children:
            print(f"\n  Top 5 Actions by Visit Count:")
            sorted_children = sorted(
                service.root.children, key=lambda c: c.visits, reverse=True
            )[:5]
            for i, child in enumerate(sorted_children, 1):
                avg_value = child.value / child.visits if child.visits > 0 else 0
                confidence = child.visits / service.root.visits
                print(
                    f"    {i}. {child.action.name:25} | "
                    f"Visits: {child.visits:4} ({confidence:5.1%}) | "
                    f"Avg Reward: {avg_value:7.2f} | "
                    f"Cost: {child.action.cost:5.2f}"
                )

    print("\n" + "=" * 70)


def weapon_crafting_example():
    """Example: Craft a one-hand sword for physical damage."""
    print("\n" + "=" * 70)
    print("Path of Exile 2 - MCTS Crafting Example: Physical Weapon")
    print("=" * 70)

    # Initial state
    initial_state = ItemState(
        itemType="one_hand_sword",
        itemLevel=75,
        rarity=Rarity.NORMAL,
        modifiers=[],
        baseStats={"physicalDamageMin": 15, "physicalDamageMax": 30},
    )

    # Goal - maximize physical damage
    goal = CraftingGoal(
        goalType="score-based",
        targetModifiers=[
            {"modifierId": "physical_damage", "tier": 3, "weight": 2.0},
            {"modifierId": "attack_speed", "tier": 2, "weight": 1.5},
        ],
        baseItem={
            "itemType": "one_hand_sword",
            "itemLevel": 75,
            "influence": None,
        },
        constraints={"maxSteps": 10, "budgetLimit": 50.0},
    )

    # Reward system
    reward_system = RewardSystem(
        rewardType="heuristic",
        scoringFunction={
            "modifierWeights": {
                "physical_damage": 2.0,
                "attack_speed": 1.5,
            },
            "penaltyForUnwantedMods": -0.5,
        },
    )

    # Get available actions
    available_actions = [
        config_loader.create_crafting_action(currency)
        for currency in config_loader.get_all_currencies()
    ]

    print(f"\nBase: {initial_state.itemType} (iLvl {initial_state.itemLevel})")
    print(f"Goal: Physical damage weapon")
    print(f"Budget: {goal.constraints['budgetLimit']} chaos\n")

    # Run optimization
    print("Running MCTS (300 simulations)...")
    service = CraftingMCTSService(exploration_weight=1.2)
    best_action = service.search(
        initial_state, goal, reward_system, available_actions, 300
    )

    if best_action:
        print(f"\nRecommended: {best_action.name} ({best_action.cost} chaos)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run examples
    chest_armour_crafting_example()
    weapon_crafting_example()

    print("\n✓ Examples completed successfully!")
    print("\nTo run the API server:")
    print("  python -m src.main")
    print("\nThen visit:")
    print("  http://localhost:8000/docs  (API documentation)")
    print("  http://localhost:8000/health  (Health check)")

