"""Example usage of MCTS with MDP."""
from src.application.use_cases.mcts_service import MCTSService
from src.domain.entities.mdp import Action, MDP, State, Transition


def simple_grid_world_example():
    """Example: Simple 2x2 grid world navigation."""
    print("=" * 60)
    print("MCTS Example: Simple Grid World Navigation")
    print("=" * 60)

    # Define states (positions in grid)
    s_start = State(id="0,0", data={"x": 0, "y": 0}, is_terminal=False)
    s_middle1 = State(id="1,0", data={"x": 1, "y": 0}, is_terminal=False)
    s_middle2 = State(id="0,1", data={"x": 0, "y": 1}, is_terminal=False)
    s_goal = State(id="1,1", data={"x": 1, "y": 1}, is_terminal=True)

    # Define actions
    move_right = Action(id="right", name="Move Right")
    move_down = Action(id="down", name="Move Down")
    move_up = Action(id="up", name="Move Up")
    move_left = Action(id="left", name="Move Left")

    # Create MDP
    mdp = MDP(
        states=[s_start, s_middle1, s_middle2, s_goal],
        actions=[move_right, move_down, move_up, move_left],
        initial_state=s_start,
        gamma=0.95,
    )

    # Add transitions with rewards
    # From start (0,0)
    mdp.add_transition(Transition(s_start, move_right, s_middle1, reward=-1.0))
    mdp.add_transition(Transition(s_start, move_down, s_middle2, reward=-1.0))

    # From middle1 (1,0)
    mdp.add_transition(Transition(s_middle1, move_down, s_goal, reward=100.0))
    mdp.add_transition(Transition(s_middle1, move_left, s_start, reward=-1.0))

    # From middle2 (0,1)
    mdp.add_transition(Transition(s_middle2, move_right, s_goal, reward=100.0))
    mdp.add_transition(Transition(s_middle2, move_up, s_start, reward=-1.0))

    print("\nMDP Configuration:")
    print(f"  States: {len(mdp.states)}")
    print(f"  Actions: {len(mdp.actions)}")
    print(f"  Transitions: {len(mdp.transitions)}")
    print(f"  Initial State: {mdp.initial_state.id}")
    print(f"  Discount Factor (gamma): {mdp.gamma}")

    # Run MCTS
    print("\nRunning MCTS...")
    service = MCTSService(exploration_weight=1.41)
    num_simulations = 500
    best_action = service.search(mdp, s_start, num_simulations)

    # Display results
    print(f"\nMCTS Results ({num_simulations} simulations):")
    if best_action:
        print(f"  Best Action: {best_action.name} (ID: {best_action.id})")
    else:
        print("  No action found (terminal state?)")

    # Display search tree statistics
    if service.root:
        print(f"\nSearch Tree Statistics:")
        print(f"  Root visits: {service.root.visits}")
        print(f"  Root value: {service.root.value:.2f}")
        print(f"  Number of children: {len(service.root.children)}")

        if service.root.children:
            print("\n  Child nodes:")
            for child in service.root.children:
                avg_value = child.value / child.visits if child.visits > 0 else 0
                print(
                    f"    Action: {child.action.name:12} | "
                    f"Visits: {child.visits:4} | "
                    f"Total Value: {child.value:8.2f} | "
                    f"Avg Value: {avg_value:6.2f}"
                )

    print("\n" + "=" * 60)


def decision_problem_example():
    """Example: Simple decision problem with two choices."""
    print("\n" + "=" * 60)
    print("MCTS Example: Simple Decision Problem")
    print("=" * 60)

    # States
    s_initial = State(id="initial", is_terminal=False)
    s_good = State(id="good_outcome", is_terminal=True)
    s_bad = State(id="bad_outcome", is_terminal=True)

    # Actions
    a_risky = Action(id="risky", name="Risky Choice")
    a_safe = Action(id="safe", name="Safe Choice")

    # Create MDP
    mdp = MDP(
        states=[s_initial, s_good, s_bad],
        actions=[a_risky, a_safe],
        initial_state=s_initial,
        gamma=1.0,  # No discount for single-step problem
    )

    # Risky choice: high reward but uncertain
    mdp.add_transition(Transition(s_initial, a_risky, s_good, reward=100.0))
    
    # Safe choice: moderate reward
    mdp.add_transition(Transition(s_initial, a_safe, s_good, reward=50.0))

    print("\nProblem: Choose between risky (reward=100) and safe (reward=50)")

    # Run MCTS
    service = MCTSService(exploration_weight=1.0)
    num_simulations = 200
    best_action = service.search(mdp, s_initial, num_simulations)

    print(f"\nMCTS Recommendation ({num_simulations} simulations):")
    if best_action:
        print(f"  Recommended Action: {best_action.name}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run examples
    simple_grid_world_example()
    decision_problem_example()
    
    print("\n✓ Examples completed successfully!")
    print("\nTo run the API server, use:")
    print("  python -m src.main")
    print("\nThen visit http://localhost:8000/docs for interactive API documentation")
