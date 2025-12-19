# P2C MCTS Core

A microservice implementation of Monte Carlo Tree Search (MCTS) algorithm with Markov Decision Process (MDP) using FastAPI and Hexagonal Architecture.

## Features

- **MCTS Algorithm**: Full implementation of Monte Carlo Tree Search with UCB1 selection
- **MDP Support**: Complete Markov Decision Process model with states, actions, and transitions
- **Hexagonal Architecture**: Clean separation of concerns with domain, application, and infrastructure layers
- **FastAPI**: Modern, fast web framework with automatic API documentation
- **Python 3.11**: Latest Python features and performance improvements

## Architecture

The project follows Hexagonal Architecture (Ports and Adapters):

```
src/
├── domain/              # Core business logic
│   └── entities/        # Domain entities (MDP, State, Action, MCTSNode)
├── application/         # Use cases and ports
│   ├── ports/          # Interface definitions
│   └── use_cases/      # MCTS algorithm implementation
└── infrastructure/      # External adapters
    └── adapters/
        ├── api/        # FastAPI REST endpoints
        └── persistence/ # Repository implementations
```

### Layers

1. **Domain Layer**: Contains core entities (MDP, State, Action, MCTSNode) with no external dependencies
2. **Application Layer**: Implements business logic (MCTS algorithm) and defines ports (interfaces)
3. **Infrastructure Layer**: Provides adapters for external systems (FastAPI, repositories)

## Installation

### Requirements

- Python 3.11 or higher
- pip or poetry

### Setup

```bash
# Clone the repository
git clone https://github.com/danieljaraba/p2c-mcts-core.git
cd p2c-mcts-core

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Usage

### Starting the Server

```bash
# Using Python
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### Health Check

```bash
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

#### MCTS Search

```bash
POST /api/v1/mcts/search
```

Request body:
```json
{
  "mdp": {
    "states": [
      {"id": "s1", "data": {}, "is_terminal": false},
      {"id": "s2", "data": {}, "is_terminal": true}
    ],
    "actions": [
      {"id": "a1", "name": "move", "parameters": {}}
    ],
    "transitions": [
      {
        "from_state_id": "s1",
        "action_id": "a1",
        "to_state_id": "s2",
        "reward": 10.0,
        "probability": 1.0
      }
    ],
    "initial_state_id": "s1",
    "gamma": 0.95
  },
  "num_simulations": 100,
  "exploration_weight": 1.41
}
```

Response:
```json
{
  "best_action": {
    "id": "a1",
    "name": "move",
    "parameters": {}
  },
  "search_tree": {
    "state_id": "s1",
    "visits": 100,
    "value": 950.0,
    "action": null,
    "children": [...]
  },
  "simulations_run": 100
}
```

## MCTS Algorithm

The implementation follows the standard MCTS algorithm with four phases:

1. **Selection**: Traverse the tree using UCB1 formula to balance exploration/exploitation
2. **Expansion**: Add a new child node for an untried action
3. **Simulation**: Perform a random rollout from the new node
4. **Backpropagation**: Update statistics for all nodes in the path

### UCB1 Formula

```
UCB1 = (Q/N) + c * sqrt(ln(N_parent) / N)
```

Where:
- Q: Total reward for the node
- N: Number of visits to the node
- c: Exploration weight (default: √2 ≈ 1.41)
- N_parent: Number of visits to parent node

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_mcts_service.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Project Structure

```
p2c-mcts-core/
├── src/
│   ├── domain/
│   │   └── entities/
│   │       ├── mdp.py              # MDP, State, Action, Transition
│   │       └── mcts_node.py        # MCTS tree node
│   ├── application/
│   │   ├── ports/
│   │   │   └── input_ports.py      # Port interfaces
│   │   └── use_cases/
│   │       └── mcts_service.py     # MCTS algorithm
│   ├── infrastructure/
│   │   └── adapters/
│   │       ├── api/
│   │       │   ├── models.py       # Pydantic models
│   │       │   └── routes.py       # FastAPI routes
│   │       └── persistence/
│   │           └── in_memory_mdp_repository.py
│   └── main.py                     # FastAPI application
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Examples

### Simple Grid World

```python
from src.domain.entities.mdp import State, Action, MDP, Transition
from src.application.use_cases.mcts_service import MCTSService

# Define states
s1 = State(id="start", data={"position": (0, 0)})
s2 = State(id="goal", data={"position": (1, 1)}, is_terminal=True)

# Define actions
move_right = Action(id="right", name="Move Right")
move_down = Action(id="down", name="Move Down")

# Create MDP
mdp = MDP(
    states=[s1, s2],
    actions=[move_right, move_down],
    initial_state=s1,
    gamma=0.95
)

# Add transitions
mdp.add_transition(Transition(s1, move_right, s2, reward=10.0))
mdp.add_transition(Transition(s1, move_down, s2, reward=8.0))

# Run MCTS
service = MCTSService()
best_action = service.search(mdp, s1, num_simulations=1000)
print(f"Best action: {best_action.name}")
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Author

Daniel Jaraba

## References

- [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)