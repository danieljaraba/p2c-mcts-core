# Path of Exile 2 - MCTS Crafting Core Service

## Description

This microservice implements the core decision-making logic for optimizing item crafting in Path of Exile 2 using **Monte Carlo Tree Search (MCTS)** over a **Markov Decision Process (MDP)** model. 

Given a crafting goal, reward system, and available crafting actions, the service computes the optimal sequence of crafting steps to achieve the desired item outcome.

## Overview

Path of Exile 2 features a complex crafting system where players can modify items through various currency items and crafting methods. Finding the optimal crafting path is challenging due to: 

- **Probabilistic outcomes**: Many crafting actions have random results
- **Large state space**: Items can have numerous possible modifier combinations
- **Complex dependencies**: Some crafts unlock or prevent other modifications
- **Resource constraints**: Crafting materials have varying costs and availability

This service models the crafting problem as an MDP and uses MCTS to efficiently explore the decision tree and find near-optimal crafting sequences.

## Architecture

### Markov Decision Process (MDP) Components

The crafting process is modeled as an MDP with: 

- **State (S)**: Current item state (base type, modifiers, influence, etc.)
- **Actions (A)**: Available crafting operations (currency usage, bench crafts, etc.)
- **Transition Function (T)**: Probability distribution of resulting states given current state and action
- **Reward Function (R)**: Evaluation of how close a state is to the crafting goal
- **Goal State**:  Target item configuration defined by desired modifiers

### Monte Carlo Tree Search (MCTS)

The MCTS algorithm explores the decision tree through four phases:

1. **Selection**: Navigate the tree using UCB1 (Upper Confidence Bound) policy
2. **Expansion**: Add new child nodes for unexplored actions
3. **Simulation**: Run random rollout to estimate action value
4. **Backpropagation**: Update node statistics with simulation results

## API Contract

### Input

The service consumes three primary inputs:

#### 1. Goal Definition
```json
{
  "goalType": "exact|partial|score-based",
  "targetModifiers": [
    {
      "modifierId": "string",
      "tier": "integer",
      "weight": "float"
    }
  ],
  "baseItem": {
    "itemType": "string",
    "itemLevel": "integer",
    "influence": "string|null"
  },
  "constraints": {
    "maxSteps": "integer",
    "budgetLimit": "float"
  }
}
```

#### 2. Reward System
```json
{
  "rewardType": "heuristic|learned|hybrid",
  "scoringFunction": {
    "modifierWeights": "map<string, float>",
    "penaltyForUnwantedMods": "float",
    "progressBonus": "float"
  },
  "terminalReward": {
    "successValue": "float",
    "failureValue": "float"
  }
}
```

#### 3. Available Actions (Movements)
```json
{
  "actions": [
    {
      "actionId": "string",
      "actionType": "currency|bench|other",
      "name": "string",
      "cost": "float",
      "effects": {
        "deterministic": "boolean",
        "outcomes": [
          {
            "probability": "float",
            "stateTransformation": "object"
          }
        ]
      },
      "prerequisites": {
        "requiredState": "object",
        "blockedBy": "array<string>"
      }
    }
  ]
}
```

### Output

The service returns the optimal crafting path: 

```json
{
  "success": "boolean",
  "bestPath": [
    {
      "step": "integer",
      "actionId": "string",
      "actionName": "string",
      "estimatedCost": "float",
      "expectedReward": "float",
      "confidence": "float"
    }
  ],
  "summary": {
    "totalSteps": "integer",
    "totalCost": "float",
    "expectedFinalReward": "float",
    "successProbability": "float",
    "alternativePaths": "integer"
  },
  "metadata": {
    "iterations": "integer",
    "computationTime": "float",
    "treeDepth": "integer",
    "nodesExplored": "integer"
  }
}
```

## Technology Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI (for REST API)
- **Core Libraries**:
  - NumPy (numerical computations)
  - NetworkX (tree/graph structures)
  - Pydantic (data validation)
- **Optional**:
  - Redis (caching common game states)
  - PostgreSQL (storing crafting knowledge base)

## Project Structure

```
seek-client-manager/
├── src/
│   ├── core/
│   │   ├── mdp/
│   │   │   ├── state.py          # Item state representation
│   │   │   ├── action.py         # Crafting action definitions
│   │   │   ├── transition.py     # State transition logic
│   │   │   └── reward.py         # Reward function implementation
│   │   ├── mcts/
│   │   │   ├── node.py           # MCTS tree node
│   │   │   ├── tree.py           # MCTS tree structure
│   │   │   ├── search.py         # MCTS algorithm implementation
│   │   │   └── policies.py       # Selection/expansion policies
│   │   └── optimizer.py          # Main optimization orchestrator
│   ├── api/
│   │   ├── routes. py             # API endpoints
│   │   ├── models.py             # Request/response schemas
│   │   └── validators.py         # Input validation
│   ├── config/
│   │   └── settings.py           # Configuration management
│   └── utils/
│       ├── logger.py             # Logging utilities
│       └── metrics.py            # Performance tracking
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── mcts_algorithm.md
│   ├── mdp_model.md
│   └── api_reference.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/danieljaraba/p2c-mcts-core.git
cd p2c-mcts-core
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally

#### Option 1: Direct Python Execution
```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

#### Option 2: Docker
```bash
docker-compose up --build -d
```

Access the API documentation at: 
```
http://localhost:8000/docs
```

## Configuration

Key configuration parameters in `src/config/settings.py`:

- `MCTS_ITERATIONS`: Number of MCTS iterations (default: 10000)
- `EXPLORATION_CONSTANT`: UCB1 exploration parameter (default: 1.414)
- `MAX_TREE_DEPTH`: Maximum search depth (default: 50)
- `SIMULATION_POLICY`: Rollout strategy ("random", "heuristic", "learned")
- `ENABLE_CACHING`: Cache computed states (default: True)

## API Usage Example

```python
import requests

payload = {
    "goal": {
        "goalType":  "partial",
        "targetModifiers":  [
            {"modifierId": "life_regen", "tier": 1, "weight": 1.0},
            {"modifierId": "crit_chance", "tier": 2, "weight": 0.8}
        ],
        "baseItem": {
            "itemType": "chest_armour",
            "itemLevel": 86,
            "influence": "hunter"
        }
    },
    "rewardSystem": {
        "rewardType": "heuristic",
        "scoringFunction": {
            "modifierWeights":  {"life_regen": 1.0, "crit_chance":  0.8},
            "penaltyForUnwantedMods": -0.5
        }
    },
    "actions": [
        {
            "actionId": "chaos_orb",
            "actionType":  "currency",
            "name":  "Chaos Orb",
            "cost": 1.5,
            "effects": {
                "deterministic": False,
                "outcomes": [{"probability": 1.0, "stateTransformation": {"reroll": "all_mods"}}]
            }
        }
    ]
}

response = requests.post("http://localhost:8000/api/v1/optimize", json=payload)
print(response.json())
```

## Development Roadmap

- [x] Project initialization
- [ ] MDP state representation
- [ ] Action and transition modeling
- [ ] Reward function implementation
- [ ] MCTS core algorithm
- [ ] REST API with FastAPI
- [ ] Integration with PoE2 game data
- [ ] Performance optimization (parallelization, pruning)
- [ ] Caching layer for common scenarios
- [ ] Web UI for visualization
- [ ] Advanced reward learning (reinforcement learning)

## Contributing

Contributions are welcome! Please: 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. 

## Acknowledgments

- Path of Exile 2 by Grinding Gear Games
- MCTS algorithm research and implementations
- Open source Python community

## Contact

Daniel Jaraba - [@danieljaraba](https://github.com/danieljaraba)

Project Link: [https://github.com/danieljaraba/p2c-mcts-core](https://github.com/danieljaraba/p2c-mcts-core)
