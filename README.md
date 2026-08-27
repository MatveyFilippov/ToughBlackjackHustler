# Tough Blackjack Hustler

**Tough Blackjack Hustler (TBJH)** is a Python framework for training and evaluating reinforcement learning agents for Blackjack. It implements a Q-Learning approach with a rich state representation that includes probability estimates, making it suitable for research and experimentation in game-playing AI.


## Features

- **Modular Architecture**: Clean separation between environment, agent, and Q-Learning components
- **Rich State Representation**: Includes probability estimates for player busting, dealer busting, and dealer drawing
- **Q-Learning with Epsilon-Greedy**: Standard Q-Learning algorithm with configurable exploration rate
- **Checkpoint Support**: Save and resume training with automatic checkpoints at configurable intervals
- **State Parsing**: Find closest matching states when exact states are not in the Q-Table
- **Evaluation Tools**: Built-in performance evaluation with win rate and expected value calculation
- **Basic Strategy Agent**: Reference implementation using standard Blackjack basic strategy
- **Extensible**: Easy to add new agents, environments, or Q-Learning strategies


## Project Structure

```
PythonProject/
├── LICENSE
├── requirements.txt
├── training.py                       # Main training/evaluation script
└── src/
    ├── __init__.py
    ├── __version__.py
    ├── agent/
    │   ├── base.py                   # Abstract Agent class
    │   └── for_simple_game/
    │       ├── by_basic_strategy.py  # Agent for SimpleGame (based on hardcoded basic strategy)
    │       └── by_q_table.py         # Agent for SimpleGame (based on provided Q-Table)
    ├── environment/
    │   ├── base.py                   # Core game components
    │   ├── probability_tools.py      # Probability calculations
    │   └── games/
    │       └── simple.py             # Simple Blackjack game implementation (just STAND or HIT)
    └── q_learning/
        ├── q_table.py                # Q-Table data structure
        ├── strategies/
        │   ├── base.py               # Abstract QLearner
        │   └── epsilon_greedy.py     # Q-Learning ε-greedy strategy
        └── utils/
            └── q_table_tools.py      # State parsing and narrowing tools for Q-Table
```


## Installation

1. Clone the repository:
```bash
git clone https://github.com/MatveyFilippov/ToughBlackjackHustler.git
cd ToughBlackjackHustler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

### Training a New Q-Table

```bash
python training.py train -e 100000 -d 6 -a 0.1 -g 0.95 -eps 0.1 -s q_table.pkl -c checkpoints/
```

**Options:**
- `-e, --episodes`: Number of training episodes (default: 10000)
- `-d, --decks`: Number of card decks (default: 6)
- `-a, --alpha`: Learning rate (default: 0.1)
- `-g, --gamma`: Discount factor (default: 0.95)
- `-eps, --epsilon`: Exploration rate (default: 0.1)
- `-s, --save`: Path to save the Q-Table (default: q_table.pkl)
- `--soft17`: Dealer hits on soft 17 (default: False)
- `-c, --checkpoint-dir`: Directory to save checkpoints (default: None - no checkpoints)
- `--checkpoint-interval`: Checkpoint interval as fraction of total episodes (default: 0.05 = 5%%)
- `-v, --verbose`: Enable verbose logging

### Continuing Training

```bash
python training.py continue q_table.pkl -e 50000 -a 0.05 -g 0.95 -eps 0.05 -c checkpoints/
```

**Options:**
- `q_table_path`: Path to existing Q-Table pickle file
- `-e, --episodes`: Additional episodes to train (default: 5000)
- `-d, --decks`: Number of card decks (default: 6)
- `-a, --alpha`: Learning rate (default: 0.05)
- `-g, --gamma`: Discount factor (default: 0.95)
- `-eps, --epsilon`: Exploration rate (default: 0.05)
- `-s, --save`: Path to save updated Q-Table (defaults to input path)
- `--soft17`: Dealer hits on soft 17 (default: False)
- `-c, --checkpoint-dir`: Directory to save checkpoints (default: None - no checkpoints)
- `--checkpoint-interval`: Checkpoint interval as fraction of total episodes (default: 0.05 = 5%%)
- `-v, --verbose`: Enable verbose logging

### Evaluating a Trained Q-Table

```bash
python training.py evaluate q_table.pkl -r 10000 -d 6
```

**Options:**
- `q_table_path`: Path to Q-Table pickle file
- `-r, --rounds`: Number of rounds for evaluation (default: 1000)
- `-d, --decks`: Number of card decks (default: 6)
- `-v, --verbose`: Enable verbose logging


## Architecture Details

### Game Environment (`SimpleGame`)

The `SimpleGame` class implements a simplified Blackjack environment with:
- Configurable number of card decks
- Configurable dealer behavior (soft 17)
- State representation including probability estimates
- Available actions: STAND, HIT

### State Representation (`SimpleGameState`)

Each state includes:
- Player hand: card count, sum, and whether it's a soft hand
- Probability of player busting on the next card
- Dealer's open card
- Probability of dealer's sum being less than 17
- Probability of dealer busting

### Q-Table

The Q-table stores Q-values for state-action pairs with:
- Support for arbitrary state types
- Efficient storage using NumPy arrays
- Serialization via pickle
- State parsing for finding nearest neighbors

### Agents

1. **AgentForSimpleGameByQTable**: Uses a trained Q-table to make decisions
2. **AgentForSimpleGameByBasicStrategy**: Implements standard Blackjack basic strategy with optional probability-based adjustments

### Q-Learning

- **EpsilonGreedyQLearner**: Implements ε-greedy exploration strategy
- Configurable reward structure for different game outcomes
- Standard Q-learning update: Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Development
The project wants to develop and welcome contributions from the community. Whether you want to:

- Implement a new feature
- Fix a bug or improve documentation
- Share ideas and suggestions

Please check our [Issues](https://github.com/MatveyFilippov/ToughBlackjackHustler/issues) page to see what's currently planned or to suggest changes.
