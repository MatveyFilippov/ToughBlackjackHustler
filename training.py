#!/usr/bin/env python3
"""
Training script for Q-Table in the Tough Blackjack Hustler project.
"""

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
from src.agent.for_simple_game import AgentForSimpleGameByQTable
from src.environment.base import GameActionResult
from src.environment.games.simple import SimpleGame
from src.q_learning import QTable
from src.q_learning.strategies import EpsilonGreedyQLearner
from src.q_learning.strategies.base import QLearnerRewardAfterAction


def setup_logging(verbose: bool = False):
    """Configure logging for the training process."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.Formatter.formatTime = (
        lambda self, record, datefmt=None: (
            datetime
            .fromtimestamp(record.created, tz=timezone.utc)
            .astimezone(datetime.now().tzinfo)
            .isoformat(timespec='milliseconds')
        )
    )
    logging.basicConfig(
        encoding="UTF-8",
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def train_q_table(
    episodes: int,
    card_decks_qty: int,
    dealer_hit_on_soft_17: bool,
    alpha: float,
    gamma: float,
    epsilon: float,
    save_path: Path,
    verbose: bool = False,
) -> QTable:
    """
    Train a Q-Table for the Simple Blackjack game.

    Args:
        episodes: Number of training episodes
        card_decks_qty: Number of card decks to use
        dealer_hit_on_soft_17: Whether dealer hits on soft 17
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy
        save_path: Path to save the trained Q-Table
        verbose: Enable verbose logging

    Returns:
        Trained QTable instance
    """
    setup_logging(verbose)
    log = logging.getLogger("tbjh.train")

    log.info("=" * 60)
    log.info("Starting Q-Table Training")
    log.info("=" * 60)
    log.info(f"Episodes: {episodes}")
    log.info(f"Card Decks: {card_decks_qty}")
    log.info(f"Dealer Hits Soft 17: {dealer_hit_on_soft_17}")
    log.info(f"Alpha (learning rate): {alpha}")
    log.info(f"Gamma (discount factor): {gamma}")
    log.info(f"Epsilon (exploration rate): {epsilon}")
    log.info(f"Save path: {save_path}")
    log.info("=" * 60)

    # Create game environment
    game = SimpleGame(
        card_decks_qty=card_decks_qty,
        dealer_hit_on_soft_17=dealer_hit_on_soft_17,
    )

    # Define rewards for each action result
    # Blackjack pays 3:2, wins pay 1:1, pushes return the bet, losses lose the bet
    rewards = {
        GameActionResult.WAIT_ACTION: QLearnerRewardAfterAction(0.0),
        GameActionResult.BLACKJACK: QLearnerRewardAfterAction(1.5),  # 3:2 payout
        GameActionResult.WINS: QLearnerRewardAfterAction(1.0),
        GameActionResult.PUSH: QLearnerRewardAfterAction(0.0),
        GameActionResult.LOSS: QLearnerRewardAfterAction(-1.0),
        GameActionResult.BUST: QLearnerRewardAfterAction(-1.0),
    }

    # Initialize Q-Table
    q_table = QTable(*game.available_actions)

    # Create Q-Learner with epsilon-greedy strategy
    learner = EpsilonGreedyQLearner(
        game_environment=game,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        agent_rewards=rewards,
        q_table=q_table,
    )

    # Training loop
    log.info("Starting training...")
    progress_interval = max(1, episodes // 20)  # Show progress every 5%

    for episode in range(1, episodes + 1):
        learner.run_train_iteration()

        # Progress reporting
        if episode % progress_interval == 0 or episode == episodes:
            progress = (episode / episodes) * 100
            q_table_size = len(q_table)
            log.info(f"Progress: {progress:.1f}% ({episode}/{episodes}) - Q-Table size: {q_table_size} states")

    log.info("Training completed!")
    log.info(f"Final Q-Table size: {len(q_table)} states")

    # Save the trained Q-Table
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        q_table.save(str(save_path))
        log.info(f"Q-Table saved to {save_path}")
    except Exception as ex:
        log.error(f"Failed to save Q-Table: {ex}", exc_info=ex)

    return q_table


def continue_training(
    q_table_path: Path,
    episodes: int,
    card_decks_qty: int,
    dealer_hit_on_soft_17: bool,
    alpha: float,
    gamma: float,
    epsilon: float,
    save_path: Path | None = None,
    verbose: bool = False,
) -> QTable:
    """
    Continue training an existing Q-Table.

    Args:
        q_table_path: Path to existing Q-Table pickle file
        episodes: Additional episodes to train
        card_decks_qty: Number of card decks to use
        dealer_hit_on_soft_17: Whether dealer hits on soft 17
        alpha: Learning rate (typically lower for fine-tuning)
        gamma: Discount factor
        epsilon: Exploration rate
        save_path: Path to save the updated Q-Table (defaults to q_table_path)
        verbose: Enable verbose logging

    Returns:
        Updated QTable instance
    """
    setup_logging(verbose)
    log = logging.getLogger("tbjh.train")

    if save_path is None:
        save_path = q_table_path

    log.info("=" * 60)
    log.info("Continuing Q-Table Training")
    log.info(f"Loading from: {q_table_path}")
    log.info("=" * 60)
    log.info(f"Episodes: {episodes}")
    log.info(f"Card Decks: {card_decks_qty}")
    log.info(f"Dealer Hits Soft 17: {dealer_hit_on_soft_17}")
    log.info(f"Alpha (learning rate): {alpha}")
    log.info(f"Gamma (discount factor): {gamma}")
    log.info(f"Epsilon (exploration rate): {epsilon}")
    log.info(f"Save path: {save_path}")
    log.info("=" * 60)

    # Create game environment
    game = SimpleGame(
        card_decks_qty=card_decks_qty,
        dealer_hit_on_soft_17=dealer_hit_on_soft_17,
    )

    # Define rewards for each action result
    # Blackjack pays 3:2, wins pay 1:1, pushes return the bet, losses lose the bet
    rewards = {
        GameActionResult.WAIT_ACTION: QLearnerRewardAfterAction(0.0),
        GameActionResult.BLACKJACK: QLearnerRewardAfterAction(1.5),
        GameActionResult.WINS: QLearnerRewardAfterAction(1.0),
        GameActionResult.PUSH: QLearnerRewardAfterAction(0.0),
        GameActionResult.LOSS: QLearnerRewardAfterAction(-1.0),
        GameActionResult.BUST: QLearnerRewardAfterAction(-1.0),
    }

    # Load existing Q-Table
    q_table = QTable.load(str(q_table_path))
    log.info(f"Loaded Q-Table with {len(q_table)} states")

    # Create Q-Learner with the existing Q-Table
    learner = EpsilonGreedyQLearner(
        game_environment=game,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        agent_rewards=rewards,
        q_table=q_table,
    )

    # Training loop
    log.info("Starting training...")
    progress_interval = max(1, episodes // 20)  # Show progress every 5%

    for episode in range(1, episodes + 1):
        learner.run_train_iteration()

        if episode % progress_interval == 0 or episode == episodes:
            progress = (episode / episodes) * 100
            log.info(f"Progress: {progress:.1f}% ({episode}/{episodes}) - Q-Table size: {len(q_table)} states")

    log.info("Training completed!")
    log.info(f"Final Q-Table size: {len(q_table)} states")

    # Save the updated Q-Table
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        q_table.save(str(save_path))
        log.info(f"Updated Q-Table saved to {save_path}")
    except Exception as ex:
        log.error(f"Failed to save Q-Table: {ex}", exc_info=ex)

    return q_table


def evaluate_q_table(
    q_table_path: Path,
    num_rounds: int,
    card_decks_qty: int,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Evaluate the performance of a trained Q-Table.

    Args:
        q_table_path: Path to Q-Table pickle file
        num_rounds: Number of rounds to evaluate
        card_decks_qty: Number of card decks to use
        verbose: Enable verbose logging

    Returns:
        Dictionary with evaluation results
    """
    setup_logging(verbose)
    log = logging.getLogger("tbjh.evaluate")

    log.info("=" * 60)
    log.info("Evaluating Q-Table Performance")
    log.info(f"Loading from: {q_table_path}")
    log.info("=" * 60)
    log.info(f"Number of rounds: {num_rounds}")
    log.info(f"Card Decks: {card_decks_qty}")
    log.info("=" * 60)

    # Load Q-Table
    q_table = QTable.load(str(q_table_path))
    log.info(f"Loaded Q-Table with {len(q_table)} states")

    # Create game
    game = SimpleGame(card_decks_qty=card_decks_qty)

    # Create agent with the Q-Table
    agent = AgentForSimpleGameByQTable(q_table)

    # Statistics
    results = {
        "wins": 0.0,
        "losses": 0.0,
        "pushes": 0.0,
        "blackjacks": 0.0,
        "busts": 0.0,
        "total_rounds": 0.0,
    }

    log.info(f"Evaluating over {num_rounds} rounds...")
    progress_interval = max(1, num_rounds // 10)

    for round_num in range(1, num_rounds + 1):
        if round_num % progress_interval == 0 or round_num == num_rounds:
            log.info(f"Progress: {(round_num / num_rounds) * 100:.1f}%")

        # Start a new round
        game.reset()
        game.start_new_round()

        # Play until round ends
        while game.is_round_playing:
            state = game.state
            action = agent.decide(state)
            result = game.play(action)

            # Track final results
            if not game.is_round_playing:
                if result == GameActionResult.WINS:
                    results["wins"] += 1
                elif result == GameActionResult.LOSS:
                    results["losses"] += 1
                elif result == GameActionResult.PUSH:
                    results["pushes"] += 1
                elif result == GameActionResult.BLACKJACK:
                    results["blackjacks"] += 1
                elif result == GameActionResult.BUST:
                    results["busts"] += 1

        results["total_rounds"] += 1

    # Calculate statistics
    total = results["total_rounds"]
    win_rate = results["wins"] / total * 100
    blackjack_rate = results["blackjacks"] / total * 100
    bust_rate = results["busts"] / total * 100
    push_rate = results["pushes"] / total * 100
    loss_rate = results["losses"] / total * 100

    log.info("=" * 60)
    log.info("Evaluation Results:")
    log.info(f"  Total Rounds: {total}")
    log.info(f"  Wins: {results['wins']} ({win_rate:.1f}%)")
    log.info(f"  Losses: {results['losses']} ({loss_rate:.1f}%)")
    log.info(f"  Pushes: {results['pushes']} ({push_rate:.1f}%)")
    log.info(f"  Blackjacks: {results['blackjacks']} ({blackjack_rate:.1f}%)")
    log.info(f"  Busts: {results['busts']} ({bust_rate:.1f}%)")

    # Calculate expected value
    expected_value = (
                             results["wins"] * 1.0 +
                             results["blackjacks"] * 1.5 +
                             results["pushes"] * 0.0 +
                             results["losses"] * -1.0 +
                             results["busts"] * -1.0
                     ) / total

    log.info(f"  Expected Value per $1 bet: ${expected_value:.3f}")
    log.info("=" * 60)

    return {**results, "expected_value": expected_value}


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Q-Table for Blackjack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new Q-Table")
    train_parser.add_argument(
        "-e", "--episodes", type=int, default=10_000,
        help="Number of training episodes (default: 10000)",
    )
    train_parser.add_argument(
        "-d", "--decks", type=int, default=6,
        help="Number of card decks (default: 6)",
    )
    train_parser.add_argument(
        "-a", "--alpha", type=float, default=0.1,
        help="Learning rate (default: 0.1)",
    )
    train_parser.add_argument(
        "-g", "--gamma", type=float, default=0.95,
        help="Discount factor (default: 0.95)",
    )
    train_parser.add_argument(
        "-eps", "--epsilon", type=float, default=0.1,
        help="Exploration rate (default: 0.1)",
    )
    train_parser.add_argument(
        "-s", "--save", type=Path, default=Path("q_table.pkl"),
        help="Path to save Q-Table (default: q_table.pkl)",
    )
    train_parser.add_argument(
        "--soft17", action="store_true",
        help="Dealer hits on soft 17",
    )
    train_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    # Continue training command
    continue_parser = subparsers.add_parser("continue", help="Continue training an existing Q-Table")
    continue_parser.add_argument(
        "q_table_path", type=Path,
        help="Path to existing Q-Table pickle file",
    )
    continue_parser.add_argument(
        "-e", "--episodes", type=int, default=5_000,
        help="Additional episodes to train (default: 5000)",
    )
    continue_parser.add_argument(
        "-d", "--decks", type=int, default=6,
        help="Number of card decks (default: 6)",
    )
    continue_parser.add_argument(
        "-a", "--alpha", type=float, default=0.05,
        help="Learning rate (default: 0.05)",
    )
    continue_parser.add_argument(
        "-g", "--gamma", type=float, default=0.95,
        help="Discount factor (default: 0.95)",
    )
    continue_parser.add_argument(
        "-eps", "--epsilon", type=float, default=0.05,
        help="Exploration rate (default: 0.05)",
    )
    continue_parser.add_argument(
        "-s", "--save", type=Path, default=None,
        help="Path to save updated Q-Table (defaults to input path)",
    )
    continue_parser.add_argument(
        "--soft17", action="store_true",
        help="Dealer hits on soft 17",
    )
    continue_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained Q-Table")
    eval_parser.add_argument(
        "q_table_path", type=Path,
        help="Path to Q-Table pickle file",
    )
    eval_parser.add_argument(
        "-r", "--rounds", type=int, default=1_000,
        help="Number of rounds for evaluation (default: 1000)",
    )
    eval_parser.add_argument(
        "-d", "--decks", type=int, default=6,
        help="Number of card decks (default: 6)",
    )
    eval_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_q_table(
            episodes=args.episodes,
            card_decks_qty=args.decks,
            dealer_hit_on_soft_17=args.soft17,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            save_path=args.save,
            verbose=args.verbose,
        )

    elif args.command == "continue":
        continue_training(
            q_table_path=args.q_table_path,
            episodes=args.episodes,
            card_decks_qty=args.decks,
            dealer_hit_on_soft_17=args.soft17,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            save_path=args.save,
            verbose=args.verbose,
        )

    elif args.command == "evaluate":
        evaluate_q_table(
            q_table_path=args.q_table_path,
            num_rounds=args.rounds,
            card_decks_qty=args.decks,
            verbose=args.verbose,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
