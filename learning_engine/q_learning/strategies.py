from .base import QLearner, QTable, AgentReward
from environment import GameEnvironment, GameState, GameAction, GameActionResult
import random


class EpsilonGreedyQLearner(QLearner):
    def __init__(self, game_environment: GameEnvironment, alpha: float, gamma: float, epsilon: float,
                 rewards: dict[GameActionResult, AgentReward], q_table: QTable | None = None):
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in diapason [0-1]")
        self._EPSILON = epsilon
        self._REWARDS = {}
        for action_result in GameActionResult:
            try:
                self._REWARDS[action_result] = rewards[action_result]
            except KeyError:
                raise ValueError(f"You forget to set AgentReward for {action_result}")
        super().__init__(game_environment, alpha, gamma, q_table)

    def _choose_action(self, state: GameState) -> GameAction:
        if state not in self.Q_TABLE or random.random() < self._EPSILON:
            return GameAction.get_by_random(*self._GAME_ENVIRONMENT.available_actions)
        else:
            return self.Q_TABLE.get_best_action(state)

    def _get_reward_for_action_result(self, action_result: GameActionResult) -> AgentReward:
        return self._REWARDS[action_result]
