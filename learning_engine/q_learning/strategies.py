from .base import QLearner, QTable
from environment import GameEnvironment, GameState, AgentAction
import random


class EpsilonGreedyQLearner(QLearner):
    def __init__(self, game_environment: GameEnvironment, alpha: float, gamma: float, epsilon: float, q_table: QTable | None = None):
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in diapason [0-1]")
        super().__init__(game_environment, alpha, gamma, q_table)
        self._EPSILON = epsilon

    def _choose_action(self, state: GameState) -> AgentAction:
        if state not in self.Q_TABLE or random.random() < self._EPSILON:
            return AgentAction.get_by_random(*self._GAME_ENVIRONMENT.available_actions)
        else:
            return self.Q_TABLE.get_best_action(state)
