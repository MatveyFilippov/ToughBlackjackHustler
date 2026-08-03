from abc import ABC, abstractmethod
import logging
from ..q_table import QTable, QValue
from ...environment import GameAction, GameActionResult, GameEnvironment, GameStateType


log = logging.getLogger("tbjh.q_learning.strategies")


class QLearnerRewardAfterAction(float):
    # def __new__(cls, value):
    #     return float(value)

    def __repr__(self):
        return f"QLearnerRewardAfterAction({super().__repr__()})"

    def __str__(self):
        return super().__repr__()


class QLearner(ABC):
    def __init__(self, game_environment: GameEnvironment[GameStateType], alpha: float, gamma: float, q_table: QTable[GameStateType] | None = None):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be in diapason [0-1]")
        self._ALPHA = alpha
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be in diapason [0-1]")
        self._GAMMA = gamma

        self._Q_TABLE: QTable[GameStateType] = q_table if q_table else QTable(*game_environment.available_actions)
        self._GAME_ENVIRONMENT = game_environment

    @property
    def q_table(self) -> QTable:
        return self._Q_TABLE

    @property
    def game_environment(self) -> GameEnvironment:
        return self._GAME_ENVIRONMENT

    @abstractmethod
    def _choose_action(self, state: GameStateType) -> GameAction:
        ...

    @abstractmethod
    def _get_reward_for_action_result(self, action_result: GameActionResult) -> QLearnerRewardAfterAction:
        ...

    def _update_q_table(self, state: GameStateType, action: GameAction, reward: QLearnerRewardAfterAction, next_state: GameStateType | None):
        current_q = self._Q_TABLE.get_q_value(state, action)
        max_next_q = self._Q_TABLE.get_max_q_value(next_state) if next_state else QValue.NEUTRAL
        new_q = QValue(current_q + self._ALPHA * (reward + self._GAMMA * max_next_q - current_q))
        log.debug(
            "Compute NewQ[%s] = CurrentQ[%s] + Alpha[%s] * (Reward[%s] + Gamma[%s] * MaxNextQ[%s] - CurrentQ[%s])",
            new_q, current_q, self._ALPHA, reward, self._GAMMA, max_next_q, current_q,
        )
        self._Q_TABLE.set_q_value(state, action, new_q)

    def run_train_iteration(self):
        log.debug("Start train iteration for QTable(uuid=%s) by %s", self._Q_TABLE.uuid, self._GAME_ENVIRONMENT)
        self._GAME_ENVIRONMENT.reset()
        self._GAME_ENVIRONMENT.start_new_round()
        state = self._GAME_ENVIRONMENT.state
        log.debug("GameState: %s", state)
        while True:
            action = self._choose_action(state)
            log.debug("GameAction: %s", action.name)
            action_result = self._GAME_ENVIRONMENT.play(action)
            log.debug("GameActionResult: %s", action_result.name)
            reward = self._get_reward_for_action_result(action_result)
            log.debug("QLearnerRewardAfterAction: %s", reward)
            if not self._GAME_ENVIRONMENT.is_round_playing:
                if self._GAME_ENVIRONMENT.is_terminated:
                    self._update_q_table(state=state, action=action, reward=reward, next_state=None)
                    log.debug("End train iteration for QTable(uuid=%s) because %s is terminated", self._Q_TABLE.uuid, self._GAME_ENVIRONMENT)
                    return
                self._GAME_ENVIRONMENT.start_new_round()
            next_state = self._GAME_ENVIRONMENT.state
            self._update_q_table(state=state, action=action, reward=reward, next_state=next_state)
            state = next_state
            log.debug("GameState: %s", state)

    @abstractmethod
    def __repr__(self):
        ...

    @abstractmethod
    def __str__(self):
        ...
