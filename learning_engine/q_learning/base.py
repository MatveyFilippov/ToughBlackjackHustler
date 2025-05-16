from environment import GameEnvironment, GameState, GameAction, GameActionResult
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable
import numpy as np


class QValue(float):
    # def __new__(cls, value):
    #     return float(value)

    NEUTRAL = 0.0


class QTable:
    __DEFAULT_INIT: dict[tuple[GameAction, ...], Callable[[GameState], np.ndarray]] = {}

    @classmethod
    def __get_default_init(cls, *available_actions: GameAction):
        return cls.__DEFAULT_INIT.setdefault(available_actions, (
            lambda: np.full(len(available_actions), QValue.NEUTRAL)
        ))

    def __init__(self, *available_actions: GameAction, _from: dict[GameState, dict[GameAction, QValue]] | None = None):
        if len(available_actions) < 2:
            raise ValueError("Q-Table must provide 2 or more GameAction")
        self.__available_actions = available_actions

        self.__table: dict[GameState, np.ndarray] = defaultdict(self.__get_default_init(*self.__available_actions))

        self.__action_to_index: dict[GameAction, int] = {}
        self.__index_to_action: dict[int, GameAction] = {}
        for index, action in enumerate(self.__available_actions):
            self.__action_to_index[action] = index
            self.__index_to_action[index] = action

        if _from:
            for state, action_value in _from.items():
                for action, value in action_value.items():
                    self.set_q_value(state, action, value)

    @property
    def available_actions(self) -> tuple[GameAction, ...]:
        return self.__available_actions

    def set_q_value(self, state: GameState, action: GameAction, value: QValue):
        self.__table[state][self.__action_to_index[action]] = float(value)

    def get_q_value(self, state: GameState, action: GameAction) -> QValue:
        return QValue(self.__table[state][self.__action_to_index[action]])

    def get_best_action(self, state: GameState) -> GameAction:
        return self.__index_to_action[np.argmax(self.__table[state])]

    def get_max_q_value(self, state: GameState) -> QValue:
        return QValue(np.max(self.__table[state]))

    def __contains__(self, state: GameState) -> bool:
        return state in self.__table

    def __len__(self) -> int:
        return len(self.__table)

    def to_dict(self) -> dict[GameState, dict[GameAction, QValue]]:
        return {
            state: {
                action: QValue(values[index])
                for action, index in self.__action_to_index.items()
            }
            for state, values in self.__table.items()
        }

    def copy(self) -> 'QTable':
        return QTable(*self.__available_actions, _from=self.to_dict())

    def save(self, filename: str):
        dict_to_save = {
            "available_actions": self.__available_actions,
            "q_table": self.to_dict(),
        }
        with open(filename, 'wb') as f:
            pickle.dump(dict_to_save, f)

    @classmethod
    def load(cls, filename: str) -> 'QTable':
        try:
            with open(filename, 'rb') as f:
                saved_dict = pickle.load(f)
                return QTable(*saved_dict["available_actions"], _from=saved_dict["q_table"])
        except (pickle.PickleError, EOFError, FileNotFoundError):
            raise ValueError(f"Error loading Q-Table from {filename}")


class QLearnerRewardAfterAction(float):
    # def __new__(cls, value):
    #     return float(value)
    pass


class QLearner(ABC):
    def __init__(self, game_environment: GameEnvironment, alpha: float, gamma: float, q_table: QTable | None = None):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be in diapason [0-1]")
        self._ALPHA = alpha
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be in diapason [0-1]")
        self._GAMMA = gamma

        self.Q_TABLE = q_table if q_table else QTable(*game_environment.available_actions)
        self._GAME_ENVIRONMENT = game_environment

    @abstractmethod
    def _choose_action(self, state: GameState) -> GameAction:
        pass

    @abstractmethod
    def _get_reward_for_action_result(self, action_result: GameActionResult) -> QLearnerRewardAfterAction:
        pass

    def _update_q_table(self, state: GameState, action: GameAction, reward: QLearnerRewardAfterAction, next_state: GameState):
        current_q = self.Q_TABLE.get_q_value(state, action)
        max_next_q = self.Q_TABLE.get_max_q_value(next_state)
        new_q = QValue(current_q + self._ALPHA * (reward + self._GAMMA * max_next_q - current_q))
        self.Q_TABLE.set_q_value(state, action, new_q)

    def train(self, episodes: int):
        for _ in range(episodes):
            self._GAME_ENVIRONMENT.reset()
            state = self._GAME_ENVIRONMENT.state
            while not self._GAME_ENVIRONMENT.is_terminated:
                action = self._choose_action(state)
                action_result = self._GAME_ENVIRONMENT.play(action)
                reward = self._get_reward_for_action_result(action_result)
                next_state = self._GAME_ENVIRONMENT.state
                self._update_q_table(state=state, action=action, reward=reward, next_state=next_state)
                state = next_state
