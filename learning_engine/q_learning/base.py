from environment import GameEnvironment, GameState, AgentAction, AgentReward
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable
import numpy as np


class QTable:
    __DEFAULT_INIT: dict[tuple[AgentAction, ...], Callable[[GameState], np.ndarray]] = {}

    @classmethod
    def __get_default_init(cls, *available_actions: AgentAction):
        return cls.__DEFAULT_INIT.setdefault(available_actions, (
            lambda: np.full(len(available_actions), AgentReward.NEUTRAL)
        ))

    def __init__(self, *available_actions: AgentAction, _from: dict[GameState, dict[AgentAction, AgentReward]] | None = None):
        if len(available_actions) < 2:
            raise ValueError("Q-Table must provide 2 or more AgentAction")
        self.__available_actions = available_actions

        self.__table: dict[GameState, np.ndarray] = defaultdict(self.__get_default_init(*self.__available_actions))

        self.__action_to_index: dict[AgentAction, int] = {}
        self.__index_to_action: dict[int, AgentAction] = {}
        for index, action in enumerate(self.__available_actions):
            self.__action_to_index[action] = index
            self.__index_to_action[index] = action

        if _from:
            for state, action_reward in _from.items():
                for action, reward in action_reward.items():
                    self.set_q_value(state, action, reward)

    @property
    def available_actions(self) -> tuple[AgentAction, ...]:
        return self.__available_actions

    def set_q_value(self, state: GameState, action: AgentAction, value: AgentReward):
        self.__table[state][self.__action_to_index[action]] = float(value)

    def get_q_value(self, state: GameState, action: AgentAction) -> AgentReward:
        return AgentReward(self.__table[state][self.__action_to_index[action]])

    def get_best_action(self, state: GameState) -> AgentAction:
        return self.__index_to_action[np.argmax(self.__table[state])]

    def get_max_q_value(self, state: GameState) -> AgentReward:
        return AgentReward(np.max(self.__table[state]))

    def __contains__(self, state: GameState) -> bool:
        return state in self.__table

    def __len__(self) -> int:
        return len(self.__table)

    def to_dict(self) -> dict[GameState, dict[AgentAction, AgentReward]]:
        return {
            state: {
                action: AgentReward(rewards[index])
                for action, index in self.__action_to_index.items()
            }
            for state, rewards in self.__table.items()
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
            raise SystemError(f"Error loading Q-Table from {filename}")


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
    def _choose_action(self, state: GameState) -> AgentAction:
        pass

    def _update_q_table(self, state: GameState, action: AgentAction, reward: AgentReward, next_state: GameState):
        current_q = self.Q_TABLE.get_q_value(state, action)
        max_next_q = self.Q_TABLE.get_max_q_value(next_state)
        new_q = AgentReward(current_q + self._ALPHA * (reward + self._GAMMA * max_next_q - current_q))
        self.Q_TABLE.set_q_value(state, action, new_q)

    def train(self, episodes: int):
        for _ in range(episodes):
            self._GAME_ENVIRONMENT.reset()
            state = self._GAME_ENVIRONMENT.state
            while not self._GAME_ENVIRONMENT.is_terminated:
                action = self._choose_action(state)
                reward = self._GAME_ENVIRONMENT.play(action)
                next_state = self._GAME_ENVIRONMENT.state
                self._update_q_table(state=state, action=action, reward=reward, next_state=next_state)
                state = next_state
