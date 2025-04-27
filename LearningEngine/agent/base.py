import logging
import pickle
import numpy as np
from collections import defaultdict
from environment import GameEnvironment, GameState, UserAction, AgentReward
from abc import ABC, abstractmethod


class QTable:
    __DEFAULT_INIT = lambda key: np.full(len(UserAction), AgentReward.NEUTRAL)

    def __init__(self, _from: dict[GameState, dict[UserAction, AgentReward]] | None = None):
        self.__table = defaultdict(self.__DEFAULT_INIT, {
            state: {act.value: float(rew) for act, rew in action_reward_dict.items()}
            for state, action_reward_dict in _from.items()
        }) if _from else defaultdict(self.__DEFAULT_INIT)

    def set_q_value(self, state: GameState, action: UserAction, value: AgentReward):
        self.__table[state][action.value] = float(value)

    def get_q_value(self, state: GameState, action: UserAction) -> AgentReward:
        return AgentReward(self.__table[state][action.value])

    def get_best_action(self, state: GameState) -> UserAction:
        return UserAction(np.argmax(self.__table[state]))

    def get_max_q_value(self, state: GameState) -> AgentReward:
        return AgentReward(np.max(self.__table[state]))

    def __contains__(self, state: GameState) -> bool:
        return state in self.__table

    def to_dict(self) -> dict[GameState, dict[UserAction, AgentReward]]:
        return {
            state: {
                UserAction(action_idx): AgentReward(float(reward))
                for action_idx, reward in enumerate(action_reward_list)
            }
            for state, action_reward_list in dict(self.__table).items()
        }

    def copy(self) -> 'QTable':
        return QTable(self.to_dict())

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filename: str) -> 'QTable':
        try:
            with open(filename, 'rb') as f:
                return QTable(pickle.load(f))
        except (pickle.PickleError, EOFError, FileNotFoundError):
            logging.warning(f"Error loading QTable from {filename}, creating new QTable", exc_info=True)
            return QTable()


class QLearner(ABC):
    def __init__(self, game_environment: GameEnvironment, alpha: float, gamma: float, q_table: QTable | None = None):
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be in diapason [0-1]")
        self._ALPHA = alpha
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be in diapason [0-1]")
        self._GAMMA = gamma

        self.Q_TABLE = q_table if q_table else QTable()
        self._GAME_ENVIRONMENT = game_environment

    @abstractmethod
    def _choose_action(self, state: GameState) -> UserAction:
        pass

    def _update_q_table(self, state: GameState, action: UserAction, reward: AgentReward, next_state: GameState):
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
