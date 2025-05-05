from ..base import QTable
from environment import GameState, AgentAction, AgentReward
from abc import ABC, abstractmethod
from statistics import mode, median
from typing import Callable
from progress.bar import Bar


class QTableStatesParser(ABC):
    def __init__(self, q_table: QTable):
        self._ORIGIN_STATES = list(q_table.to_dict().keys())
        self.__states_with_distance_cache: dict[GameState, dict[GameState, float]] = {}

    @abstractmethod
    def _calculate_distances(self, target_state: GameState) -> dict[GameState, float]:
        pass

    def get_states_with_distance(self, target_state: GameState) -> dict[GameState, float]:
        if target_state not in self.__states_with_distance_cache:
            self.__states_with_distance_cache[target_state] = self._calculate_distances(target_state)
        return self.__states_with_distance_cache[target_state]

    def find_close_states(self, target_state: GameState, n: int) -> list[GameState]:
        if n <= 0:
            raise ValueError("Argument 'n' must be positive")
        states_with_distance = self.get_states_with_distance(target_state)
        states = sorted(states_with_distance, key=states_with_distance.get)
        return states if n > len(states) else states[:n]

    def find_closest_state(self, target_state: GameState) -> GameState:
        states_with_distance = self.get_states_with_distance(target_state)
        return min(states_with_distance, key=states_with_distance.get)


class QTableNarrower:
    def __init__(self, q_table: QTable):
        self._ORIGIN = q_table.to_dict()
        self._AVAILABLE_ACTIONS = q_table.available_actions

        self.__STATES_TO_NARROW_DOWN: dict[GameState, dict[AgentAction, list[AgentReward]]] = {}

    @abstractmethod
    def _export_old_state_to_new(self, old_state: GameState) -> GameState:
        pass

    def __compute_states_to_narrow_down_if_clean(self):
        if self.__STATES_TO_NARROW_DOWN:
            return
        for old_state, old_table in self._ORIGIN.items():
            new_state = self._export_old_state_to_new(old_state)
            if new_state not in self.__STATES_TO_NARROW_DOWN:
                self.__STATES_TO_NARROW_DOWN[new_state] = {act: [] for act in self._AVAILABLE_ACTIONS}
            for act, rew in old_table.items():
                self.__STATES_TO_NARROW_DOWN[new_state][act].append(rew)

    def _narrow_down(self, mapper: Callable[[GameState, AgentAction, list[AgentReward]], AgentReward], ignore_neutral: bool) -> QTable:
        self.__compute_states_to_narrow_down_if_clean()
        progress_bar = Bar(
            'Narrow down %(max)d states', max=len(self.__STATES_TO_NARROW_DOWN),
            suffix='%(index)d/%(remaining)d %(percent).2f%% [%(avg)d - %(elapsed)d/%(eta)d]s'
        )
        narrowed_q_table = {}
        for narrowed_state, old_q_values in self.__STATES_TO_NARROW_DOWN.items():
            new_q_values = {}
            for action, rewards in old_q_values.items():
                while ignore_neutral and AgentReward.NEUTRAL in rewards:
                    rewards.remove(AgentReward.NEUTRAL)
                new_q_values[action] = mapper(narrowed_state, action, rewards) if len(rewards) > 0 else AgentReward.NEUTRAL
            narrowed_q_table[narrowed_state] = new_q_values
            progress_bar.next()
        progress_bar.finish()
        return QTable(*self._AVAILABLE_ACTIONS, _from=narrowed_q_table)

    def average(self, ignore_neutral=True) -> QTable:
        return self._narrow_down(lambda s, a, rewards: AgentReward(sum(rewards) / len(rewards)), ignore_neutral)

    def max(self, ignore_neutral=True) -> QTable:
        return self._narrow_down(lambda s, a, rewards: AgentReward(max(rewards)), ignore_neutral)

    def moda(self, ignore_neutral=True) -> QTable:
        return self._narrow_down(lambda s, a, rewards: AgentReward(mode(rewards)), ignore_neutral)

    def median(self, ignore_neutral=True) -> QTable:
        return self._narrow_down(lambda s, a, rewards: AgentReward(median(rewards)), ignore_neutral)

    def weight_average_by_distance(self, distance_calculator: Callable[[GameState], dict[GameState, float]], ignore_neutral=True) -> QTable:
        def mapper(narrowed_state: GameState, action: AgentAction, r) -> AgentReward:
            weights = {
                state: 1.0 / (distance + 1e-5)
                for state, distance in distance_calculator(narrowed_state).items()
                if distance < 0.01
            }
            total_weight = sum(weights.values())

            weighted_sum = 0.0
            total_action_weight = 0.0

            for state, weight in weights.items():
                origin_reward = self._ORIGIN[state][action]
                if ignore_neutral and origin_reward == AgentReward.NEUTRAL:
                    continue
                weight = weight / total_weight
                weighted_sum += origin_reward * weight
                total_action_weight += weight

            return AgentReward(weighted_sum / total_action_weight) if total_action_weight > 0 else AgentReward.NEUTRAL

        return self._narrow_down(mapper, ignore_neutral)
