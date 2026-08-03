from abc import ABC, abstractmethod
from statistics import median, mode
from typing import Callable, Generic, TypeVar
from ..q_table import QTable, QValue
from ...environment import GameAction, GameState


SourceGameState = TypeVar('SourceGameState', bound=GameState)
TargetGameState = TypeVar('TargetGameState', bound=GameState)


class QTableStatesParser(Generic[SourceGameState], ABC):
    def __init__(self, q_table: QTable[SourceGameState]):
        self._ORIGIN_STATES = list(q_table.to_dict().keys())
        self.__states_with_distance_cache: dict[SourceGameState, dict[SourceGameState, float]] = {}
        self.__sorted_states_by_distance_cache: dict[SourceGameState, list[SourceGameState]] = {}

    @abstractmethod
    def _calculate_distances(self, target_state: SourceGameState) -> dict[SourceGameState, float]:
        ...

    def get_states_with_distance(self, target_state: SourceGameState) -> dict[SourceGameState, float]:
        if target_state not in self.__states_with_distance_cache:
            self.__states_with_distance_cache[target_state] = self._calculate_distances(target_state)
        return self.__states_with_distance_cache[target_state]

    def __get_sorted_states_by_distance(self, target_state: SourceGameState) -> list[SourceGameState]:
        if target_state not in self.__sorted_states_by_distance_cache:
            states_with_distance = self.get_states_with_distance(target_state)
            self.__sorted_states_by_distance_cache[target_state] = sorted(
                states_with_distance.keys(), key=states_with_distance.get,
            )
        return self.__sorted_states_by_distance_cache[target_state]

    def find_close_states(self, target_state: SourceGameState, n: int) -> list[SourceGameState]:
        if n <= 0:
            raise ValueError("Argument 'n' must be positive")
        states = self.__get_sorted_states_by_distance(target_state=target_state)
        return states if n >= len(states) else states[:n]

    def find_closest_state(self, target_state: SourceGameState) -> SourceGameState:
        states = self.__get_sorted_states_by_distance(target_state=target_state)
        return states[0]


class QTableNarrower(Generic[SourceGameState, TargetGameState], ABC):
    def __init__(self, q_table: QTable[SourceGameState]):
        self._ORIGIN = q_table.to_dict()
        self._AVAILABLE_ACTIONS = q_table.available_actions

        self.__narrowed_states_cache: dict[TargetGameState, dict[GameAction, list[QValue]]] = {}

    @abstractmethod
    def _narrow_down_state(self, origin: SourceGameState) -> TargetGameState:
        ...

    def __get_narrowed_states(self) -> dict[TargetGameState, dict[GameAction, list[QValue]]]:
        if not self.__narrowed_states_cache:
            for origin_state, origin_action_value in self._ORIGIN.items():
                narrowed_state = self._narrow_down_state(origin_state)
                if narrowed_state not in self.__narrowed_states_cache:
                    self.__narrowed_states_cache[narrowed_state] = {act: [] for act in self._AVAILABLE_ACTIONS}
                for action, values in origin_action_value.items():
                    self.__narrowed_states_cache[narrowed_state][action].append(values)
        return self.__narrowed_states_cache

    def narrow_down(self, mapper: Callable[[TargetGameState, GameAction, list[QValue]], QValue], ignore_neutral: bool = True) -> QTable[TargetGameState]:
        narrowed_q_table: dict[TargetGameState, dict[GameAction, QValue]] = {}
        for narrowed_state, origin_action_values in self.__get_narrowed_states().items():
            new_row: dict[GameAction, QValue] = {}
            for action, values in origin_action_values.items():
                while ignore_neutral and QValue.NEUTRAL in values:
                    values.remove(QValue.NEUTRAL)
                new_row[action] = mapper(narrowed_state, action, values) if len(values) > 0 else QValue.NEUTRAL
            narrowed_q_table[narrowed_state] = new_row
        return QTable(*self._AVAILABLE_ACTIONS, _from=narrowed_q_table)

    def narrow_down_by_average(self, ignore_neutral: bool = True) -> QTable[TargetGameState]:
        return self.narrow_down(lambda s, a, values: QValue(sum(values) / len(values)), ignore_neutral)

    def narrow_down_by_max(self, ignore_neutral: bool = True) -> QTable[TargetGameState]:
        return self.narrow_down(lambda s, a, values: QValue(max(values)), ignore_neutral)

    def narrow_down_by_moda(self, ignore_neutral: bool = True) -> QTable[TargetGameState]:
        return self.narrow_down(lambda s, a, values: QValue(mode(values)), ignore_neutral)

    def narrow_down_by_median(self, ignore_neutral: bool = True) -> QTable[TargetGameState]:
        return self.narrow_down(lambda s, a, values: QValue(median(values)), ignore_neutral)

    def narrow_down_by_distance_weight_average(self, distance_calculator: Callable[[TargetGameState], dict[SourceGameState, float]], ignore_neutral: bool = True) -> QTable[TargetGameState]:
        def mapper(narrowed_state: TargetGameState, action: GameAction, values: list[QValue]) -> QValue:
            weights = {
                state: 1.0 / (distance + 1e-5)
                for state, distance in distance_calculator(narrowed_state).items()
                if distance < 0.01
            }
            total_weight = sum(weights.values())

            weighted_sum = 0.0
            total_action_weight = 0.0

            for state, weight in weights.items():
                origin_value = self._ORIGIN[state][action]
                if ignore_neutral and origin_value == QValue.NEUTRAL:
                    continue
                weight /= total_weight
                weighted_sum += origin_value * weight
                total_action_weight += weight

            return QValue(weighted_sum / total_action_weight) if total_action_weight > 0 else QValue.NEUTRAL

        return self.narrow_down(mapper, ignore_neutral)
