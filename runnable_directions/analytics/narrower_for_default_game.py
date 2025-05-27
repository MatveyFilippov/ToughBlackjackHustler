from agent.for_default_game import QTableStatesParser4DefaultGame
from environment import GameState
from environment.default_game import DefaultGameState
from learning_engine.q_learning import QTable
from learning_engine.q_learning.misc_tools import QTableNarrower
from enum import Enum
from typing import Callable


class ProbabilityCategory(float, Enum):
    VERY_LOW = 0.2  # [0-0.2)
    LOW = 0.4  # [0.2 - 0.4)
    MIDDLE = 0.6  # [0.4-0.6)
    HIGH = 0.8  # [0.6-0.8)
    VERY_HIGH = 1 + 1e-10  # [0.8-1]

    @classmethod
    def from_probability(cls, prob_num: float) -> 'ProbabilityCategory':
        for prob_cat in cls:
            if prob_num < prob_cat.value:
                return prob_cat


class DefaultGameStateWithNarrowedProbability(GameState):
    player_cards_qty: int
    player_cards_sum: int
    player_has_soft_hand: int
    player_busting_probability: ProbabilityCategory
    dealer_open_card: int
    dealer_cards_sum_less_than_17_probability: ProbabilityCategory
    dealer_busting_probability: ProbabilityCategory


class QTableNarrower4DefaultGame(QTableNarrower):
    def __init__(self, q_table: QTable):
        super().__init__(q_table)
        self.__PARSER = QTableStatesParser4DefaultGame(q_table)

    def _export_old_state_to_new(self, old_state: DefaultGameState) -> DefaultGameStateWithNarrowedProbability:
        return DefaultGameStateWithNarrowedProbability(
            player_cards_qty=old_state.player_cards_qty,
            player_cards_sum=old_state.player_cards_sum,
            player_has_soft_hand=old_state.player_has_soft_hand,
            player_busting_probability=ProbabilityCategory.from_probability(old_state.player_busting_probability),
            dealer_open_card=old_state.dealer_open_card,
            dealer_cards_sum_less_than_17_probability=ProbabilityCategory.from_probability(old_state.dealer_cards_sum_less_than_17_probability),
            dealer_busting_probability=ProbabilityCategory.from_probability(old_state.dealer_busting_probability),
        )

    def weight_average_by_distance(
            self,
            distance_calculator: Callable[[DefaultGameState], dict[DefaultGameState, float]] | None = None,
            ignore_neutral=True
    ) -> QTable:
        if distance_calculator is None:
            distance_calculator = self.__PARSER.get_states_with_distance
        return super().weight_average_by_distance(
            distance_calculator=distance_calculator, ignore_neutral=ignore_neutral
        )
