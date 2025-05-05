from ..base import Agent
from environment import GameState, GameAction
from environment.default_game import DefaultGameState
from learning_engine.q_learning import QTable
from learning_engine.q_learning.misc_tools import QTableStatesParser


class QTableStatesParser4DefaultGame(QTableStatesParser):
    class _ShortDefaultGameState(GameState):
        player_cards_qty: int
        player_cards_sum: int
        player_has_soft_hand: int
        dealer_open_card: int

    def __init__(self, q_table: QTable):
        super().__init__(q_table)
        self.__NEW_TO_OLD: dict[QTableStatesParser4DefaultGame._ShortDefaultGameState, list[DefaultGameState]] = {}

    def __compute_new_to_old_if_clean(self):
        if self.__NEW_TO_OLD:
            return
        for full_old_state in self._ORIGIN_STATES:
            self.__NEW_TO_OLD.setdefault(self._ShortDefaultGameState(
                player_cards_qty=full_old_state.player_cards_qty,
                player_cards_sum=full_old_state.player_cards_sum,
                player_has_soft_hand=full_old_state.player_has_soft_hand,
                dealer_open_card=full_old_state.dealer_open_card,
            ), list()).append(full_old_state)

    def _calculate_distances(self, target_state: DefaultGameState) -> dict[DefaultGameState, float]:
        self.__compute_new_to_old_if_clean()

        target_short_state = self._ShortDefaultGameState(
            player_cards_qty=target_state.player_cards_qty,
            player_cards_sum=target_state.player_cards_sum,
            player_has_soft_hand=target_state.player_has_soft_hand,
            dealer_open_card=target_state.dealer_open_card,
        )

        return {
            state: abs(
                state.player_busting_probability - target_state.player_busting_probability
            ) + abs(
                state.dealer_cards_sum_less_than_17_probability - target_state.dealer_cards_sum_less_than_17_probability
            ) + abs(
                state.dealer_busting_probability - target_state.dealer_busting_probability
            )
            for state in self.__NEW_TO_OLD[target_short_state]
        }


class AgentForDefaultGameByQTable(Agent):
    def __init__(self, q_table: QTable):
        self.__Q_TABLE = q_table
        self.__PARSER = QTableStatesParser4DefaultGame(q_table)

    def decide(self, state: GameState) -> GameAction:
        if state not in self.__Q_TABLE:
            state = self.__PARSER.find_closest_state(state)
        return self.__Q_TABLE.get_best_action(state)
