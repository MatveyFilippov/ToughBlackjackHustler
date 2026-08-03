import logging
from ..base import Agent
from ...environment import GameAction, GameState
from ...environment.games.simple import SimpleGameState
from ...q_learning import QTable
from ...q_learning.utils import QTableStatesParser


log = logging.getLogger("tbjh.agent.forsimplegame.byqtable")


class QTableStatesParserForSimpleGame(QTableStatesParser[SimpleGameState]):
    class _ShortSimpleGameState(GameState):
        player_cards_qty: int
        player_cards_sum: int
        player_has_soft_hand: int
        dealer_open_card: int

    def __init__(self, q_table: QTable[SimpleGameState]):
        super().__init__(q_table=q_table)
        self.__short_states_cache: dict[QTableStatesParserForSimpleGame._ShortSimpleGameState, list[SimpleGameState]] = {}

    def __get_short_states(self) -> dict['QTableStatesParserForSimpleGame._ShortSimpleGameState', list[SimpleGameState]]:
        if not self.__short_states_cache:
            for full_state in self._ORIGIN_STATES:
                self.__short_states_cache.setdefault(
                    self._ShortSimpleGameState(
                        player_cards_qty=full_state.player_cards_qty,
                        player_cards_sum=full_state.player_cards_sum,
                        player_has_soft_hand=full_state.player_has_soft_hand,
                        dealer_open_card=full_state.dealer_open_card,
                    ), list(),
                ).append(full_state)
        return self.__short_states_cache

    def _calculate_distances(self, target_state: SimpleGameState) -> dict[SimpleGameState, float]:
        short_states = self.__get_short_states()

        target_short_state = self._ShortSimpleGameState(
            player_cards_qty=target_state.player_cards_qty,
            player_cards_sum=target_state.player_cards_sum,
            player_has_soft_hand=target_state.player_has_soft_hand,
            dealer_open_card=target_state.dealer_open_card,
        )

        return {
            state: abs(
                state.player_busting_probability - target_state.player_busting_probability,
            ) + abs(
                state.dealer_cards_sum_less_than_17_probability - target_state.dealer_cards_sum_less_than_17_probability,
            ) + abs(
                state.dealer_busting_probability - target_state.dealer_busting_probability,
            )
            for state in short_states[target_short_state]
        }


class AgentForSimpleGameByQTable(Agent[SimpleGameState]):
    def __init__(self, q_table: QTable[SimpleGameState]):
        self.__Q_TABLE = q_table
        self.__PARSER = QTableStatesParserForSimpleGame(q_table)

    def decide(self, state: SimpleGameState) -> GameAction:
        if state not in self.__Q_TABLE:
            closest_state = self.__PARSER.find_closest_state(state)
            log.warning(
                "No such GameState in QTable(uuid=%s), will use closest (%s -> %s)",
                self.__Q_TABLE.uuid, state, closest_state,
            )
            state = closest_state
        return self.__Q_TABLE.get_best_action(state)
