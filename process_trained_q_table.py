from environment import GameState
from environment.default_game import DefaultGameState
from learning_engine.q_learning import QTable
from learning_engine.q_learning.misc_tools import QTableParser, QTableNarrower


class QTableParser4DefaultGame(QTableParser):
    class _ShortDefaultGameState(GameState):
        player_cards_qty: int
        player_cards_sum: int
        player_has_soft_hand: int
        dealer_open_card: int

    def __init__(self, q_table: QTable):
        super().__init__(q_table)
        self.__NEW_TO_OLD: dict[QTableParser4DefaultGame._ShortDefaultGameState, list[DefaultGameState]] = {}

    def __compute_new_to_old_if_clean(self):
        if self.__NEW_TO_OLD:
            return
        for full_old_state, old_table in self._ORIGIN.items():
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


class QTableNarrower4DefaultGame(QTableNarrower):
    def _export_old_state_to_new(self, old_state: DefaultGameState) -> DefaultGameState:  # probability .3f -> .2f
        return DefaultGameState(
            player_cards_qty=old_state.player_cards_qty,
            player_cards_sum=old_state.player_cards_sum,
            player_has_soft_hand=old_state.player_has_soft_hand,
            player_busting_probability=round(old_state.player_busting_probability, 2),
            dealer_open_card=old_state.dealer_open_card,
            dealer_cards_sum_less_than_17_probability=round(old_state.dealer_cards_sum_less_than_17_probability, 2),
            dealer_busting_probability=round(old_state.dealer_busting_probability, 2),
        )


if __name__ == "__main__":
    origin_q_table = QTable.load("old_q_table.tbjh")
    parser = QTableParser4DefaultGame(origin_q_table)
    narrower = QTableNarrower4DefaultGame(origin_q_table)
    narrowed_q_table = narrower.weight_average_by_distance(parser.get_states_with_distance, ignore_neutral=True)
    narrowed_q_table.save("new_q_table.tbjh")
    print(f"Successfully narrow {len(origin_q_table) - len(narrowed_q_table)} states")
