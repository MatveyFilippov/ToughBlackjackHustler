from agent.for_default_game import QTableStatesParser4DefaultGame
from environment.default_game import DefaultGameState
from learning_engine.q_learning import QTable
from learning_engine.q_learning.misc_tools import QTableNarrower


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
    parser = QTableStatesParser4DefaultGame(origin_q_table)
    narrower = QTableNarrower4DefaultGame(origin_q_table)
    narrowed_q_table = narrower.weight_average_by_distance(parser.get_states_with_distance, ignore_neutral=True)
    narrowed_q_table.save("new_q_table.tbjh")
    print(f"Successfully narrow {len(origin_q_table) - len(narrowed_q_table)} states")
