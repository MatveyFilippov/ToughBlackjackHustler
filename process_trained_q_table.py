from agent import Agent
from agent.for_default_game import (
    AgentForDefaultGameByBasicStrategy, AgentForDefaultGameByQTable, QTableStatesParser4DefaultGame,
)
from environment import GameEnvironment, GameActionResult
from environment.default_game import DefaultGameState, DefaultGame
from learning_engine.q_learning import QTable
from learning_engine.q_learning.misc_tools import QTableNarrower
import time
import threading


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


class GameSimulator:
    def __init__(self, game_environment: GameEnvironment, agent: Agent):
        self.__GAME_ENVIRONMENT = game_environment
        self.__AGENT = agent

        self.__score = 0.0
        self.__thread: threading.Thread = None

    def __count_up(self, result: GameActionResult):
        if result in [GameActionResult.WINS, GameActionResult.BLACKJACK]:
            self.__score += 1
        elif result in [GameActionResult.LOSS, GameActionResult.BUST]:
            self.__score -= 1

    def __run(self, iterations: int):
        subtrahend = 0 if iterations == -1 else 1
        if iterations < 0:
            iterations = 0
        while True:
            iterations -= subtrahend
            self.__GAME_ENVIRONMENT.reset()
            while not self.__GAME_ENVIRONMENT.is_terminated:
                state = self.__GAME_ENVIRONMENT.state
                action = self.__AGENT.decide(state)
                result = self.__GAME_ENVIRONMENT.play(action)
                self.__count_up(result)
            if iterations < 0:
                break

    def start(self, iterations: int = -1):
        self.__thread = threading.Thread(target=self.__run, args=(iterations,), name="Simulate game", daemon=True)
        self.__thread.start()

    def stop(self):
        if self.__thread and self.__thread.is_alive():
            self.__thread.join(5)
            self.__thread = None

    @property
    def is_running(self) -> bool:
        return self.__thread.is_alive()

    def reset(self):
        self.stop()
        self.__score = 0.0

    @property
    def score(self) -> float:
        return self.__score

    def __enter__(self) -> 'GameSimulator':
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset()

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    # origin_q_table = QTable.load("old_q_table.tbjh")
    # parser = QTableStatesParser4DefaultGame(origin_q_table)
    # narrower = QTableNarrower4DefaultGame(origin_q_table)
    # narrowed_q_table = narrower.weight_average_by_distance(parser.get_states_with_distance, ignore_neutral=True)
    # narrowed_q_table.save("new_q_table.tbjh")
    # print(f"Successfully narrow {len(origin_q_table) - len(narrowed_q_table)} states")

    sim_by_basic_strategy = GameSimulator(
        game_environment=DefaultGame(4), agent=AgentForDefaultGameByBasicStrategy()
    )
    sim_by_q_table = GameSimulator(
        game_environment=DefaultGame(4), agent=AgentForDefaultGameByQTable(QTable.load("q_table.tbjh"))
    )
    sim_by_basic_strategy.start()
    sim_by_q_table.start()
    while sim_by_basic_strategy.is_running and sim_by_q_table.is_running:
        sim_info = f"Q-Table: {sim_by_q_table.score}\tBasicStrategy: {sim_by_basic_strategy.score}"
        print(sim_info, end="")
        time.sleep(1.5)
        print("\b"*len(sim_info), end="")
    sim_by_basic_strategy.stop()
    sim_by_q_table.stop()
