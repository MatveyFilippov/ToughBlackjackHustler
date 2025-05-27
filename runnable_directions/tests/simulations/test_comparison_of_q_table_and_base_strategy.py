from agent.for_default_game import AgentForDefaultGameByBasicStrategy, AgentForDefaultGameByQTable
from environment.default_game import DefaultGame
from learning_engine.q_learning import QTable
from runnable_directions.tests.simulations.game_simulator import GameSimulator
import time
import os


if __name__ == "__main__":
    sim_by_basic_strategy = GameSimulator(
        game_environment=DefaultGame(4), agent=AgentForDefaultGameByBasicStrategy(),
    )
    sim_by_q_table = GameSimulator(
        game_environment=DefaultGame(4),
        agent=AgentForDefaultGameByQTable(QTable.load(os.path.join("..", "..", "q_table.tbjh"))),
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
