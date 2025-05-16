from agent.for_default_game import AgentForDefaultGameByQTable
from environment.default_game import DefaultGame
from learning_engine.q_learning import QTable
from tests.simulations.game_simulator import GameSimulator
import time
import os


if __name__ == "__main__":
    with GameSimulator(
            game_environment=DefaultGame(4),
            agent=AgentForDefaultGameByQTable(QTable.load(os.path.join("..", "..", "q_table.tbjh"))),
    ) as sim:
        while sim.is_running:
            sim_info = f"Score: {sim.score}"
            print(sim_info, end="")
            time.sleep(1.5)
            print("\b"*len(sim_info), end="")
