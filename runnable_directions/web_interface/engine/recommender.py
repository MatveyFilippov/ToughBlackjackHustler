from agent.for_default_game import AgentForDefaultGameByQTable
from environment import GameAction
from environment.default_game import DefaultGameState
from learning_engine.q_learning import QTable
import os


Q_TABLE = QTable.load(os.path.join("..", "q_table.tbjh"))
AGENT = AgentForDefaultGameByQTable(Q_TABLE)


def get_recommendation(state: DefaultGameState) -> GameAction:
    return AGENT.decide(state)
