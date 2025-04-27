import logging
import sys
from agent import QTable
from agent.strategies import EpsilonGreedyQLearner
from environment.default_game import DefaultGame


logging.basicConfig(
    level=logging.INFO, filename=f"LearningEngine.log", encoding="UTF-8", datefmt="%Y-%m-%d %H:%M:%S",
    format="\n'%(name)s':\n%(levelname)s %(asctime)s --> %(message)s"
)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Unexpected error", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


q_table_filepath = "q_table.tbjh"
q_table = QTable.load(q_table_filepath)
learner = EpsilonGreedyQLearner(
    game_environment=DefaultGame(card_decks_qty=4),
    alpha=0.7,
    gamma=0.9,
    epsilon=0.5,
    q_table=q_table,
)
save_q_table_after_n_iterations = 10_000
if __name__ == "__main__":
    while True:
        try:
            learner.train(save_q_table_after_n_iterations)
            learner.Q_TABLE.save(q_table_filepath)
            logging.info(f"Successfully train {save_q_table_after_n_iterations} iterations")
        except Exception as ex:
            logging.error("Unexpected error", exc_info=True)
