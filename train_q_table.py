from environment.default_game import DefaultGame
from learning_engine.q_learning import QTable
from learning_engine.q_learning.strategies import EpsilonGreedyQLearner
import logging
import signal
import sys
import threading


logging.basicConfig(
    level=logging.INFO, filename=f"TrainQTable.log", encoding="UTF-8", datefmt="%Y-%m-%d %H:%M:%S",
    format="\n'%(name)s':\n%(levelname)s %(asctime)s --> %(message)s"
)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Unexpected error", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

Q_TABLE_FILEPATH = "q_table.tbjh"
LEARNER = EpsilonGreedyQLearner(
    game_environment=DefaultGame(card_decks_qty=4),
    alpha=0.45,
    gamma=0.9,
    epsilon=0.1,
    q_table=QTable.load(Q_TABLE_FILEPATH),
)


def save_by_signal(signum, frame):
    global LEARNER
    LEARNER.Q_TABLE.save(Q_TABLE_FILEPATH)


def save_and_exit_by_signal(signum, frame):
    save_by_signal(signum, frame)
    sys.exit(0)


def save_by_timer():
    global LEARNER
    LEARNER.Q_TABLE.save(Q_TABLE_FILEPATH)
    threading.Timer(1800, save_by_timer).start()


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, save_by_signal)  # kill -USR1 <PID>  (ps aux | grep python)
    signal.signal(signal.SIGUSR1, save_and_exit_by_signal)  # systemctl stop
    signal.signal(signal.SIGUSR1, save_and_exit_by_signal)  # Ctrl+C
    save_by_timer()
    try:
        train_iterations = 100
        while True:
            try:
                LEARNER.train(train_iterations)
                LEARNER.Q_TABLE.save(Q_TABLE_FILEPATH)
                logging.info(f"Successfully train {train_iterations} iterations")
            except Exception as ex:
                logging.error("Unexpected error", exc_info=True)
    finally:
        LEARNER.Q_TABLE.save(Q_TABLE_FILEPATH)
