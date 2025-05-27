from learning_engine.q_learning import QTable
import os
from progress.spinner import Spinner


Q_TABLES_HISTORY: dict[int, QTable] = {}

history_q_table_path = os.path.join("..", "q_table_backups", "{}.tbjh")

loading_bar = Spinner('Loading Q-Table history %(index)d, please, wait ')
loading_bar.start()
i = 44
while True:
    i += 1
    try:
        old_q_table = QTable.load(history_q_table_path.format(i))
    except ValueError:
        break
    loading_bar.next()
    Q_TABLES_HISTORY[i] = old_q_table
loading_bar.finish()

if len(Q_TABLES_HISTORY) == 0:
    raise RuntimeError("Problem with loading old Q-Tables: nothing was found")
