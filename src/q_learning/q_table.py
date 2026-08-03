import logging
import pickle
from typing import Generic
import uuid
from bidict import bidict
import numpy as np
from ..environment import GameAction, GameStateType


log = logging.getLogger("tbjh.q_learning.q_table")


class QValue(float):
    # def __new__(cls, value):
    #     return float(value)

    NEUTRAL: 'QValue' = 0.0

    def __repr__(self):
        return f"QValue({super().__repr__()})"

    def __str__(self):
        return super().__repr__()


class QTable(Generic[GameStateType]):
    def __init__(self, *available_actions: GameAction, _from: dict[GameStateType, dict[GameAction, QValue]] | None = None, _uuid: str | None = None):
        self.__UUID = _uuid or str(uuid.uuid4())

        if len(available_actions) < 2:
            raise ValueError("QTable must provide 2 or more GameActions")
        self.__action_map = bidict({action: i for i, action in enumerate(available_actions)})

        self.__NEUTRAL_ROW: np.ndarray[tuple[int], np.dtype[np.float64]] = np.full(
            len(available_actions), QValue.NEUTRAL, dtype=np.float64,
        )
        self.__table: dict[GameStateType, np.ndarray[tuple[int], np.dtype[np.float64]]] = dict()

        if _from:
            for state, action_value in _from.items():
                for action, value in action_value.items():
                    self.set_q_value(state, action, value)

        available_actions_str = ", ".join(a.name for a in available_actions)
        log.info("Initialize QTable(uuid=%s) with available GameActions: %s", self.__UUID, available_actions_str)

    @property
    def uuid(self) -> str:
        return self.__UUID

    @property
    def available_actions(self) -> tuple[GameAction, ...]:
        # return tuple(self.__action_map.keys())  # bidict does not guarantee the order of saving
        result = []
        for i in sorted(self.__action_map.values()):
            result.append(self.__action_map.inverse[i])
        return tuple(result)

    def set_q_value(self, state: GameStateType, action: GameAction, value: QValue):
        if state not in self.__table:
            self.__table[state] = self.__NEUTRAL_ROW.copy()

        self.__table[state][self.__action_map[action]] = float(value)
        log.debug("Set QValue[%s] for %s on %s in QTable(uuid=%s)", value, action.name, state, self.__UUID)

    def get_q_value(self, state: GameStateType, action: GameAction) -> QValue:
        if state not in self.__table:
            return QValue.NEUTRAL

        return QValue(self.__table[state][self.__action_map[action]])

    def get_max_q_value(self, state: GameStateType) -> QValue:
        if state not in self.__table:
            return QValue.NEUTRAL

        return QValue(np.max(self.__table[state]))

    def get_best_action(self, state: GameStateType) -> GameAction:
        if state not in self.__table:
            raise KeyError(f"No such GameState ({state}) in QTable(uuid={self.__UUID})")

        return self.__action_map.inverse[np.argmax(self.__table[state])]

    def __contains__(self, state: GameStateType) -> bool:
        return state in self.__table

    def __len__(self) -> int:
        return len(self.__table)

    def to_dict(self) -> dict[GameStateType, dict[GameAction, QValue]]:
        return {
            state: {
                action: QValue(values[index])
                for action, index in self.__action_map.items()
            }
            for state, values in self.__table.items()
        }

    def copy(self) -> 'QTable[GameStateType]':
        log.debug("Creating copy of QTable(uuid=%s)", self.__UUID)
        return QTable(*self.available_actions, _from=self.to_dict())

    def save(self, filename: str):
        dict_to_save = {
            "uuid": self.__UUID,
            "available_actions": self.available_actions,
            "q_table": self.to_dict(),
        }
        with open(filename, 'wb') as f:
            pickle.dump(dict_to_save, f)
        log.info("Save QTable(uuid=%s) to %s", self.__UUID, filename)

    @classmethod
    def load(cls, filename: str) -> 'QTable':  # can raise (pickle.PickleError, EOFError, FileNotFoundError)
        log.info("Loading QTable from %s", filename)
        with open(filename, 'rb') as f:
            saved_dict = pickle.load(f)
            return QTable(
                *saved_dict["available_actions"],
                _from=saved_dict["q_table"],
                _uuid=saved_dict["uuid"],
            )

    def __repr__(self):
        return f"QTable(uuid={self.uuid}, available_actions={self.available_actions!r}, length={self.__len__()})"

    def __str__(self):
        return self.uuid
