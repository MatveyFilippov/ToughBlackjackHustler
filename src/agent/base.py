from abc import ABC, abstractmethod
from typing import Generic
from ..environment import GameAction, GameStateType


class Agent(Generic[GameStateType], ABC):
    @abstractmethod
    def decide(self, state: GameStateType) -> GameAction:
        ...
