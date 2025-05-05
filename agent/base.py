from environment import GameState, GameAction
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def decide(self, state: GameState) -> GameAction:
        pass
