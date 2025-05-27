from threading import Thread
from agent import Agent
from environment import GameEnvironment, GameActionResult


class GameSimulator:
    def __init__(self, game_environment: GameEnvironment, agent: Agent):
        self.__GAME_ENVIRONMENT = game_environment
        self.__AGENT = agent

        self.__score = 0.0
        self.__thread: Thread = None

    def __count_up(self, result: GameActionResult):
        if result in (GameActionResult.WINS, GameActionResult.BLACKJACK):
            self.__score += 1
        elif result in (GameActionResult.LOSS, GameActionResult.BUST):
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
        self.__thread = Thread(target=self.__run, args=(iterations,), name="Simulate game", daemon=True)
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
