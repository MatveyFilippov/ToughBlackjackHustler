import logging
import random
from .base import QLearner, QLearnerRewardAfterAction
from ..q_table import QTable
from ...environment import GameAction, GameActionResult, GameEnvironment, GameStateType


log = logging.getLogger("tbjh.q_learning.strategies.epsilongreedy")


class EpsilonGreedyQLearner(QLearner):
    def __init__(
        self, game_environment: GameEnvironment[GameStateType], alpha: float, gamma: float, epsilon: float,
        agent_rewards: dict[GameActionResult, QLearnerRewardAfterAction], q_table: QTable[GameStateType] | None = None,
    ):
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in diapason [0-1]")
        self._EPSILON = epsilon

        self._REWARDS: dict[GameActionResult, QLearnerRewardAfterAction] = {}
        for action_result in GameActionResult:
            try:
                self._REWARDS[action_result] = agent_rewards[action_result]
            except KeyError:
                raise ValueError(f"You forget to set AgentReward for {action_result}")

        super().__init__(game_environment=game_environment, alpha=alpha, gamma=gamma, q_table=q_table)
        agent_rewards_str = "(" + ",".join(f"{k.name}:{v}" for k, v in self._REWARDS.items()) + ")"
        log.info(
            "Initialize EpsilonGreedyQLearner for QTable(uuid=%s) by %s with (Epsilon=%s, Alpha=%s, Gamma=%s) & AgentRewards=%s",
            self._Q_TABLE.uuid, self._GAME_ENVIRONMENT, self._EPSILON, self._ALPHA, self._GAMMA, agent_rewards_str,
        )

    def _choose_action(self, state: GameStateType) -> GameAction:
        if state not in self._Q_TABLE:
            log.debug("Use exploring GameAction because GameState not in QTable")
            return GameAction.get_by_random(*self._GAME_ENVIRONMENT.available_actions)
        elif random.random() < self._EPSILON:
            log.debug("Use exploring GameAction because random is less than Epsilon")
            return GameAction.get_by_random(*self._GAME_ENVIRONMENT.available_actions)
        else:
            log.debug("Use exploiting by best GameAction from QTable")
            return self._Q_TABLE.get_best_action(state)

    def _get_reward_for_action_result(self, action_result: GameActionResult) -> QLearnerRewardAfterAction:
        return self._REWARDS[action_result]

    def __repr__(self):
        return f"EpsilonGreedyQLearner(epsilon={self._EPSILON}, alpha={self._ALPHA}, gamma={self._GAMMA}, q_table={self._Q_TABLE!r}, game_environment={self._GAME_ENVIRONMENT!r}, agent_rewards={self._REWARDS!r})"

    def __str__(self):
        agent_rewards_str = "(" + ",".join(f"{k.name}:{v}" for k, v in self._REWARDS.items()) + ")"
        return f"EpsilonGreedyQLearner(epsilon={self._EPSILON}, alpha={self._ALPHA}, gamma={self._GAMMA}, q_table_uuid={self._Q_TABLE.uuid}, game_environment={self._GAME_ENVIRONMENT}, agent_rewards={agent_rewards_str})"
