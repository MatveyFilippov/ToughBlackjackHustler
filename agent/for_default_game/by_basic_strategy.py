from ..base import Agent
from environment import GameAction
from environment.default_game import DefaultGameState


class AgentForDefaultGameByBasicStrategy(Agent):
    def __init__(self, ignore_probabilities: bool = False):
        self.__ignore_probabilities = ignore_probabilities

    @staticmethod
    def soft_hand_strategy(player: int, dealer: int) -> GameAction:
        if player >= 19:
            return GameAction.STAND
        elif player == 18:
            if dealer in (2, 7, 8):
                return GameAction.STAND
            else:
                return GameAction.HIT
        else:  # 13-17
            return GameAction.HIT

    @staticmethod
    def hard_hand_strategy(player: int, dealer: int) -> GameAction:
        if player >= 17:
            return GameAction.STAND
        elif player <= 11:
            return GameAction.HIT
        else:  # 12-16
            if dealer >= 7:
                return GameAction.HIT
            else:
                return GameAction.STAND

    def decide(self, state: DefaultGameState) -> GameAction:
        player = state.player_cards_sum
        dealer = state.dealer_open_card
        base_strategy = self.soft_hand_strategy if state.player_has_soft_hand else self.hard_hand_strategy

        if self.__ignore_probabilities:
            return base_strategy(player=player, dealer=dealer)

        if state.player_busting_probability > 0.7:
            return GameAction.STAND
        if state.player_busting_probability > 0.35 and state.dealer_cards_sum_less_than_17_probability < 0.6:
            return GameAction.STAND
        if state.dealer_busting_probability > 0.8 and player < 16 and state.player_busting_probability < 0.3:
            return GameAction.HIT

        return base_strategy(player=player, dealer=dealer)
