from .base import GameEnvironment, GameState, GameAction, GameActionResult, CardDeck, CardHand
from .probability_tools import (
    calculate_player_busting_probability, calculate_dealer_busting_probability, calculate_dealer_will_take_cards_probability
)
from functools import lru_cache


class DefaultGameState(GameState):
    player_cards_qty: int  # [2-11]
    player_cards_sum: int  # [4-21]
    player_has_soft_hand: int  # [0, 1]
    player_busting_probability: float  # [0-1]
    dealer_open_card: int  # [2-11]
    dealer_cards_sum_less_than_17_probability: float  # [0-1]
    dealer_busting_probability: float  # [0-1]

    @staticmethod
    def round_probability(probability: float) -> float:
        return round(probability, 2)


@lru_cache
def _calculate_player_busting_probability(deck: CardDeck, player: CardHand, dealer: CardHand) -> float:
    return DefaultGameState.round_probability(calculate_player_busting_probability(
        CardDeck.of(deck.init_decks_qty, deck.remaining_cards + [dealer[1]]), player
    ))


@lru_cache
def _calculate_dealer_cards_sum_less_than_17_probability(deck: CardDeck, dealer: CardHand, hit_on_soft_17: bool) -> float:
    return DefaultGameState.round_probability(calculate_dealer_will_take_cards_probability(
        CardDeck.of(deck.init_decks_qty, deck.remaining_cards + [dealer[1]]), dealer[0], hit_on_soft_17=hit_on_soft_17,
    ))


@lru_cache
def _calculate_dealer_busting_probability(deck: CardDeck, dealer: CardHand, hit_on_soft_17: bool) -> float:
    return DefaultGameState.round_probability(calculate_dealer_busting_probability(
        CardDeck.of(deck.init_decks_qty, deck.remaining_cards + [dealer[1]]), dealer[0], hit_on_soft_17=hit_on_soft_17,
    ))


class DefaultGame(GameEnvironment):
    def __init__(self, card_decks_qty: int, dealer_hit_on_soft_17: bool | None = False):
        self.__AVAILABLE_ACTIONS = (GameAction.STAND, GameAction.HIT)

        self.__CARD_DECK: CardDeck = CardDeck(card_decks_qty)
        self.__PLAYER_HAND: CardHand = CardHand()
        self.__DEALER_HAND: CardHand = CardHand()

        self.__DEALER_HIT_ON_SOFT_17: bool = dealer_hit_on_soft_17
        self.__is_round_playing: bool = False

    @property
    def available_actions(self) -> tuple[GameAction, ...]:
        return self.__AVAILABLE_ACTIONS

    def reset(self):
        self.__CARD_DECK.reset()
        self.__start_new_round()

    def __start_new_round(self):
        self.__PLAYER_HAND.clean()
        self.__PLAYER_HAND.add(self.__CARD_DECK.draw(), self.__CARD_DECK.draw())

        self.__DEALER_HAND.clean()
        self.__DEALER_HAND.add(self.__CARD_DECK.draw(), self.__CARD_DECK.draw())

    def __play_hit(self) -> tuple[GameActionResult, bool]:
        self.__PLAYER_HAND.add(self.__CARD_DECK.draw())
        player_sum = sum(self.__PLAYER_HAND)
        if player_sum > 21:
            return GameActionResult.BUST, True
        elif player_sum == 21:
            return GameActionResult.BLACKJACK, True
        return GameActionResult.WAIT_ACTION, False

    def __play_stand(self) -> GameActionResult:
        while sum(self.__DEALER_HAND) < 17 or (self.__DEALER_HIT_ON_SOFT_17 and self.__DEALER_HAND.is_soft):
            self.__DEALER_HAND.add(self.__CARD_DECK.draw())
        if sum(self.__DEALER_HAND) > 21 or self.__DEALER_HAND < self.__PLAYER_HAND:
            return GameActionResult.WINS
        elif self.__DEALER_HAND == self.__PLAYER_HAND:
            return GameActionResult.PUSH
        elif self.__DEALER_HAND > self.__PLAYER_HAND:
            return GameActionResult.LOSS

    def play(self, game_action: GameAction) -> GameActionResult:
        if game_action == GameAction.HIT:
            result, is_round_over = self.__play_hit()
        elif game_action == GameAction.STAND:
            result = self.__play_stand()
            is_round_over = True
        else:
            raise ValueError(f"Invalid GameAction, available only: {self.__AVAILABLE_ACTIONS}")

        self.__is_round_playing = not is_round_over
        if is_round_over:
            self.__start_new_round()
        return result

    @property
    def is_terminated(self) -> bool:
        return not self.__is_round_playing and not self.__CARD_DECK.is_playable

    @property
    def state(self) -> DefaultGameState:
        return DefaultGameState(
            player_cards_qty=len(self.__PLAYER_HAND),
            player_cards_sum=sum(self.__PLAYER_HAND),
            player_has_soft_hand=int(self.__PLAYER_HAND.is_soft),
            player_busting_probability=_calculate_player_busting_probability(
                deck=self.__CARD_DECK, player=self.__PLAYER_HAND, dealer=self.__DEALER_HAND,
            ),
            dealer_open_card=self.__DEALER_HAND[0].rank,
            dealer_cards_sum_less_than_17_probability=_calculate_dealer_cards_sum_less_than_17_probability(
                deck=self.__CARD_DECK, dealer=self.__DEALER_HAND, hit_on_soft_17=self.__DEALER_HIT_ON_SOFT_17,
            ),
            dealer_busting_probability=_calculate_dealer_busting_probability(
                deck=self.__CARD_DECK, dealer=self.__DEALER_HAND, hit_on_soft_17=self.__DEALER_HIT_ON_SOFT_17,
            ),
        )
