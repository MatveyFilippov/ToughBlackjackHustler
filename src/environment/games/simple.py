from functools import lru_cache
import logging
from ..base import CardDeck, CardHand, GameAction, GameActionResult, GameEnvironment, GameState
from ..probability_tools import (
    calculate_dealer_busting_probability, calculate_dealer_will_take_cards_probability,
    calculate_player_busting_probability,
)


log = logging.getLogger("tbjh.environment.games.simple")


class SimpleGameState(GameState):
    player_cards_qty: int  # [2-21]
    player_cards_sum: int  # [4-21]
    player_has_soft_hand: int  # [0, 1]
    player_busting_probability: float  # [0-1]
    dealer_open_card: int  # [2-11]
    dealer_cards_sum_less_than_17_probability: float  # [0-1]
    dealer_busting_probability: float  # [0-1]

    @staticmethod
    def round_probability(probability: float) -> float:  # All probabilities in SimpleGameState must be rounded
        return round(probability, 2)


@lru_cache
def _calculate_player_busting_simple_probability(deck: CardDeck, player: CardHand, dealer: CardHand) -> float:
    return SimpleGameState.round_probability(
        calculate_player_busting_probability(
            deck=deck.copy_with(dealer[1]),
            player=player,
        ),
    )


@lru_cache
def _calculate_dealer_cards_sum_less_than_17_simple_probability(deck: CardDeck, dealer: CardHand, hit_on_soft_17: bool) -> float:
    return SimpleGameState.round_probability(
        calculate_dealer_will_take_cards_probability(
            deck=deck.copy_with(dealer[1]),
            dealer_open_card=dealer[0],
            hit_on_soft_17=hit_on_soft_17,
        ),
    )


@lru_cache
def _calculate_dealer_busting_simple_probability(deck: CardDeck, dealer: CardHand, hit_on_soft_17: bool) -> float:
    return SimpleGameState.round_probability(
        calculate_dealer_busting_probability(
            deck=deck.copy_with(dealer[1]),
            open_card=dealer[0],
            hit_on_soft_17=hit_on_soft_17,
        ),
    )


class SimpleGame(GameEnvironment[SimpleGameState]):
    __AVAILABLE_ACTIONS = (
        GameAction.STAND,
        GameAction.HIT,
    )

    def __init__(self, card_decks_qty: int, dealer_hit_on_soft_17: bool = False):
        self.__CARD_DECK = CardDeck(qty=card_decks_qty)
        self.__PLAYER_HAND = CardHand()
        self.__DEALER_HAND = CardHand()

        self.__DEALER_HIT_ON_SOFT_17 = dealer_hit_on_soft_17
        self.__is_round_playing = False

        log.debug(
            "Initialize SimpleGame with %s CardDecks, Dealer %s hit on soft 17",
            self.__CARD_DECK.init_decks_qty, ("will" if self.__DEALER_HIT_ON_SOFT_17 else "won't"),
        )

    @property
    def is_terminated(self) -> bool:
        return not self.__is_round_playing and not self.__CARD_DECK.is_playable

    def reset(self):
        self.__CARD_DECK.reset()
        log.debug("Reset CardDeck")

    @property
    def is_round_playing(self) -> bool:
        return self.__is_round_playing

    def start_new_round(self):
        log.debug("Starting new round")
        if not self.__CARD_DECK.is_playable:
            raise ValueError("Can't start new round because CardDeck is not playable, call reset()")

        self.__PLAYER_HAND.clean()
        self.__PLAYER_HAND.add(self.__CARD_DECK.draw(), self.__CARD_DECK.draw())
        log.debug("Draw %s&%s for Player", *self.__PLAYER_HAND.cards)

        self.__DEALER_HAND.clean()
        self.__DEALER_HAND.add(self.__CARD_DECK.draw(), self.__CARD_DECK.draw())
        log.debug("Draw %s&%s for Dealer", *self.__DEALER_HAND.cards)

        self.__is_round_playing = True
        log.debug("Start new round: Player(%s) vs Dealer(%s)", self.__PLAYER_HAND, self.__DEALER_HAND)

    @property
    def state(self) -> SimpleGameState:
        if not self.__is_round_playing:
            raise ValueError("State doesn't exist because round is not playing, call start_new_round()")

        return SimpleGameState(
            player_cards_qty=len(self.__PLAYER_HAND),
            player_cards_sum=self.__PLAYER_HAND.total,
            player_has_soft_hand=int(self.__PLAYER_HAND.is_soft),
            player_busting_probability=_calculate_player_busting_simple_probability(
                deck=self.__CARD_DECK, player=self.__PLAYER_HAND, dealer=self.__DEALER_HAND,
            ),
            dealer_open_card=self.__DEALER_HAND[0].rank,
            dealer_cards_sum_less_than_17_probability=_calculate_dealer_cards_sum_less_than_17_simple_probability(
                deck=self.__CARD_DECK, dealer=self.__DEALER_HAND, hit_on_soft_17=self.__DEALER_HIT_ON_SOFT_17,
            ),
            dealer_busting_probability=_calculate_dealer_busting_simple_probability(
                deck=self.__CARD_DECK, dealer=self.__DEALER_HAND, hit_on_soft_17=self.__DEALER_HIT_ON_SOFT_17,
            ),
        )

    @property
    def available_actions(self) -> tuple[GameAction, ...]:
        return self.__AVAILABLE_ACTIONS

    @property
    def dealer_hit_on_soft_17(self) -> bool:
        return self.__DEALER_HIT_ON_SOFT_17

    def __play_hit(self) -> GameActionResult:
        player_new_card = self.__CARD_DECK.draw()
        self.__PLAYER_HAND.add(player_new_card)
        log.debug("Draw %s for Player", player_new_card)

        if self.__PLAYER_HAND.total > 21:
            log.debug("Player(%s) catch Bust", self.__PLAYER_HAND)
            return GameActionResult.BUST
        elif self.__PLAYER_HAND.total == 21:
            log.debug("Player(%s) catch Blackjack", self.__PLAYER_HAND)
            return GameActionResult.BLACKJACK
        log.debug("Player(%s) waiting new GameAction", self.__PLAYER_HAND)
        return GameActionResult.WAIT_ACTION

    def __play_stand(self) -> GameActionResult:
        while self.__DEALER_HAND.total < 17 or (self.__DEALER_HIT_ON_SOFT_17 and self.__DEALER_HAND.total == 17 and self.__DEALER_HAND.is_soft):
            dealer_new_card = self.__CARD_DECK.draw()
            self.__DEALER_HAND.add(dealer_new_card)
            log.debug("Draw %s for Dealer", dealer_new_card)

        if self.__DEALER_HAND.total > 21 or self.__DEALER_HAND < self.__PLAYER_HAND:
            log.debug("Player(%s) wins Dealer(%s)", self.__PLAYER_HAND, self.__DEALER_HAND)
            return GameActionResult.WINS
        elif self.__DEALER_HAND == self.__PLAYER_HAND:
            log.debug("Player(%s) push Dealer(%s)", self.__PLAYER_HAND, self.__DEALER_HAND)
            return GameActionResult.PUSH
        elif self.__DEALER_HAND > self.__PLAYER_HAND:
            log.debug("Player(%s) loss Dealer(%s)", self.__PLAYER_HAND, self.__DEALER_HAND)
            return GameActionResult.LOSS

    def play(self, game_action: GameAction) -> GameActionResult:
        if not self.__is_round_playing:
            raise ValueError("Can't play GameAction because round is not playing, call start_new_round()")

        log.debug("Play %s", game_action.name)
        if game_action == GameAction.HIT:
            result = self.__play_hit()
            self.__is_round_playing = result == GameActionResult.WAIT_ACTION
        elif game_action == GameAction.STAND:
            result = self.__play_stand()
            self.__is_round_playing = False
        else:
            raise ValueError(f"Invalid GameAction, available only: {self.__AVAILABLE_ACTIONS}")

        return result

    def __repr__(self):
        round_str = (
            f"(Player({self.__PLAYER_HAND!r}) vs Dealer({self.__DEALER_HAND!r}))"
            if self.__is_round_playing else
            None
        )
        return f"SimpleGame(card_deck={self.__CARD_DECK!r}, dealer_hit_on_soft_17={self.dealer_hit_on_soft_17}, round={round_str})"

    def __str__(self):
        return f"SimpleGame(card_decks_qty={self.__CARD_DECK.init_decks_qty}, dealer_hit_on_soft_17={self.dealer_hit_on_soft_17})"
