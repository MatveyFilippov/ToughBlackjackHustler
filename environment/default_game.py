from .base import (
    GameState, GameEnvironment, UserAction, AgentReward, CardDeck, CardHand
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
def __calculate_hand_sum_over_by_next_card_probability(deck: CardDeck, hand: CardHand, over: int) -> float:
    not_bust_rank = over - sum(hand)
    return sum(1 for card in deck if card.rank > not_bust_rank) / len(deck)


def _calculate_player_busting_probability(deck: CardDeck, hand: CardHand) -> float:
    return __calculate_hand_sum_over_by_next_card_probability(deck, hand, 21)


@lru_cache
def _calculate_dealer_cards_sum_less_than_17_probability(deck: CardDeck, hand: CardHand) -> float:
    return 1 - __calculate_hand_sum_over_by_next_card_probability(
        CardDeck.of(deck.init_decks_qty, deck.remaining_cards + [hand[1]]), CardHand(hand[0]), 17
    )


@lru_cache
def _calculate_dealer_busting_probability(deck: CardDeck, hand: CardHand) -> float:
    total_bust = 0
    total_possibilities = 0

    remaining_cards = deck.remaining_cards + [hand[1]]
    for probable_hidden_card in set(remaining_cards):
        probable_hidden_card_count = remaining_cards.count(probable_hidden_card)
        probable_hand = CardHand(hand[0], probable_hidden_card)

        probable_deck_cards = remaining_cards.copy()
        probable_deck_cards.remove(probable_hidden_card)
        probable_deck = CardDeck.of(deck.init_decks_qty, probable_deck_cards)

        bust, possibilities = __simulate_dealer(probable_deck, probable_hand)

        total_bust += bust * probable_hidden_card_count
        total_possibilities += possibilities * probable_hidden_card_count

    return 0.0 if total_possibilities == 0 else total_bust / total_possibilities


@lru_cache
def __simulate_dealer(deck: CardDeck, hand: CardHand) -> tuple[int, int]:
    current_sum = sum(hand)
    if current_sum >= 17:
        return (1 if current_sum > 21 else 0), 1

    total_bust = 0
    total_possibilities = 0

    for card in set(deck):
        card_count = deck.count(card)

        new_hand = CardHand(*hand)
        new_hand.add(card)

        new_deck_cards = deck.remaining_cards
        new_deck_cards.remove(card)
        new_deck = CardDeck.of(deck.init_decks_qty, new_deck_cards)

        bust, possibilities = __simulate_dealer(new_deck, new_hand)

        total_bust += bust * card_count
        total_possibilities += possibilities * card_count

    return total_bust, total_possibilities


class DefaultGameAgentRewards:
    BLACKJACK = AgentReward(1.5)
    WINS = AgentReward(1.0)
    PUSH = AgentReward(0.5)
    LOSS = AgentReward(-1.0)
    BUST = AgentReward(-1.5)


class DefaultGame(GameEnvironment):
    def __init__(self, card_decks_qty: int):
        self.__CARD_DECK: CardDeck = CardDeck(card_decks_qty)
        self.__PLAYER_HAND: CardHand = CardHand()
        self.__DEALER_HAND: CardHand = CardHand()
        self.__is_round_playing: bool = False

    def reset(self):
        self.__CARD_DECK.reset()
        self.__start_new_round()

    def __start_new_round(self):
        self.__PLAYER_HAND.clean()
        self.__PLAYER_HAND.add(self.__CARD_DECK.draw(), self.__CARD_DECK.draw())

        self.__DEALER_HAND.clean()
        self.__DEALER_HAND.add(self.__CARD_DECK.draw(), self.__CARD_DECK.draw())

    def __play_hit(self) -> tuple[AgentReward, bool]:
        self.__PLAYER_HAND.add(self.__CARD_DECK.draw())
        player_sum = sum(self.__PLAYER_HAND)
        if player_sum > 21:
            return DefaultGameAgentRewards.BUST, True
        elif player_sum == 21:
            return DefaultGameAgentRewards.BLACKJACK, True
        return AgentReward.NEUTRAL, False

    def __play_stand(self) -> AgentReward:
        while sum(self.__DEALER_HAND) < 17:
            self.__DEALER_HAND.add(self.__CARD_DECK.draw())
        if sum(self.__DEALER_HAND) > 21 or self.__DEALER_HAND < self.__PLAYER_HAND:
            return DefaultGameAgentRewards.WINS
        elif self.__DEALER_HAND == self.__PLAYER_HAND:
            return DefaultGameAgentRewards.PUSH
        elif self.__DEALER_HAND > self.__PLAYER_HAND:
            return DefaultGameAgentRewards.LOSS

    def play(self, user_action: UserAction) -> AgentReward:
        reward = AgentReward.NEUTRAL
        is_round_over = False

        if user_action == UserAction.HIT:
            reward, is_round_over = self.__play_hit()
        elif user_action == UserAction.STAND:
            reward = self.__play_stand()
            is_round_over = True

        self.__is_round_playing = not is_round_over
        if is_round_over:
            self.__start_new_round()
        return reward

    @property
    def is_terminated(self) -> bool:
        return not self.__is_round_playing and not self.__CARD_DECK.is_playable

    @property
    def state(self) -> DefaultGameState:
        return DefaultGameState(
            player_cards_qty=len(self.__PLAYER_HAND),
            player_cards_sum=sum(self.__PLAYER_HAND),
            player_has_soft_hand=int(self.__PLAYER_HAND.is_soft),
            player_busting_probability=DefaultGameState.round_probability(_calculate_player_busting_probability(
                deck=self.__CARD_DECK, hand=self.__PLAYER_HAND,
            )),
            dealer_open_card=self.__DEALER_HAND[0].rank,
            dealer_cards_sum_less_than_17_probability=DefaultGameState.round_probability(_calculate_dealer_cards_sum_less_than_17_probability(
                deck=self.__CARD_DECK, hand=self.__DEALER_HAND,
            )),
            dealer_busting_probability=DefaultGameState.round_probability(_calculate_dealer_busting_probability(
                deck=self.__CARD_DECK, hand=self.__DEALER_HAND,
            )),
        )
