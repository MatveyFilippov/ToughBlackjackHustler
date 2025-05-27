from environment import probability_tools as prob
from environment.base import Card, CardHand, CardDeck
from environment.default_game import DefaultGameState
from functools import lru_cache


@lru_cache
def __get_remaining_cards(*used_cards: Card, decks_qty: int) -> list[Card]:
    result = CardDeck(decks_qty).remaining_cards
    for card in used_cards:
        if card not in result:
            raise ValueError("The deck can't contain such a card (or not in such quantity)")
        result.remove(card)
    return result


@lru_cache
def get_game_state(player: tuple[Card, ...], dealer_open: Card,
                   decks_qty: int, used_cards: tuple[Card, ...]) -> DefaultGameState:
    used_cards = used_cards + player + (dealer_open,)
    player_hand = CardHand(*player)
    cards_deck = CardDeck.of(decks_qty, __get_remaining_cards(*used_cards, decks_qty=decks_qty))
    return DefaultGameState(
        player_cards_qty=len(player_hand),
        player_cards_sum=sum(player_hand),
        player_has_soft_hand=int(player_hand.is_soft),
        dealer_open_card=dealer_open.rank,
        player_busting_probability=DefaultGameState.round_probability(
            prob.calculate_player_busting_probability(cards_deck, player_hand)
        ),
        dealer_cards_sum_less_than_17_probability=DefaultGameState.round_probability(
            prob.calculate_dealer_will_take_cards_probability(cards_deck, dealer_open)
        ),
        dealer_busting_probability=DefaultGameState.round_probability(
            prob.calculate_dealer_busting_probability(cards_deck, dealer_open)
        ),
    )
