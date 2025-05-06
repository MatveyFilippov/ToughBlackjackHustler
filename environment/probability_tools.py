from environment.base import Card, CardDeck, CardHand
from functools import lru_cache


@lru_cache
def __calculate_hand_sum_over_by_next_card_probability(deck: CardDeck, hand: CardHand, over: int) -> float:
    not_over_rank = over - sum(1 if card.rank == 11 else card.rank for card in hand)
    return sum(1 for card in deck if (card.rank if card.rank != 11 else 1) > not_over_rank) / len(deck)


def calculate_player_busting_probability(deck: CardDeck, player: CardHand) -> float:
    return __calculate_hand_sum_over_by_next_card_probability(deck, player, 21)


@lru_cache
def calculate_dealer_will_take_cards_probability(deck: CardDeck, dealer_open_card: Card, hit_on_soft_17=False):
    if hit_on_soft_17:
        return 1 - __calculate_hand_sum_over_by_next_card_probability(deck, CardHand(dealer_open_card), 17)
    else:
        not_over_rank = 17 - dealer_open_card.rank
        return sum(1 for card in deck if card.rank < not_over_rank) / len(deck)


@lru_cache
def calculate_dealer_busting_probability(deck: CardDeck, open_card: Card, hit_on_soft_17=False) -> float:
    total_bust = 0
    total_possibilities = 0

    remaining_cards = deck.remaining_cards
    for probable_hidden_card in set(remaining_cards):
        probable_hidden_card_count = remaining_cards.count(probable_hidden_card)
        probable_hand = CardHand(open_card, probable_hidden_card)

        probable_deck_cards = remaining_cards.copy()
        probable_deck_cards.remove(probable_hidden_card)
        probable_deck = CardDeck.of(deck.init_decks_qty, probable_deck_cards)

        bust, possibilities = __simulate_dealer(probable_deck, probable_hand, hit_on_soft_17)

        total_bust += bust * probable_hidden_card_count
        total_possibilities += possibilities * probable_hidden_card_count

    return 0.0 if total_possibilities == 0 else total_bust / total_possibilities


@lru_cache
def __simulate_dealer(deck: CardDeck, hand: CardHand, hit_on_soft_17: bool) -> tuple[int, int]:
    current_sum = sum(hand)
    if current_sum >= 17 and not (hit_on_soft_17 and hand.is_soft):
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

        bust, possibilities = __simulate_dealer(new_deck, new_hand, hit_on_soft_17)

        total_bust += bust * card_count
        total_possibilities += possibilities * card_count

    return total_bust, total_possibilities
