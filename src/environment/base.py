from abc import ABC, abstractmethod
from enum import IntEnum, auto
from functools import cached_property, lru_cache
import random
from typing import Generic, Iterator, NamedTuple, TypeVar
import numpy as np
from sortedcontainers import SortedList


class Card:
    def __init__(self, rank: int):
        if not 2 <= rank <= 10:
            raise ValueError(f"Invalid rank. Available only [2-10]")
        self._rank = rank
        self._hash = None

    @property
    def rank(self) -> int:
        return self._rank

    def __int__(self) -> int:
        return self._rank

    def __add__(self, other):
        if isinstance(other, Card):
            return self._rank + other._rank
        elif isinstance(other, int):
            return self._rank + other
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Card):
            return self._rank + other._rank
        elif isinstance(other, int):
            return self._rank + other
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Card):
            return self._rank == other._rank
        elif isinstance(other, int):
            return self._rank == other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Card):
            return self._rank < other._rank
        elif isinstance(other, int):
            return self._rank < other
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Card):
            return self._rank <= other._rank
        elif isinstance(other, int):
            return self._rank <= other
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Card):
            return self._rank > other._rank
        elif isinstance(other, int):
            return self._rank > other
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Card):
            return self._rank >= other._rank
        elif isinstance(other, int):
            return self._rank >= other
        return NotImplemented

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._rank)
        return self._hash

    def __repr__(self):
        return f"Card(rank={self._rank})"

    def __str__(self):
        return str(self._rank)


class AceCard(Card):
    def __init__(self, is_soft: bool = True):
        super().__init__(rank=7)  # Temp init by 7 to exclude ValueError
        self._rank = 11 if is_soft else 1

    @property
    def is_soft(self) -> bool:
        return self._rank == 11

    def __repr__(self):
        return f"AceCard(rank={self.rank}, is_soft={self.is_soft})"

    def __str__(self):
        return ("Soft" if self.is_soft else "Hard") + "Ace"


class CardDeck:
    __SOFT_ACE_CARD = AceCard(is_soft=True)  # Ace in deck is always soft
    __ALL_CARDS = [Card(__rank) for __rank in range(2, 11)] + [Card(10) for _ in range(3)] + [__SOFT_ACE_CARD]
    MIN_DECK_PERCENT = 0.25

    @classmethod
    @lru_cache
    def __get_default_deck(cls, qty: int) -> list[Card]:
        return sorted(np.repeat(cls.__ALL_CARDS, 4 * qty))

    def __new__(cls, qty: int = 1):
        if qty < 1:
            raise ValueError("Quantity of decks must be positive")
        __obj = super().__new__(cls)
        __obj.__init_decks_qty = qty
        __obj.__min_cards_qty = 52 * qty * cls.MIN_DECK_PERCENT
        return __obj

    def __init__(self, qty: int = 1):
        self.__deck = SortedList(self.__get_default_deck(qty))

    @classmethod
    def of(cls, init_decks_qty: int, remaining_cards: list[Card]) -> 'CardDeck':
        obj = cls.__new__(cls, qty=init_decks_qty)
        obj.__deck = SortedList(
            cls.__SOFT_ACE_CARD if isinstance(card, AceCard) else card
            for card in remaining_cards
        )
        return obj

    def reset(self):
        self.__deck.clear()
        self.__deck.update(self.__get_default_deck(self.__init_decks_qty))

    def __contains__(self, item) -> bool:
        if isinstance(item, Card):
            return item in self.__deck
        return NotImplemented

    def __iter__(self) -> Iterator[Card]:
        for card in self.__deck:
            yield card

    def __len__(self):
        return len(self.__deck)

    def __hash__(self):  # Only for using in cache
        return hash(tuple(self.__deck))

    def draw(self) -> Card:
        cards_remain = self.__len__()
        if cards_remain == 0:
            raise IndexError("All cards in the deck have already been used.")
        return self.__deck.pop(random.randint(0, cards_remain - 1))

    def count(self, card: Card) -> int:
        return self.__deck.count(self.__SOFT_ACE_CARD if isinstance(card, AceCard) else card)

    @property
    def init_decks_qty(self) -> int:
        return self.__init_decks_qty

    @property
    def is_playable(self) -> bool:
        return self.__len__() >= self.__min_cards_qty

    @property
    def remaining_cards(self) -> list[Card]:
        return list(self.__deck)

    def copy_with(self, *cards_to_add: Card) -> 'CardDeck':
        return self.of(
            init_decks_qty=self.__init_decks_qty,
            remaining_cards=self.remaining_cards + list(cards_to_add),
        )

    def __repr__(self):
        return f"CardDeck(init_decks_qty={self.init_decks_qty}, is_playable={self.is_playable}, remaining_cards={self.__deck!r})"


class CardHand:
    __SOFT_ACE_CARD = AceCard(is_soft=True)
    __HARD_ACE_CARD = AceCard(is_soft=False)

    def __init__(self, *cards: Card):
        self.__cards: list[Card] = []
        self.__invalidate_total()
        self.add(*cards)

    def __invalidate_total(self):
        try:
            del self.total
        except AttributeError:
            pass

    @property
    def cards(self) -> tuple[Card, ...]:
        return tuple(self.__cards)

    @cached_property
    def total(self) -> int:
        return sum(int(card) for card in self.__cards)

    def __len__(self):
        return len(self.__cards)

    def __getitem__(self, index: int) -> Card:
        return self.__cards[index]

    def __iter__(self) -> Iterator[Card]:
        for card in self.__cards:
            yield card

    def __hash__(self):  # Only for using in cache
        return hash(tuple(sorted(self.__cards)))

    def __eq__(self, other):
        if isinstance(other, CardHand):
            return self.total == other.total
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, CardHand):
            return self.total < other.total
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, CardHand):
            return self.total <= other.total
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, CardHand):
            return self.total > other.total
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, CardHand):
            return self.total >= other.total
        return NotImplemented

    @property
    def is_soft(self) -> bool:
        return any(card == self.__SOFT_ACE_CARD for card in self.__cards)

    def __migrate_to_hard_if_needed(self):
        if self.total <= 21:
            return
        for i in range(len(self.__cards)):
            if self.__cards[i] == self.__SOFT_ACE_CARD:
                self.__cards[i] = self.__HARD_ACE_CARD
                self.__invalidate_total()
                if self.total <= 21:
                    return

    def add(self, *cards: Card):
        self.__cards.extend(
            self.__SOFT_ACE_CARD if isinstance(card, AceCard) else card  # Always add ace as soft
            for card in cards
        )
        self.__invalidate_total()
        self.__migrate_to_hard_if_needed()

    def clean(self):
        self.__cards.clear()
        self.__invalidate_total()

    def __repr__(self):
        return f"CardHand(is_soft={self.is_soft}, total={self.total}, cards={self.__cards!r})"

    def __str__(self):
        cards_str = ','.join(str(c) for c in self.__cards)
        return f"{'soft' if self.is_soft else 'hard'}_{self.total}:Cards({cards_str})"


class GameAction(IntEnum):
    STAND = auto()
    HIT = auto()
    SPLIT = auto()
    DOUBLE_DOWN = auto()
    INSURANCE = auto()
    SURRENDER = auto()

    @classmethod
    def get_by_random(cls, *actions: 'GameAction') -> 'GameAction':
        return random.choice(actions)


class GameActionResult(IntEnum):
    WAIT_ACTION = auto()
    BLACKJACK = auto()
    WINS = auto()
    PUSH = auto()
    LOSS = auto()
    BUST = auto()


GameState = NamedTuple
GameStateType = TypeVar('GameStateType', bound=GameState)


class GameEnvironment(Generic[GameStateType], ABC):
    @property
    @abstractmethod
    def is_terminated(self) -> bool:
        ...

    @abstractmethod
    def reset(self):
        ...

    @property
    @abstractmethod
    def is_round_playing(self) -> bool:
        ...

    @abstractmethod
    def start_new_round(self):
        ...

    @property
    @abstractmethod
    def state(self) -> GameStateType:
        ...

    @property
    @abstractmethod
    def available_actions(self) -> tuple[GameAction, ...]:
        ...

    @abstractmethod
    def play(self, game_action: GameAction) -> GameActionResult:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...
