import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache
from typing import NamedTuple, Generator
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

    def copy(self) -> 'Card':
        return Card(self._rank)

    def __add__(self, other):
        if isinstance(other, Card):
            return self._rank + other._rank
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
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Card):
            return self._rank < other._rank
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Card):
            return self._rank <= other._rank
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Card):
            return self._rank > other._rank
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Card):
            return self._rank >= other._rank
        return NotImplemented

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._rank)
        return self._hash

    def __str__(self):
        return f"Card(rank={self._rank})"


class AceCard(Card):
    def __init__(self, is_soft: bool | None = True):
        super().__init__(7)
        self._rank = 11 if is_soft else 1

    @property
    def is_soft(self) -> bool:
        return self._rank == 11

    def set_hard(self):
        if self.is_soft:
            self._rank = 1
            self._hash = None

    def copy(self) -> 'AceCard':
        return AceCard(self.is_soft)

    def __str__(self):
        return f"AceCard(is_soft={self.is_soft})"


class CardDeck:
    @staticmethod
    @lru_cache
    def __get_default_deck(qty: int) -> list[Card]:
        return sorted(np.repeat(
            [Card(__rank) for __rank in range(2, 11)] + [Card(10) for _ in range(3)] + [AceCard()], 4 * qty
        ))

    def __new__(cls, qty: int | None = 1):
        if qty < 1:
            raise ValueError("QTY of decks must be greater than 0")
        __obj = super().__new__(cls)
        __obj.__init_decks_qty = qty
        __obj.__min_cards_qty = 52 * qty * 0.25
        return __obj

    def __init__(self, qty: int | None = 1):
        self.__deck: SortedList[Card] = SortedList(self.__get_default_deck(qty))

    def reset(self):
        self.__deck.clear()
        self.__deck.update(self.__get_default_deck(self.__init_decks_qty))

    def draw(self) -> Card:
        cards_remain = self.__len__()
        if cards_remain == 0:
            raise IndexError("All cards in the deck have already been used.")
        return self.__deck.pop(random.randint(0, cards_remain - 1)).copy()

    def __contains__(self, item) -> bool:
        if isinstance(item, Card):
            return item in self.__deck
        return NotImplemented

    def __len__(self):
        return len(self.__deck)

    def __hash__(self):
        return hash(tuple(self.__deck))

    def __iter__(self) -> Generator[Card, None, None]:
        return (card.copy() for card in self.__deck)

    def count(self, card: Card) -> int:
        return self.__deck.count(AceCard() if isinstance(card, AceCard) else card)

    @property
    def init_decks_qty(self) -> int:
        return self.__init_decks_qty

    @property
    def is_playable(self) -> bool:
        return self.__len__() >= self.__min_cards_qty

    @property
    def remaining_cards(self) -> list[Card]:
        return list(self)

    @classmethod
    def of(cls, init_decks_qty: int, remaining_cards: list[Card]) -> 'CardDeck':
        obj = cls.__new__(cls, init_decks_qty)
        obj.__deck = SortedList(AceCard() if isinstance(card, AceCard) else card.copy() for card in remaining_cards)
        return obj


class CardHand:
    __SOFT_ACE = AceCard(is_soft=True)

    def __init__(self, *cards: Card):
        self.__cards: list[Card] = []
        self.add(*cards)

    def __len__(self):
        return len(self.__cards)

    def __getitem__(self, index) -> Card:
        return self.__cards[index].copy()

    def __iter__(self) -> Generator[Card, None, None]:
        return (card.copy() for card in self.__cards)

    def __hash__(self):
        return hash(tuple(sorted(self.__cards)))

    def __eq__(self, other):
        if isinstance(other, CardHand):
            return sum(self) == sum(other)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, CardHand):
            return sum(self) < sum(other)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, CardHand):
            return sum(self) <= sum(other)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, CardHand):
            return sum(self) > sum(other)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, CardHand):
            return sum(self) >= sum(other)
        return NotImplemented

    @property
    def is_soft(self) -> bool:
        return any(card == self.__SOFT_ACE for card in self.__cards)

    def __migrate_to_hard_if_needed(self):
        if not self.is_soft:
            return
        for card in self.__cards:
            if isinstance(card, AceCard) and sum(self) > 21:
                card.set_hard()

    def add(self, *cards: Card):
        self.__cards.extend(AceCard() if isinstance(card, AceCard) else card.copy() for card in cards)
        self.__migrate_to_hard_if_needed()

    def clean(self):
        self.__cards.clear()


class GameAction(Enum):
    STAND = auto()
    HIT = auto()
    SPLIT = auto()
    DOUBLE_DOWN = auto()
    INSURANCE = auto()
    SURRENDER = auto()

    @classmethod
    def get_by_random(cls, *actions: 'GameAction') -> 'GameAction':
        return random.choice(actions)


class GameActionResult(Enum):
    WAIT_ACTION = auto()
    BLACKJACK = auto()
    WINS = auto()
    PUSH = auto()
    LOSS = auto()
    BUST = auto()


GameState = NamedTuple


class GameEnvironment(ABC):
    @property
    @abstractmethod
    def available_actions(self) -> tuple[GameAction, ...]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def play(self, game_action: GameAction) -> GameActionResult:
        pass

    @property
    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    @property
    @abstractmethod
    def state(self) -> GameState:
        pass
