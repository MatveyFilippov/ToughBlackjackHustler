import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple


class Card:
    def __init__(self, rank: int):
        if not 2 <= rank <= 10:
            raise ValueError(f"Invalid rank. Available only [2-10]")
        self._rank = rank

    @property
    def rank(self) -> int:
        return self._rank

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
        return hash(self._rank)

    def __str__(self):
        return f"Card(rank='{self._rank}')"

    def __repr__(self):
        return f"Card(rank='{self._rank}')"


class AceCard(Card):
    def __init__(self, is_soft: bool | None = True):
        super().__init__(7)
        self._rank = 11 if is_soft else 1

    def set_hard(self):
        self._rank = 1


class CardDeck:
    __DEFAULT_DECK: list[Card] = [
        Card(__rank) for __rank in range(2, 11) for _ in range(4)  # Numbered Cards (2–10)
    ] + [
        Card(10) for _ in range(3*4)  # Face Cards (J, Q, K)
    ] + [
        AceCard() for _ in range(4)  # Ace Cards
    ]

    def __init__(self, qty: int | None = 1):
        if qty < 1:
            raise ValueError("QTY of decks must be greater than 0")
        self.__deck = self.__DEFAULT_DECK * qty
        random.shuffle(self.__deck)
        self.__init_decks_qty = qty
        self.__min_cards_qty = self.__len__() * 0.25

    def __len__(self):
        return len(self.__deck)

    def __hash__(self):
        return hash(sum(card.rank for card in self.__deck))

    def draw(self) -> Card:
        if self.__len__() == 0:
            raise IndexError("All cards in the deck have already been used.")
        return self.__deck.pop()

    @property
    def init_decks_qty(self) -> int:
        return self.__init_decks_qty

    @property
    def is_playable(self) -> bool:
        return self.__len__() >= self.__min_cards_qty

    @property
    def remaining_cards(self) -> list[Card]:
        return self.__deck.copy()

    @classmethod
    def of(cls, init_decks_qty: int, remaining_cards: list[Card]) -> 'CardDeck':
        deck = cls(init_decks_qty)
        deck.__deck = remaining_cards.copy()
        random.shuffle(deck.__deck)
        return deck


class CardHand:
    __SOFT_ACE = AceCard(is_soft=True)

    def __init__(self, *cards: Card):
        self.__cards = []
        self.add(*cards)

    def __len__(self):
        return len(self.__cards)

    def __getitem__(self, index) -> Card:
        return self.__cards[index]

    def __iter__(self):
        return (card for card in self.__cards)

    def __hash__(self):
        return hash(sum(self))

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
            if isinstance(card, AceCard):
                card.set_hard()
                if sum(self) <= 21:
                    break

    def add(self, *cards: Card):
        self.__cards.extend(cards)
        self.__migrate_to_hard_if_needed()


class UserAction(Enum):
    STAND = 0
    HIT = 1
    __actions: list['UserAction'] = None

    @classmethod
    def get_by_random(cls) -> 'UserAction':
        if not cls.__actions:
            cls.__actions = list(UserAction)
        return random.choice(cls.__actions)


class AgentReward(float):
    NEUTRAL = 0.0


GameState = NamedTuple


class GameEnvironment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def play(self, user_action: UserAction) -> AgentReward:
        pass

    @property
    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    @property
    @abstractmethod
    def state(self) -> GameState:
        pass
