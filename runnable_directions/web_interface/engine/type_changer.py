from environment.base import Card, AceCard


__CARDS: dict[str, Card] = {
    "J": Card(10),
    "Q": Card(10),
    "K": Card(10),
    "A": AceCard(),
}
for numbered_rank in range(2, 11):
    __CARDS[str(numbered_rank)] = Card(numbered_rank)


def is_cardable(_str: str, raise_if_not=False) -> bool:
    result = _str.strip().upper() in __CARDS
    if raise_if_not and not result:
        raise ValueError(f"Invalid card name: '{_str}'")
    return result


def str_to_card(_str: str) -> Card:
    try:
        return __CARDS[_str.strip().upper()]
    except KeyError:
        raise ValueError(f"Invalid card name: '{_str}'")
