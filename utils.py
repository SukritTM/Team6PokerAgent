from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from math import comb
from typing import List, Tuple, NamedTuple

import numpy as np
from numpy._typing import NDArray

from treys import Card

SUIT_INT_TO_SUIT_IDX = {
    1: 0,
    2: 1,
    4: 2,
    8: 3
}

def card_to_suit_idx(c):
    return SUIT_INT_TO_SUIT_IDX[Card.get_suit_int(c)]

def cards_to_arr(cards: List[Card]) -> NDArray[np.bool_]:
    arr = np.zeros((13, 4), dtype=bool)
    for c in cards:
        arr[Card.get_rank_int(c), card_to_suit_idx(c)] = True
    return arr

# `SuitState` is a tuple of rank indices (0-12) for a suit, representing the ranks of cards in that suit.
SuitState = Tuple[int, ...]

# `AllSuitState` is a tuple of four `SuitState` tuples, representing the state of all four suits.
# SuitState (Club, Diamond, Heart, Spade)
AllSuitState = Tuple[SuitState, SuitState, SuitState, SuitState]

# `RankState` is a tuple of 13 integers, each representing the count of cards of that rank.
RankState = Tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]

# `State` is a tuple of `AllSuitState` and `RankState`, representing the complete state of the cards.
State = Tuple[AllSuitState, RankState]

def cards_to_state(cards: List[Card]) -> State:
    """
    Convert a list of cards to a state representation that is invariant to permutations.
    The state is represented as a tuple of two tuples:
    - The first tuple contains the indices of the ranks of the cards for each suit. Normalized to sorted order.
    - The second tuple contains the counts of cards for each rank.
    :param cards:
    :return: suits, ranks
    """
    arr = cards_to_arr(cards)
    suits = tuple(sorted([tuple(np.flatnonzero(arr[:, col]).tolist()) for col in range(arr.shape[1])]))
    ranks = tuple(arr.sum(axis=1).tolist())
    # noinspection PyTypeChecker
    return suits, ranks

CARDS = tuple(Card.new(rank + suit) for rank in reversed(Card.STR_RANKS) for suit in Card.STR_SUITS)

@dataclass
class HandInfo:
    hand: Tuple[Card, ...]
    count: int = 1

@dataclass
class TwoHandInfo:
    hand1: Tuple[Card, ...]
    hand2: Tuple[Card, ...]
    count1: int = 1
    count2: int = 1


def compute_hands() -> dict[State, HandInfo]:
    state_to_hand_info: dict[State, HandInfo] = {}
    for hand in combinations(CARDS, 2):
        # Convert the hand to a state representation
        state = cards_to_state(hand)
        # Count the number of ways to achieve this hand
        if state not in state_to_hand_info:
            state_to_hand_info[state] = HandInfo(hand=hand)
        else:
            state_to_hand_info[state].count += 1
    return state_to_hand_info

def compute_2_hands(state_to_hand_info) -> dict[State, HandInfo]:
    state_to_2_hand_info: dict[State, HandInfo] = {}
    for handinf1 in state_to_hand_info.values():
        hand1 = handinf1.hand
        count1 = handinf1.count
        rem_cards = [c for c in CARDS if c not in hand1]
        for hand2 in combinations(rem_cards, 2):
            # Convert the hand to a state representation
            state = cards_to_state(hand1) + cards_to_state(hand2) + cards_to_state(hand1 + hand2)
            # Count the number of ways to achieve this hand
            if state not in state_to_2_hand_info:
                state_to_2_hand_info[state] = TwoHandInfo(hand1=hand1, hand2=hand2, count1=count1)
            else:
                state_to_2_hand_info[state].count2 += 1
    return state_to_2_hand_info


class HandLookup:
    def __init__(self):
        self.compute_score_lookup_tables()


def compute_score_lookup_tables():
    # Suited tables (straight flush, flush)
    num_suit_patterns = comb(13, 5)
    print(f"Generating {num_suit_patterns} suited patterns...")
    suit_combs = np.array(list(combinations(range(13-1,-1,-1),5)))
    assert suit_combs.shape == (num_suit_patterns, 5)
    suit_indices = np.arange(num_suit_patterns).reshape((num_suit_patterns, 1))
    suit_patterns = np.zeros((num_suit_patterns, 13), dtype=bool)
    suit_patterns[suit_indices, suit_combs] = True

    # Straight flush patterns
    num_straight_flush_patterns = 10
    print(f"Generating {num_straight_flush_patterns} straight patterns...")
    straight_flush_combs = np.array([np.arange(start, start + 5)[::-1] for start in range(8, -1, -1)])
    # Wheel straight (Ace to 5)
    wheel_comb = np.array([12, 3, 2, 1, 0]).reshape(1, 5)
    straight_flush_combs = np.concatenate((straight_flush_combs, wheel_comb), axis=0)
    assert straight_flush_combs.shape == (num_straight_flush_patterns, 5), f"Expected {num_straight_flush_patterns} straight patterns, got {straight_flush_combs.shape[0]}"
    straight_indices = np.arange(num_straight_flush_patterns).reshape((num_straight_flush_patterns, 1))
    straight_flush_patterns = np.zeros((num_straight_flush_patterns, 13), dtype=bool)
    straight_flush_patterns[straight_indices, straight_flush_combs] = True

    # Non-straight flushes (suited & ~straight)
    num_xstraight_flush_patterns = num_suit_patterns - num_straight_flush_patterns
    print(f"Computing {num_xstraight_flush_patterns} non-straight flush patterns...")
    # Use row-wise comparison to filter out straight patterns from suited patterns
    mask_suited_straight = np.any(np.all(suit_patterns[:, None, :] == straight_flush_patterns[None, :, :], axis=2), axis=1)
    xstraight_flush_patterns = suit_patterns[~mask_suited_straight]
    assert xstraight_flush_patterns.shape == (num_xstraight_flush_patterns, 13), f"Expected {num_xstraight_flush_patterns} non-straight flush patterns, got {xstraight_flush_patterns.shape[0]}"

    # All Rank patterns (Four of a Kind, Full House, Straight, Three of a Kind, Two Pair, High Card)
    num_rank_patterns = comb(13 + 5 - 1, 5) - 13 # 5 of a kind
    print(f"Generating {num_rank_patterns} rank patterns...")
    rank_combs = np.array(list(combinations_with_replacement(range(13-1,-1,-1), 5)))
    assert rank_combs.shape == (num_rank_patterns + 13, 5), f"Expected {num_rank_patterns + 13} rank combs, got {rank_combs.shape[0]}"

    four_of_a_kind_patterns = []
    full_house_patterns = []
    straight_patterns = []
    three_of_a_kind_patterns = []
    two_pair_patterns = []
    pair_patterns = []
    high_card_patterns = []


    for rank_comb in rank_combs:
        row = np.zeros((13,), dtype=int)
        values, counts = np.unique(rank_comb, return_counts=True)
        row[values] = counts

        counts = sorted(counts, reverse=True)
        # 5 of a kind, skip
        if counts == [5]:
            continue

        # Four of a Kind
        if counts == [4, 1]:
            four_of_a_kind_patterns.append(row)
            continue

        # Full House
        if counts == [3, 2]:
            full_house_patterns.append(row)
            continue

        # Check if it's a straight
        if counts == [1, 1, 1, 1, 1]:
            if all(rank_comb[i] - rank_comb[i+1] == 1 for i in range(4)):
                straight_patterns.append(row)
                continue
            # check for wheel
            elif set(rank_comb) == {12, 0, 1, 2, 3}:
                straight_patterns.append(row)
                continue

        # Three of a Kind
        if counts == [3, 1, 1]:
            three_of_a_kind_patterns.append(row)
            continue

        # Two Pair
        if counts == [2, 2, 1]:
            two_pair_patterns.append(row)
            continue

        # One Pair
        if counts == [2, 1, 1, 1]:
            pair_patterns.append(row)
            continue

        # High Card
        if counts == [1, 1, 1, 1, 1]:
            high_card_patterns.append(row)
            continue

        else:
            raise ValueError(f"Unexpected pattern: {row}")

    # Convert to numpy arrays
    four_of_a_kind_patterns = np.array(four_of_a_kind_patterns)
    full_house_patterns = np.array(full_house_patterns)
    straight_patterns = np.array(straight_patterns)
    three_of_a_kind_patterns = np.array(three_of_a_kind_patterns)
    two_pair_patterns = np.array(two_pair_patterns)
    pair_patterns = np.array(pair_patterns)
    high_card_patterns = np.array(high_card_patterns)

    print(f"Four of a Kind patterns: {len(four_of_a_kind_patterns)}")
    print(f"Full House patterns: {len(full_house_patterns)}")
    print(f"Straight patterns: {len(straight_patterns)}")
    print(f"Three of a Kind patterns: {len(three_of_a_kind_patterns)}")
    print(f"Two Pair patterns: {len(two_pair_patterns)}")
    print(f"Pair patterns: {len(pair_patterns)}")
    print(f"High Card patterns: {len(high_card_patterns)}")


    overall_rank = 0
    print(f"Overall, Straight Flush: ({overall_rank}, {len(straight_flush_patterns)})")
    overall_rank += len(straight_flush_patterns)
    print(f"Overall, Four of a Kind: ({overall_rank}, {len(four_of_a_kind_patterns)})")
    overall_rank += len(four_of_a_kind_patterns)
    print(f"Overall, Full House: ({overall_rank}, {len(full_house_patterns)})")
    overall_rank += len(full_house_patterns)
    print(f"Overall, Flush: ({overall_rank}, {len(xstraight_flush_patterns)})")
    overall_rank += len(xstraight_flush_patterns)
    print(f"Overall, Straight: ({overall_rank}, {len(straight_patterns)})")
    overall_rank += len(straight_patterns)
    print(f"Overall, Three of a Kind: ({overall_rank}, {len(three_of_a_kind_patterns)})")
    overall_rank += len(three_of_a_kind_patterns)
    print(f"Overall, Two Pair: ({overall_rank}, {len(two_pair_patterns)})")
    overall_rank += len(two_pair_patterns)
    print(f"Overall, Pair: ({overall_rank}, {len(pair_patterns)})")
    overall_rank += len(pair_patterns)
    print(f"Overall, High Card: ({overall_rank}, {len(high_card_patterns)})")
    print(f"Overall, Total: ({overall_rank}, {len(straight_flush_patterns) + len(xstraight_flush_patterns) + len(four_of_a_kind_patterns) + len(full_house_patterns) + len(straight_patterns) + len(three_of_a_kind_patterns) + len(two_pair_patterns) + len(pair_patterns) + len(high_card_patterns)})")

    # # Save the patterns to files
    # np.savez("hand_patterns.npz",
    #             straight_flush_patterns=straight_flush_patterns,
    #             xstraight_flush_patterns=xstraight_flush_patterns,
    #             four_of_a_kind_patterns=four_of_a_kind_patterns,
    #             full_house_patterns=full_house_patterns,
    #             straight_patterns=straight_patterns,
    #             three_of_a_kind_patterns=three_of_a_kind_patterns,
    #             two_pair_patterns=two_pair_patterns,
    #             pair_patterns=pair_patterns,
    #             high_card_patterns=high_card_patterns,
    #     )
    #
    # # Load the patterns from files
    # loaded_data = np.load("hand_patterns.npz")
    # straight_flush_patterns = loaded_data["straight_flush_patterns"]
    # xstraight_flush_patterns = loaded_data["xstraight_flush_patterns"]
    # four_of_a_kind_patterns = loaded_data["four_of_a_kind_patterns"]
    # full_house_patterns = loaded_data["full_house_patterns"]
    # straight_patterns = loaded_data["straight_patterns"]
    # three_of_a_kind_patterns = loaded_data["three_of_a_kind_patterns"]
    # two_pair_patterns = loaded_data["two_pair_patterns"]
    # pair_patterns = loaded_data["pair_patterns"]
    # high_card_patterns = loaded_data["high_card_patterns"]

    return {
        'suit': {
            'straight_flush_patterns': straight_flush_patterns,
            'xstraight_flush_patterns': xstraight_flush_patterns,
        },
        'rank': {
            'four_of_a_kind_patterns': four_of_a_kind_patterns,
            'full_house_patterns': full_house_patterns,
            'straight_patterns': straight_patterns,
            'three_of_a_kind_patterns': three_of_a_kind_patterns,
            'two_pair_patterns': two_pair_patterns,
            'pair_patterns': pair_patterns,
            'high_card_patterns': high_card_patterns,
        }
    }

def get_row_idx(arr: NDArray[np.bool_], pattern: NDArray[np.bool_]) -> int:
    """
    Get the row index of a pattern in an array.
    :param arr: The array to search in.
    :param pattern: The pattern to search for.
    :return: The row index of the pattern in the array.
    """
    matches = np.all(arr == pattern, axis=1)
    return np.argmax(matches) if matches.any() else -1

def get_hand_rank(hand: List[Card], lookup_tables) -> int:
    state = cards_to_state(hand)
    suits, ranks = state

    # Check for straight flush
    for suit in suits:
        if len(suit) < 5:
            continue
        pattern = np.zeros((13,), dtype=bool)
        pattern[list(suit)] = True
        row_idx = get_row_idx(lookup_tables['suit']['straight_flush_patterns'], pattern)
        print(f"Straight flush row index: {row_idx}")
        row_idx = get_row_idx(lookup_tables['suit']['xstraight_flush_patterns'], pattern)
        print(f"Non-straight flush row index: {row_idx}")

    # Check for four of a kind
    pattern = np.array(ranks)
    row_idx = get_row_idx(lookup_tables['rank']['four_of_a_kind_patterns'], pattern)
    print(f"Four of a kind row index: {row_idx}")
    # Check for full house
    row_idx = get_row_idx(lookup_tables['rank']['full_house_patterns'], pattern)
    print(f"Full house row index: {row_idx}")
    # Check for straight
    row_idx = get_row_idx(lookup_tables['rank']['straight_patterns'], pattern)
    print(f"Straight row index: {row_idx}")
    # Check for three of a kind
    row_idx = get_row_idx(lookup_tables['rank']['three_of_a_kind_patterns'], pattern)
    print(f"Three of a kind row index: {row_idx}")
    # Check for two pair
    row_idx = get_row_idx(lookup_tables['rank']['two_pair_patterns'], pattern)
    print(f"Two pair row index: {row_idx}")
    # Check for one pair
    row_idx = get_row_idx(lookup_tables['rank']['pair_patterns'], pattern)
    print(f"One pair row index: {row_idx}")
    # Check for high card
    row_idx = get_row_idx(lookup_tables['rank']['high_card_patterns'], pattern)
    print(f"High card row index: {row_idx}")













def main():
    # cards = [Card.new("As"), Card.new("Ah"), Card.new("Ks")]
    # (s_c, s_d, s_h, s_s), s_rank = cards_to_state(cards)
    # print(f"{Card.ints_to_pretty_str(cards)},c={s_c}, d={s_d}, h={s_h}, s={s_s}, rank={s_rank}")

    # hand_states = compute_hands()
    # for hand_info in hand_states.values():
    #     hand_str = Card.ints_to_pretty_str(hand_info.hand)
    #     print(f"Hand: {hand_str}, Count: {hand_info.count}")

    # state_to_hand_info = compute_hands()
    # state_to_2_hand_info = compute_2_hands(state_to_hand_info)
    #
    # rows = list(state_to_2_hand_info.values())
    # is_printed = False
    # for two_hand_info in rows:
    #     hand1 = two_hand_info.hand1
    #     hand2 = two_hand_info.hand2
    #     count1 = two_hand_info.count1
    #     count2 = two_hand_info.count2
    #
    #     if hand1 == (Card.new('As'), Card.new('Kh')):
    #         if not is_printed:
    #             print(f"Hand1={Card.ints_to_pretty_str(hand1)}, Count1={count1}")
    #             is_printed = True
    #         if Card.get_rank_int(hand2[0]) == 12 and Card.get_rank_int(hand2[1]) == 11:
    #             print(f"Hand2={Card.ints_to_pretty_str(hand2)}, Count2={count2}, Total={count1 * count2}")
    lookup_tables = compute_score_lookup_tables()
    get_hand_rank((Card.new('9s'), Card.new('3s'), Card.new('Qs'), Card.new('Js'), Card.new('Ts')), lookup_tables)

if __name__ == "__main__":
    main()

