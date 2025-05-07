import enum
import os
import random
from itertools import combinations, combinations_with_replacement
from math import comb
from typing import List

import numpy as np
from matplotlib.style.core import available
from numpy._typing import NDArray

from Team6PokerAgent.utils import cards_to_state, SuitState, RankState, CARDS
from treys import Card


class HandCat(enum.IntEnum):
    STRAIGHT_FLUSH = 0
    FOUR_OF_A_KIND = 1
    FULL_HOUSE = 2
    FLUSH = 3
    STRAIGHT = 4
    THREE_OF_A_KIND = 5
    TWO_PAIR = 6
    PAIR = 7
    HIGH_CARD = 8

    def __str__(self):
        return self.name.replace("_", " ").title()

    def __repr__(self):
        return self.__str__()

def compute_score_lookup_tables():
    # Suited tables (straight flush, flush)
    num_suit_patterns = comb(13, 5)
    # print(f"Generating {num_suit_patterns} suited patterns...")
    suit_combs = np.array(list(combinations(range(13-1,-1,-1),5)))
    assert suit_combs.shape == (num_suit_patterns, 5)
    suit_indices = np.arange(num_suit_patterns).reshape((num_suit_patterns, 1))
    suit_patterns = np.zeros((num_suit_patterns, 13), dtype=bool)
    suit_patterns[suit_indices, suit_combs] = True

    # Straight flush patterns
    num_straight_flush_patterns = 10
    # print(f"Generating {num_straight_flush_patterns} straight patterns...")
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
    # print(f"Computing {num_xstraight_flush_patterns} non-straight flush patterns...")
    # Use row-wise comparison to filter out straight patterns from suited patterns
    mask_suited_straight = np.any(np.all(suit_patterns[:, None, :] == straight_flush_patterns[None, :, :], axis=2), axis=1)
    xstraight_flush_patterns = suit_patterns[~mask_suited_straight]
    assert xstraight_flush_patterns.shape == (num_xstraight_flush_patterns, 13), f"Expected {num_xstraight_flush_patterns} non-straight flush patterns, got {xstraight_flush_patterns.shape[0]}"

    # All Rank patterns (Four of a Kind, Full House, Straight, Three of a Kind, Two Pair, High Card)
    num_rank_patterns = comb(13 + 5 - 1, 5) - 13 # 5 of a kind
    # print(f"Generating {num_rank_patterns} rank patterns...")
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

    patterns = {
        HandCat.STRAIGHT_FLUSH: straight_flush_patterns,
        HandCat.FOUR_OF_A_KIND: four_of_a_kind_patterns,
        HandCat.FULL_HOUSE: full_house_patterns,
        HandCat.FLUSH: xstraight_flush_patterns,
        HandCat.STRAIGHT: straight_patterns,
        HandCat.THREE_OF_A_KIND: three_of_a_kind_patterns,
        HandCat.TWO_PAIR: two_pair_patterns,
        HandCat.PAIR: pair_patterns,
        HandCat.HIGH_CARD: high_card_patterns,
    }

    labels = {}
    start = 0
    for cat, pattern in patterns.items():
        cat_label = np.full(pattern.shape[0], cat)
        cat_rank = np.arange(start, start + pattern.shape[0])
        labels[cat] = np.concatenate((cat_label[:, None], cat_rank[:, None]), axis=1)
        start += pattern.shape[0]


    suit_patterns = np.concatenate((patterns[HandCat.STRAIGHT_FLUSH], patterns[HandCat.FLUSH]), axis=0)
    suit_labels = np.concatenate((labels[HandCat.STRAIGHT_FLUSH], labels[HandCat.FLUSH]), axis=0)
    rank_patterns = np.concatenate((patterns[HandCat.FOUR_OF_A_KIND], patterns[HandCat.FULL_HOUSE], patterns[HandCat.STRAIGHT], patterns[HandCat.THREE_OF_A_KIND], patterns[HandCat.TWO_PAIR], patterns[HandCat.PAIR], patterns[HandCat.HIGH_CARD]), axis=0)
    rank_labels = np.concatenate((labels[HandCat.FOUR_OF_A_KIND], labels[HandCat.FULL_HOUSE], labels[HandCat.STRAIGHT], labels[HandCat.THREE_OF_A_KIND], labels[HandCat.TWO_PAIR], labels[HandCat.PAIR], labels[HandCat.HIGH_CARD]), axis=0)

    rank_index_to_suit_index_map = {}
    for suit_index, suit_pattern in enumerate(suit_patterns):
        rank_index = get_row_idx(rank_patterns, suit_pattern)
        if rank_index != -1:
            rank_index_to_suit_index_map[rank_index] = suit_index

    return {
        'suit_patterns': suit_patterns,
        'suit_labels': suit_labels,
        'rank_patterns': rank_patterns,
        'rank_labels': rank_labels,
        'rank_index_to_suit_index_map': rank_index_to_suit_index_map,

    }


def get_row_idx(arr: NDArray[np.bool_], pattern: NDArray[np.bool_]) -> int:
    # matches = np.all(arr == pattern, axis=1)
    matches = np.all(arr <= pattern, axis=1)
    return np.argmax(matches) if matches.any() else -1


class HandLookup:

    def __init__(self):
        self.lookup_tables = compute_score_lookup_tables()


    def match_suits(self, suits: List[SuitState]):
        for suit in suits:
            if len(suit) < 5:
                continue
            pattern = np.zeros((13,), dtype=bool)
            pattern[list(suit)] = True
            # Check for straight flush

            row_idx = get_row_idx(self.lookup_tables['suit_patterns'], pattern)
            if row_idx != -1:
                return self.lookup_tables['suit_labels'][row_idx]
            # # Check for non-straight flush
            # row_idx = get_row_idx(self.lookup_tables['suit'][HandCat.FLUSH], pattern)
            # if row_idx != -1:
            #     cat = HandCat.FLUSH
            #     return HandCat.FLUSH, row_idx
        return None, -1

    def match_ranks(self, ranks: RankState):
        pattern = np.array(ranks)
        row_idx = get_row_idx(self.lookup_tables['rank_patterns'], pattern)
        if row_idx != -1:
            return self.lookup_tables['rank_labels'][row_idx]
        return None, -1

    def get_hand_rank(self, hand: List[Card]):
        state = cards_to_state(hand)
        suits, ranks = state
        cat = None
        rank = None
        for match_cat, match_rank in [self.match_suits(suits), self.match_ranks(ranks)]:
            if match_cat is None:
                continue
            if cat is None or match_rank < rank:
                cat = match_cat
                rank = match_rank

        return HandCat(cat), int(rank)



    def get_hand_rank_probs(self, hand: List[Card], cards_remaining: int = 52, draws: int = 7):
        suits_state, ranks_state = cards_to_state(hand)
        ways_to_hit = np.zeros((7462,))

        print("Suit")
        for suit in suits_state:
            pattern = np.zeros((13,), dtype=bool)
            pattern[list(suit)] = True
            needed = self.lookup_tables['suit_patterns'] & ~pattern
            print(f"needed: {needed.shape}")

            keep = np.ones((needed.shape[0]), dtype=bool)
            for index, row in enumerate(needed):
                if not keep[index]:
                    continue
                mask = needed >= row
                mask = np.all(mask, axis=1)
                keep[index:][mask[index:]] = False

                cards_needed = np.sum(row)
                if cards_needed > draws:
                    continue
                cat, rank = self.lookup_tables["suit_labels"][index]
                # print(f"cmb={cmb}, n={n}, cmb/n={cmb/n}")
                ways_to_hit[rank] += comb(cards_remaining - cards_needed, draws - cards_needed)
        # Rank
        print("Rank")
        pattern = np.array(ranks_state)
        needed = self.lookup_tables['rank_patterns'] - pattern
        needed[needed < 0] = 0
        available = 4 - pattern
        print(f"needed: {needed.shape}")

        keep = np.ones((needed.shape[0]), dtype=bool)
        for index, row in enumerate(needed):
            if not keep[index]:
                continue
            mask = needed >= row
            mask = np.all(mask, axis=1)
            keep[index:][mask[index:]] = False

            cat, rank = self.lookup_tables["rank_labels"][index]
            # print(f"cat: {cat}, rank: {rank}, index: {index}, row: {row}")
            cards_needed = np.sum(row)
            if cards_needed > draws:
                continue

            cn = 1
            # remaining = cards_remaining
            for i in range(len(row)):
                if row[i] > 0:
                    cn *= comb(available[i], row[i])
                    # remaining -= available[i]
            cmb = comb(cards_remaining - cards_needed, draws - cards_needed)
            # print(f"cat={cat}, cmb={cmb}, cn={cn}, cmb*cn={cmb*cn}, cmb*cn/n={cmb*cn/n}")
            ways_to_hit[rank] += cmb * cn

            if index in self.lookup_tables["rank_index_to_suit_index_map"]:
                suit_index = self.lookup_tables["rank_index_to_suit_index_map"][index]
                suit_rank = self.lookup_tables["suit_labels"][suit_index][1]
                ways_to_hit[rank] -= ways_to_hit[suit_rank]
                # n -= ways_to_hit[suit_rank]


        # Normalize
        ways_to_hit = ways_to_hit / ways_to_hit.sum()

        # way_to_hit_cum = np.cumsum(ways_to_hit)
        cats = {}
        labels = np.concatenate((self.lookup_tables['suit_labels'], self.lookup_tables['rank_labels']), axis=0)
        for cat in HandCat:
            ranks = labels[labels[:, 0] == cat][:, 1]
            total = ways_to_hit[ranks].sum()
            print(f"cat: {cat}, total: {total}")
        # for index, row in enumerate(ways_to_hit):
        #     if row >= 1e-6:
        #         pass
        #         print(f"{index}: {row:%}, {way_to_hit_cum[index]:%}")
        # return ways_to_hit

        expected_rank = 0
        for rank, prob in enumerate(ways_to_hit):
            expected_rank += rank * prob
        print(f"Expected rank: {expected_rank}")








    def load_or_create_hand_lookup(self):
        if not os.path.exists("hand_patterns.npz"):
            # Create the patterns if they don't exist
            self.create_hand_lookup()
        else:
            # Load the patterns from files
            loaded_data = np.load("hand_patterns.npz")
            straight_flush_patterns = loaded_data["straight_flush_patterns"]
            xstraight_flush_patterns = loaded_data["xstraight_flush_patterns"]
            four_of_a_kind_patterns = loaded_data["four_of_a_kind_patterns"]
            full_house_patterns = loaded_data["full_house_patterns"]
            straight_patterns = loaded_data["straight_patterns"]
            three_of_a_kind_patterns = loaded_data["three_of_a_kind_patterns"]
            two_pair_patterns = loaded_data["two_pair_patterns"]
            pair_patterns = loaded_data["pair_patterns"]
            high_card_patterns = loaded_data["high_card_patterns"]


if __name__ == "__main__":
    hand_lookup = HandLookup()

    for i in range(10):
        hand = random.sample(CARDS, 7)
        print(Card.ints_to_pretty_str(hand), hand_lookup.get_hand_rank(hand))

    hand_lookup.get_hand_rank_probs([], cards_remaining=50, draws=5)
        #['As', 'Ah', 'Kd', 'Kh', 'Qd', 'Qh', 'Js'])
