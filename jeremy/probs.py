import math
from itertools import combinations
from math import comb
from typing import Sequence, Dict, Tuple

import numpy as np

from treys import Card

def rank_histogram_dp(known_ranks: Sequence[int], draws: int) -> Dict[Tuple[int, ...], float]:
    """
    DP to compute probability of each final rank histogram H (length-13 tuple),
    given known rank counts and number of draws.
    """
    # initial known counts per rank
    h = [0]*13
    for r in known_ranks:
        h[r] += 1
    # total unseen cards
    total_unknown = 52 - sum(h)
    # DP over ranks to count ways to draw δ_x cards of each rank x
    ways_by_hist = {}  # key: tuple of δ_x, value: ways
    def recurse(idx: int, remaining: int, current: list[int], ways: int):
        if idx == 13:
            if remaining == 0:
                ways_by_hist[tuple(current)] = ways
            return
        max_draw = min(4 - h[idx], remaining)
        for cnt in range(max_draw + 1):
            recurse(idx+1, remaining-cnt, current + [cnt],
                    ways * comb(4 - h[idx], cnt))
    recurse(0, draws, [], 1)
    # normalize to probabilities for final H = h + δ
    total_combos = comb(total_unknown, draws)
    prob_H = {}
    for delta, ways in ways_by_hist.items():
        H = tuple(h[i] + delta[i] for i in range(13))
        prob_H[H] = ways / total_combos
    return prob_H

def classify_rank_hist(H: Tuple[int, ...]) -> str:
    """
    Classify best 5-card hand based only on rank histogram H, ignoring flush.
    Possible returns: "Straight", "Four of a Kind", "Full House",
    "Three of a Kind", "Two Pair", "Pair", "High Card"
    """
    # sorted counts
    counts = sorted(H, reverse=True)
    if counts[0] >= 4:
        return "Four of a Kind"
    if counts[0] >= 3 and counts[1] >= 2:
        return "Full House"
    # check straight (including wheel)
    # ranks 0..12 correspond to 2..A
    for i in range(9):
        if all(H[i+j] >= 1 for j in range(5)):
            return "Straight"
    # Ace-low straight
    if all(H[i] >= 1 for i in [12, 0, 1, 2, 3]):
        return "Straight"
    if counts[0] >= 3:
        return "Three of a Kind"
    pairs = sum(1 for c in H if c >= 2)
    if pairs >= 2:
        return "Two Pair"
    if pairs == 1:
        return "Pair"

    return "High Card"

def compute_flush_probs(known_cards: Sequence[int], draws: int) -> Dict[str, float]:
    """
    Compute P(flush), P(straight flush incl. royal), and P(royal flush)
    via hypergeometric formulas over suits and straight-flush patterns.
    """
    total_unknown = 52 - len(known_cards)
    total_combos = comb(total_unknown, draws)

    # count known suit+rank
    suit_map = {1:0, 2:1, 4:2, 8:3}  # Card.get_suit_int -> index
    known_suits = [0]*4
    # key by suit_index, not suit-int
    known_by_suit_rank = {
        (s_idx, r): False
        for s_idx in suit_map.values()
        for r in range(13)
    }
    for c in known_cards:
        s_int = Card.get_suit_int(c)
        s_idx = suit_map[s_int]
        r = Card.get_rank_int(c)
        known_suits[s_idx] += 1
        known_by_suit_rank[(s_idx, r)] = True

    # 5-rank straight patterns (rank indices)
    sf_patterns = [tuple(range(i, i+5)) for i in range(8, -1, -1)]
    sf_patterns.append((12, 0, 1, 2, 3))  # wheel

    P_flush = 0
    P_sf = 0
    P_rf = 0

    for suit_val, h_s in enumerate(known_suits):
        # flush (at least 5 of suit)
        rem_suit = 13 - h_s
        for k in range(max(5-h_s, 0), draws+1):
            P_flush += comb(rem_suit, k) * comb(total_unknown - rem_suit, draws - k)
        # straight flush patterns
        for pat in sf_patterns:
            # count how many of these pattern cards already known in this suit
            m = sum(1 for r in pat if known_by_suit_rank[(suit_val, r)])
            needed = 5 - m
            if needed <= draws:
                # you must remove the *needed* cards from the unknown deck, not the known ones
                P_sf += comb(total_unknown - needed, draws - needed)
                if set(pat) == {12,11,10,9,8}:  # royal‐flush pattern
                    P_rf += comb(total_unknown - needed, draws - needed)
    # normalize
    # P_flush /= total_combos
    # P_sf    /= total_combos
    # P_rf    /= total_combos

    return {
        "Flush": P_flush,
        "Straight Flush": P_sf,
        "Royal Flush": P_rf,
        "total_combos": total_combos,
    }

def hand_rank_probabilities_dp(known_cards: Sequence[int], total_cards: int = 7) -> Dict[str, float]:
    """
    Combine rank-histogram DP with hypergeom flush formulas to get
    closed-form probabilities for all 10 poker categories in a best-of-5
    from total_cards pool.
    """
    # derive known rank list and draws
    known_ranks = [Card.get_rank_int(c) for c in known_cards]
    draws = total_cards - len(known_cards)

    # compute flush-based probabilities
    flush_probs = compute_flush_probs(known_cards, draws)
    P_flush     = flush_probs["Flush"]
    P_sf_total  = flush_probs["Straight Flush"]
    P_rf        = flush_probs["Royal Flush"]

    # DP histogram probabilities
    hist_probs = rank_histogram_dp(known_ranks, draws)
    print(f"hist_probs info: {len(hist_probs)}")

    # sum rank-based categories from DP
    cat_sums = {cat: 0.0 for cat in
                ["Straight","Four of a Kind","Full House",
                 "Three of a Kind","Two Pair","Pair","High Card"]}
    for H, p in hist_probs.items():
        cat = classify_rank_hist(H)
        cat_sums[cat] += p

    # plain straight excludes SF
    P_plain_straight = cat_sums["Straight"] - P_sf_total
    # print(f"cat_sums[Straight]: {cat_sums['Straight']}, P_sf_total: {P_sf_total}, P_plain_straight: {P_plain_straight}")

    # flush excluding SF
    P_plain_flush = P_flush - P_sf_total

    # collect final probs
    probs = {
        "Royal Flush":  P_rf,
        "Straight Flush": P_sf_total - P_rf,
        "Four of a Kind": cat_sums["Four of a Kind"],
        "Full House":     cat_sums["Full House"],
        "Flush":         P_plain_flush,
        "Straight":      P_plain_straight,
        "Three of a Kind":cat_sums["Three of a Kind"],
        "Two Pair":       cat_sums["Two Pair"],
        "Pair":           cat_sums["Pair"],
        "High Card":      cat_sums["High Card"],
    }
    # normalize to sum=1
    total = sum(probs.values())
    for k in probs:
        probs[k] /= total

    return probs


# Example usage:
if __name__ == "__main__":
    # Known hole: Ace♠, King♦
    holes = Card.hand_to_binary(["As","Ad"])
    probs = hand_rank_probabilities_dp(holes, total_cards=7)
    # derive known rank list and draws
    known_ranks = [Card.get_rank_int(c) for c in holes]
    draws = 7 - len(holes)

    # compute flush-based counts
    flush_probs = compute_flush_probs(holes, draws)
    P_flush      = flush_probs["Flush"]
    P_sf_total  = flush_probs["Straight Flush"]
    P_rf        = flush_probs["Royal Flush"]
    # print flush counts
    print("Flush probabilities:")
    print(f"Flush: {P_flush:}")
    print(f"Straight Flush: {P_sf_total:}")
    print(f"Royal Flush: {P_rf:}")
    #
    #
    # for cat, p in probs.items():
    #     print(f"{cat:15s}: {p:.6%}")
    #
    # total = sum(probs.values())
    # print(f"Total probability = {total:.6%}")


