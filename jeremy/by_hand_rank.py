import numpy as np
import itertools, random
from math import comb
from typing import Sequence, List

from treys.lookup    import LookupTable
from treys.card      import Card
from treys.evaluator import Evaluator
from treys.deck      import Deck

import matplotlib.pyplot as plt

def get_full_rank_distribution_np(
        known_cards: Sequence[int],
        total_cards: int = 7,
        exact_threshold: int = 2_200_000,
        sample_size: int = 200_000
) -> np.ndarray:
    """
    Returns a numpy array `P` of shape (7462,), where
      P[r-1] = Pr[best-5 hand rank == r].
    """
    lt        = LookupTable()
    ev        = Evaluator()
    hand      = tuple(known_cards)
    remaining = [c for c in Deck.GetFullDeck() if c not in hand]
    draws     = total_cards - len(hand)
    n_remain  = len(remaining)
    Ncomb     = comb(n_remain, draws)

    # 1) build the raw list of ranks
    if Ncomb <= exact_threshold:
        # exact enumeration
        ranks = np.fromiter(
            (ev.evaluate(hand, extra) for extra in itertools.combinations(remaining, draws)),
            dtype=np.int32,
        )
    else:
        # Monte-Carlo fallback
        ranks = np.fromiter(
            (ev.evaluate(hand, tuple(random.sample(remaining, draws)))
             for _ in range(sample_size)),
            dtype=np.int32,
        )

    # 2) build the length-7462 histogram in C
    counts = np.bincount(ranks-1, minlength=lt.MAX_HIGH_CARD)

    # 3) normalize to probabilities
    P = counts / counts.sum()

    return P

if __name__ == "__main__":
    # hands = []
    # for rank_idx1 in range(13-1, -1, -1):
    #     for rank_idx2 in range(rank_idx1, -1, -1):
    #         rank1 = Card.STR_RANKS[rank_idx1]
    #         rank2 = Card.STR_RANKS[rank_idx2]
    #         if rank1 == rank2:
    #             # Pair
    #             hand = [f"{rank1}s", f"{rank1}h"]
    #             hands.append(hand)
    #             # print(f"Pair: {hand}")
    #         else:
    #             # Suited
    #             hand = [f"{rank1}s", f"{rank2}s"]
    #             hands.append(hand)
    #             # print(f"Suited: {hand}")
    #             # Unsuited
    #             hand = [f"{rank1}s", f"{rank2}h"]
    #             hands.append(hand)
    #             # print(f"Unsuited: {hand}")
    #
    # FULL_P = np.zeros((len(hands), 7462), dtype=np.float32)
    # for i, hand in enumerate(hands):
    #     # if i == 1:
    #     #     break
    #     print(f"Hand {i}: {hand}")
    #     # Convert hand to binary representation
    #     hand = Card.hand_to_binary(hand)
    #     # Get the full rank distribution for the hand
    #     P = get_full_rank_distribution_np(hand)
    #     FULL_P[i] = P
    #
    #
    # # save the full rank distribution to a file
    # np.save("full_rank_distribution.npy", FULL_P)
    FULL_P = np.load("../../project/AI-Poker-Agent/full_rank_distribution.npy")

    lt = LookupTable()
    # Precompute thresholds and rank classes
    sorted_thrs = np.array(sorted(lt.MAX_TO_RANK_CLASS.keys()))
    sorted_cls = np.array([lt.MAX_TO_RANK_CLASS[t] for t in sorted_thrs])

    # Map rank classes to colors
    rank_class_colors = {
        0: "gold",        # Royal Flush
        1: "orange",      # Straight Flush
        2: "red",         # Four of a Kind
        3: "purple",      # Full House
        4: "blue",        # Flush
        5: "green",       # Straight
        6: "pink",        # Three of a Kind
        7: "cyan",        # Two Pair
        8: "brown",       # Pair
        9: "gray"         # High Card
    }

    # Determine rank class for each rank
    idx = np.clip(np.searchsorted(sorted_thrs, np.arange(1, lt.MAX_HIGH_CARD + 1), side="left"), 0, len(sorted_cls) - 1)
    cls = sorted_cls[idx]

    hand1P = np.cumsum(FULL_P[0])
    hand2P = np.cumsum(FULL_P[1])
    heatmap = np.zeros((len(hand1P), len(hand2P)), dtype=np.float32)
    # vectorize:
    heatmap = np.sign(np.arange(len(hand2P)) - np.arange(len(hand1P))[:, None]) * hand1P[:, None] * hand2P[None, :]

    # for i in range(len(hand1P)):
    #     for j in range(len(hand2P)):
    #         heatmap[i, j] = np.sign(j-i)* hand1P[i] * hand2P[j]

    # Plot heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap of Hand Rank Probabilities")
    plt.xlabel("Hand 2 Rank")
    plt.ylabel("Hand 1 Rank")
    plt.show()



    # Plot with color coding
    x = np.arange(1, 7462 + 1)
    y = np.cumsum(FULL_P[0])  # Use the first hand's distribution for plotting

    colors = [rank_class_colors[c] for c in cls]

    plt.scatter(x, y, c=colors, s=10, label="Rank Classes")
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.title("Probability Distribution by Rank Class")
    plt.legend()
    plt.show()


    # holes = Card.hand_to_binary(["As","Ad"])
    # P     = get_full_rank_distribution_np(holes)
    #
    # lt        = LookupTable()
    # # Precompute thresholds and names once:
    # sorted_thrs = np.array(sorted(lt.MAX_TO_RANK_CLASS.keys()))
    # sorted_cls  = np.array([lt.MAX_TO_RANK_CLASS[t] for t in sorted_thrs])
    # names       = np.array([lt.RANK_CLASS_TO_STRING[i] for i in range(len(lt.RANK_CLASS_TO_STRING))])
    #
    # # 4) bucket into 10 categories in C
    # # For each rank r=1..7462, find which threshold it falls under:
    # idx = np.clip(np.searchsorted(sorted_thrs, np.arange(1, lt.MAX_HIGH_CARD+1), side="left"), 0, len(sorted_cls) - 1)
    # # idx[i] is index in sorted_thrs, so class = sorted_cls[idx[i]]
    # cls = sorted_cls[idx]
    # # now sum P over each class
    # cat_probs = np.bincount(cls, weights=P, minlength=len(names))
    #
    # # Print stats for P:
    # # min, max, mean, std
    # print(f"min: {P.min():.6%}, max: {P.max():.6%}, mean: {P.mean():.6%}, std: {P.std():.6%}")
    #
    # # 5) print them
    # for i, cat in enumerate(names):
    #     print(f"{cat:15s}: {cat_probs[i]:.6%}")
    # print(f"Total probability = {cat_probs.sum():.6%}")