# EVOLVE-BLOCK-START
"""Randomized greedy independent sets in strong powers of the 7-cycle."""

import itertools
import random

SEED = 0
TRIALS = 1


def adjacent(u, v):
    """Distinct vertices conflict when every coordinate differs by 0 or +-1 mod 7."""
    return all((a - b) % 7 in (0, 1, 6) for a, b in zip(u, v))


def construct(dimension):
    """Return an independent set as a list of coordinate tuples."""
    vertices = list(itertools.product(range(7), repeat=dimension))
    rng = random.Random(SEED)
    best = []
    for _ in range(TRIALS):
        order = vertices[:]
        rng.shuffle(order)
        chosen = []
        for v in order:
            if not any(adjacent(v, u) for u in chosen):
                chosen.append(v)
        if len(chosen) > len(best):
            best = chosen
    return best
# EVOLVE-BLOCK-END


def main():
    """Fixed harness: write the construction as candidate format v1 data."""
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    certificate = {
        "format_version": 1,
        "problem": "c7_strong_power",
        "dimension": args.dimension,
        "vertices": [list(v) for v in construct(args.dimension)],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(certificate, stream)


if __name__ == "__main__":
    main()
