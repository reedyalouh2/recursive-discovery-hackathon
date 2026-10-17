"""Seeded pseudo-random numbers that replay identically across Python versions.

Python's `random` module does not promise stable streams between releases, so
saved runs use SplitMix64 (Steele, Lea and Flood, OOPSLA 2014) instead.
"""

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def _mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        if type(seed) is not int or not 0 <= seed <= MASK64:
            raise ValueError("seed must be an integer from 0 to 2**64 - 1.")
        self.seed = seed
        self._state = seed

    def next_u64(self):
        self._state = (self._state + GOLDEN) & MASK64
        return _mix(self._state)

    def randbelow(self, n):
        """Uniform integer in [0, n), using rejection to avoid modulo bias."""
        if type(n) is not int or not 1 <= n <= MASK64 + 1:
            raise ValueError("n must be an integer from 1 to 2**64.")
        limit = (MASK64 + 1) - (MASK64 + 1) % n
        while True:
            value = self.next_u64()
            if value < limit:
                return value % n

    def random(self):
        """Uniform float in [0, 1) with 53 random bits."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def child(self, stream):
        """Independent generator for numbered sub-stream `stream` (e.g. a restart)."""
        if type(stream) is not int or stream < 0:
            raise ValueError("stream must be a non-negative integer.")
        return Rng(_mix((self.seed + GOLDEN * (stream + 1)) & MASK64))
