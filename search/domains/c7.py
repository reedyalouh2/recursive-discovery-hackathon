"""Strong powers of the 7-cycle: vertex encoding, adjacency and certificate export.

Vertex v is the base-7 number of its coordinate tuple, first coordinate most
significant, so index order equals lexicographic tuple order. Two distinct
vertices are adjacent when every coordinate differs by 0 or +-1 modulo 7.
"""

from array import array
import itertools
import json

PROBLEM = "c7_strong_power"
# 7**5 = 16807 vertices. Larger powers need a faster engine and exceed the
# starter checker's 2000-vertex limit for good constructions.
MAX_SEARCH_DIMENSION = 5


class C7StrongPower:
    def __init__(self, dimension):
        if type(dimension) is not int or not 1 <= dimension <= MAX_SEARCH_DIMENSION:
            raise ValueError(f"dimension must be an integer from 1 to {MAX_SEARCH_DIMENSION}.")
        self.dimension = dimension
        self.num_vertices = 7 ** dimension
        self._coords = list(itertools.product(range(7), repeat=dimension))
        self._place = [7 ** (dimension - 1 - i) for i in range(dimension)]
        self._neighbors = [None] * self.num_vertices

    def coords(self, v):
        return self._coords[v]

    def index(self, coords):
        value = 0
        for c in coords:
            value = value * 7 + c
        return value

    def neighbors(self, v):
        """The 3**dimension - 1 neighbors of v, computed once and cached."""
        cached = self._neighbors[v]
        if cached is None:
            offsets = [0]
            for c, place in zip(self._coords[v], self._place):
                down = ((c - 1) % 7 - c) * place
                up = ((c + 1) % 7 - c) * place
                offsets = [s + t for s in offsets for t in (0, down, up)]
            # offsets[0] is the all-zero step, i.e. v itself.
            cached = array("i", [v + s for s in offsets[1:]])
            self._neighbors[v] = cached
        return cached

    def adjacent(self, u, w):
        if u == w:
            return False
        for a, b in zip(self._coords[u], self._coords[w]):
            if (a - b) % 7 in (2, 3, 4, 5):
                return False
        return True

    def is_independent(self, vertices):
        """Search-side sanity check only; the official verdict comes from verification.c7."""
        vertices = tuple(vertices)
        chosen = bytearray(self.num_vertices)
        for v in vertices:
            if type(v) is not int or not 0 <= v < self.num_vertices or chosen[v]:
                return False
            chosen[v] = 1
        return not any(chosen[w] for v in vertices for w in self.neighbors(v))

    def certificate(self, vertices):
        """Candidate format v1 from COORDINATION.md, vertices in lexicographic order."""
        return {
            "format_version": 1,
            "problem": PROBLEM,
            "dimension": self.dimension,
            "vertices": [list(self._coords[v]) for v in sorted(vertices)],
        }

    def vertices_from_certificate(self, certificate):
        """Vertex indices of a certificate that the official checker has accepted."""
        if certificate.get("dimension") != self.dimension:
            raise ValueError(
                f"Certificate dimension {certificate.get('dimension')} does not match {self.dimension}.")
        return sorted(self.index(row) for row in certificate["vertices"])


def certificate_text(certificate):
    """Deterministic JSON text with one vertex per line."""
    rows = ",\n".join("    " + json.dumps(row) for row in certificate["vertices"])
    vertices = "[\n" + rows + "\n  ]" if rows else "[]"
    return (
        "{\n"
        f'  "format_version": {json.dumps(certificate["format_version"])},\n'
        f'  "problem": {json.dumps(certificate["problem"])},\n'
        f'  "dimension": {json.dumps(certificate["dimension"])},\n'
        f'  "vertices": {vertices}\n'
        "}\n"
    )
