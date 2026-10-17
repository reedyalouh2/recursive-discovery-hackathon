"""Check finite independent sets in a strong power of the seven-cycle.

Usage: python3 -m verification.c7 examples/c7_calibration.json

Input is JSON data only: format_version=1, problem="c7_strong_power",
dimension (integer 1..16), and vertices (at most 2000 integer vectors).
Files are limited to 2 MiB. Values outside these starter limits are unsupported.
Every pair must differ by 2, 3, 4, or 5 modulo 7 in at least one coordinate.
Checks use exact integers; the optional decimal bound is only an approximation.
Acceptance proves this finite certificate's validity, not a research record.
"""

import json
from pathlib import Path
import sys

MAX_DIMENSION = 16
MAX_VERTICES = 2000
MAX_FILE_BYTES = 2 * 1024 * 1024


def _failure(reason, *, size=None, dimension=None):
    return {
        "valid": False,
        "size": size,
        "dimension": dimension,
        "capacity_lower_bound_exact": None,
        "reason": reason,
    }


def validate(certificate):
    """Validate a decoded certificate without executing candidate code."""
    if not isinstance(certificate, dict):
        return _failure("Certificate must be a JSON object.")
    if type(certificate.get("format_version")) is not int or certificate["format_version"] != 1:
        return _failure("Unsupported format_version; expected integer 1.")
    if certificate.get("problem") != "c7_strong_power":
        return _failure("Unsupported problem; expected c7_strong_power.")
    dimension = certificate.get("dimension")
    if type(dimension) is not int or dimension < 1:
        return _failure("dimension must be a positive integer.")
    if dimension > MAX_DIMENSION:
        return _failure(f"Unsupported dimension: maximum is {MAX_DIMENSION}.", dimension=dimension)
    vertices = certificate.get("vertices")
    if not isinstance(vertices, list):
        return _failure("vertices must be a JSON array.", dimension=dimension)
    size = len(vertices)
    if size > MAX_VERTICES:
        return _failure(f"Unsupported vertex count: maximum is {MAX_VERTICES}.", size=size, dimension=dimension)

    seen = set()
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, list) or len(vertex) != dimension:
            return _failure(f"Vertex {index} must be an array of length {dimension}.", size=size, dimension=dimension)
        if any(type(value) is not int or not 0 <= value <= 6 for value in vertex):
            return _failure(f"Vertex {index} must contain only integers from 0 through 6.", size=size, dimension=dimension)
        row = tuple(vertex)
        if row in seen:
            return _failure(f"Duplicate vertex at index {index}.", size=size, dimension=dimension)
        seen.add(row)

    for left in range(size):
        for right in range(left + 1, size):
            if not any((a - b) % 7 in (2, 3, 4, 5)
                       for a, b in zip(vertices[left], vertices[right])):
                return _failure(
                    f"Vertices {left} and {right} are adjacent in the strong product.",
                    size=size, dimension=dimension,
                )
    return {
        "valid": True,
        "size": size,
        "dimension": dimension,
        "capacity_lower_bound_exact": f"{size}^(1/{dimension})",
        "capacity_lower_bound_approx": size ** (1 / dimension),
        "reason": "Valid independent-set certificate; no novelty or record claim assessed.",
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def check_file(path):
    """Read one bounded regular file and return a structured result."""
    try:
        source = Path(path)
        if not source.is_file():
            return _failure("Input must be an existing regular file.")
        with source.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            return _failure(f"Unsupported file size: maximum is {MAX_FILE_BYTES} bytes (2 MiB).")
        certificate = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, ValueError, RecursionError) as error:
        return _failure(f"Cannot read certificate: {error}")
    return validate(certificate)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        result = _failure("Usage: python3 -m verification.c7 CERTIFICATE.json")
    else:
        result = check_file(args[0])
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
