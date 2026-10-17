"""Append-only JSONL experiment history.

Events hold no timestamps or paths, so a replay with the same config, seed and
input reproduces the file byte for byte.
"""

import json


class History:
    def __init__(self, path):
        self._stream = open(path, "w", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, event, **fields):
        record = dict(fields, event=event, seq=self.count)
        self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.count += 1

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
