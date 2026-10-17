"""Official verdicts: run the checker as a separate process on certificate data.

Search code never scores its own output; manifests copy this result verbatim.
"""

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = "verification.c7"
TIMEOUT_SECONDS = 600


def check(path):
    command = [sys.executable, "-m", CHECKER, str(Path(path).resolve())]
    process = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True,
                             timeout=TIMEOUT_SECONDS, check=False)
    try:
        result = json.loads(process.stdout)
    except ValueError:
        result = None
    valid = process.returncode == 0 and isinstance(result, dict) and result.get("valid") is True
    return {
        "checker": CHECKER,
        "exit_code": process.returncode,
        "valid": valid,
        "result": result,
        "stderr": process.stderr[-4000:],
    }
