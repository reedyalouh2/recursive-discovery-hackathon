"""Keep candidate supervision alive if the evaluator process is killed.

The independent guardian inherits only the candidate's log descriptors. Its
trusted supervisor observes evaluator death, kills and reaps the candidate,
and removes the abandoned scratch directories.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys


class _Terminated(BaseException):
    pass


def supervise(command, env, stdout, stderr, limits, cleanup_paths=()):
    """Run the local supervisor in an independent process and return its result."""
    output_fd, error_fd = stdout.fileno(), stderr.fileno()
    payload = {
        "command": list(command), "env": dict(env), "limits": limits.as_dict(),
        "parent_pid": os.getpid(), "stdout_fd": output_fd, "stderr_fd": error_fd,
        "cleanup_paths": [str(Path(path).resolve()) for path in cleanup_paths],
    }
    # Literal argv is atomic at launch and never interpreted by a shell. In
    # particular, evaluator death cannot leave a partially transmitted payload.
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", str(Path(__file__).resolve()), json.dumps(payload)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        pass_fds=tuple(set((output_fd, error_fd))), env=env, start_new_session=True,
    )
    try:
        output, errors = process.communicate()
    except BaseException:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    if process.returncode != 0:
        detail = errors.decode("utf-8", "replace")[-2000:]
        raise RuntimeError("Candidate guardian failed (exit %s): %s" %
                           (process.returncode, detail))
    try:
        result = json.loads(output.decode("utf-8"))
        if not isinstance(result, dict) or "status" not in result:
            raise ValueError("missing supervisor result")
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("Candidate guardian returned an invalid result: %s" % error) from None
    return result


def _worker(payload):
    # -I excludes the caller's cwd; only this trusted checkout supplies imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from search import isolation

    interrupted = False

    def terminate(signum, frame):
        nonlocal interrupted
        interrupted = True
        raise _Terminated()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        result = isolation._supervise_local(
            payload["command"], payload["env"], payload["stdout_fd"], payload["stderr_fd"],
            isolation.Limits(**payload["limits"]), parent_pid=payload["parent_pid"],
        )
        print(json.dumps(result), flush=True)
        return 0
    except _Terminated:
        return 128 + signal.SIGTERM
    finally:
        if interrupted or os.getppid() != payload["parent_pid"]:
            # Once cancellation begins, allow cleanup to finish despite a
            # second termination request from the original evaluator.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            failures = []
            for path in payload["cleanup_paths"]:
                try:
                    isolation._cleanup_tree(path)
                except OSError as error:
                    failures.append(str(error))
            if failures:
                raise RuntimeError("Guardian cleanup failed: " + "; ".join(failures))


if __name__ == "__main__":
    raise SystemExit(_worker(json.loads(sys.argv[1])))
