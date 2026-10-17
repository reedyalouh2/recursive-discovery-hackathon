"""Run an untrusted candidate program under OS isolation (macOS only for now).

The candidate runs as `python -I -S -B main.py ARGS` in a fresh scratch
directory, under a deny-by-default Seatbelt profile (sandbox-exec):

- no network, child creation, other-process information or mach service lookups;
  interpreter-prefix re-execution is allowed and retains the sandbox;
- reads limited to the interpreter's installation, system libraries and Rosetta;
  file metadata (names, sizes) stays visible everywhere;
- writes limited to the scratch directory.

Resource limits: wall time enforced by an independent guardian; CPU seconds, per-file size,
open files and core dumps by rlimits set before the sandbox is entered; memory
(physical footprint) by a watchdog with a target polling interval of 10 ms,
because macOS rejects memory rlimits. Scheduling can delay enforcement; this is
not a hard memory cap. Total disk use in the scratch directory is bounded only
by time and per-file size. Other platforms raise IsolationUnavailable.
"""

import ctypes
import ctypes.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
POLL_SECONDS = 0.01
TAIL_CHARS = 2000
PROGRAM_NAME = "main.py"
MAX_PROGRAM_BYTES = 1024 * 1024

# Sets rlimits, enters the working directory, then execs sandbox-exec.
# A separate process avoids preexec_fn, which is unsafe in threaded parents.
_LAUNCHER = (
    "import json, os, resource, sys\n"
    "settings = json.loads(sys.argv[1])\n"
    "for name, (soft, hard) in sorted(settings['rlimits'].items()):\n"
    "    resource.setrlimit(getattr(resource, name), (soft, hard))\n"
    "os.chdir(settings['cwd'])\n"
    "os.execv(sys.argv[2], sys.argv[2:])\n"
)


class IsolationUnavailable(RuntimeError):
    """Candidates cannot be isolated here, so they must not run."""


class Limits:
    def __init__(self, wall_seconds=60, cpu_seconds=60, memory_bytes=1024 * 2 ** 20,
                 file_bytes=64 * 2 ** 20, open_files=64):
        for name, value in (("wall_seconds", wall_seconds), ("cpu_seconds", cpu_seconds),
                            ("memory_bytes", memory_bytes), ("file_bytes", file_bytes),
                            ("open_files", open_files)):
            if type(value) not in (int, float) or not math.isfinite(value) or not value > 0:
                raise ValueError(f"{name} must be a finite positive number.")
            if name != "wall_seconds" and int(value) != value:
                raise ValueError(f"{name} must be an integer.")
        self.wall_seconds = wall_seconds
        self.cpu_seconds = int(cpu_seconds)
        self.memory_bytes = int(memory_bytes)
        self.file_bytes = int(file_bytes)
        self.open_files = int(open_files)

    def rlimits(self):
        return {
            "RLIMIT_CPU": [self.cpu_seconds, self.cpu_seconds + 1],
            "RLIMIT_FSIZE": [self.file_bytes, self.file_bytes],
            "RLIMIT_NOFILE": [self.open_files, self.open_files],
            "RLIMIT_CORE": [0, 0],
        }

    def as_dict(self):
        return {"wall_seconds": self.wall_seconds, "cpu_seconds": self.cpu_seconds,
                "memory_bytes": self.memory_bytes, "file_bytes": self.file_bytes,
                "open_files": self.open_files}


def run_candidate(program_text, args, limits, output_name="output.json", max_output_bytes=2 * 2 ** 20,
                  python=None):
    """Run one candidate and return a result dict; infrastructure failures raise.

    status is "ok", "exit" (nonzero exit), "signal", "timeout", "cpu", "memory",
    "monitor" (unavailable telemetry) or "cancelled" (evaluator died).
    output holds the bytes of output_name if the candidate wrote it within
    max_output_bytes, else None.
    """
    reason = unavailable_reason()
    if reason:
        raise IsolationUnavailable(reason)
    if len(program_text.encode("utf-8")) > MAX_PROGRAM_BYTES:
        raise ValueError(f"Candidate program exceeds {MAX_PROGRAM_BYTES} bytes.")
    interpreter = _interpreter(python)
    control = Path(tempfile.mkdtemp(prefix="candidate-control-"))
    scratch = Path(tempfile.mkdtemp(prefix="candidate-scratch-"))
    try:
        scratch = Path(os.path.realpath(scratch))
        (scratch / PROGRAM_NAME).write_text(program_text, encoding="utf-8")
        profile = control / "profile.sb"
        profile.write_text(seatbelt_profile(interpreter["prefix"], scratch), encoding="utf-8")
        settings = json.dumps({"rlimits": limits.rlimits(), "cwd": str(scratch)})
        command = [sys.executable, "-I", "-S", "-c", _LAUNCHER, settings, SANDBOX_EXEC, "-f",
                   str(profile), interpreter["executable"], "-I", "-S", "-B", PROGRAM_NAME,
                   *map(str, args)]
        env = {"PATH": "/usr/bin:/bin", "HOME": str(scratch), "TMPDIR": str(scratch),
               "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}
        with open(control / "stdout", "wb") as stdout, open(control / "stderr", "wb") as stderr:
            result = _supervise(command, env, stdout, stderr, limits,
                                cleanup_paths=(scratch, control))
        result["stdout_tail"] = _tail(control / "stdout")
        result["stderr_tail"] = _tail(control / "stderr")
        result["output"] = _read_output(scratch / output_name, max_output_bytes)
        result["limits"] = limits.as_dict()
        return result
    finally:
        try:
            _cleanup_tree(scratch)
        finally:
            _cleanup_tree(control)


def seatbelt_profile(prefix, scratch):
    """Deny-by-default SBPL profile for one candidate run."""
    prefix, scratch = _sbpl_string(prefix), _sbpl_string(scratch)
    return f"""(version 1)
(deny default)
(deny process-info*)
(allow process-info* (target self))
(allow process-exec (subpath {prefix}))
(allow file-read-metadata)
(allow file-read-data (literal "/"))
(allow file-read*
       (subpath "/System") (subpath "/usr/lib") (subpath "/usr/share")
       (subpath "/private/var/db/dyld") (subpath "/private/var/db/timezone")
       (subpath "/Library/Apple/usr/libexec/oah") (subpath "/private/var/db/oah")
       (literal "/private/etc/localtime")
       (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random")
       (subpath {prefix}))
(allow file-write-data (literal "/dev/null"))
(allow file-read* file-write* (subpath {scratch}))
(allow signal (target self))
"""


_AVAILABILITY = {}


def unavailable_reason():
    """None if candidates can be isolated here, else why not. Cached per process."""
    if "reason" not in _AVAILABILITY:
        _AVAILABILITY["reason"] = _probe()
    return _AVAILABILITY["reason"]


def _probe():
    if sys.platform != "darwin":
        return f"Candidate isolation is implemented for macOS only, not {sys.platform}."
    if not os.access(SANDBOX_EXEC, os.X_OK):
        return f"{SANDBOX_EXEC} is not available."
    if _footprint_reader() is None:
        return "libproc proc_pid_rusage is unavailable, so memory cannot be limited."
    try:
        interpreter = _interpreter(None)
    except (OSError, ValueError) as error:
        return f"Cannot locate the candidate interpreter: {error}"
    control = Path(tempfile.mkdtemp(prefix="candidate-probe-"))
    try:
        scratch = Path(os.path.realpath(control))
        profile = control / "profile.sb"
        profile.write_text(seatbelt_profile(interpreter["prefix"], scratch), encoding="utf-8")
        process = subprocess.run(
            [SANDBOX_EXEC, "-f", str(profile), interpreter["executable"], "-I", "-S", "-c",
             "import socket\ntry:\n    socket.socket().connect(('127.0.0.1', 9))\n"
             "except PermissionError:\n    print('isolated')\n"],
            cwd=str(scratch), capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"sandbox-exec could not start: {error}"
    finally:
        shutil.rmtree(control, ignore_errors=True)
    if process.returncode != 0 or process.stdout.strip() != "isolated":
        detail = (process.stderr or process.stdout).strip()[-300:]
        return f"sandbox-exec self-test failed (exit {process.returncode}): {detail}"
    return None


_INTERPRETERS = {}


def _interpreter(python):
    executable = os.path.realpath(python or sys.executable)
    if executable not in _INTERPRETERS:
        if not os.path.isfile(executable):
            raise ValueError(f"{executable} is not a file.")
        prefix = subprocess.run([executable, "-I", "-S", "-c", "import sys; print(sys.base_prefix)"],
                                capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        prefix = os.path.realpath(prefix)
        if os.path.commonpath([prefix, executable]) != prefix:
            raise ValueError(f"{executable} lies outside its prefix {prefix}.")
        _INTERPRETERS[executable] = {"executable": executable, "prefix": prefix}
    return _INTERPRETERS[executable]


def _supervise(command, env, stdout, stderr, limits, cleanup_paths=()):
    from search import guard
    try:
        return guard.supervise(command, env, stdout, stderr, limits, cleanup_paths)
    except RuntimeError as error:
        raise IsolationUnavailable(str(error)) from error


def _supervise_local(command, env, stdout, stderr, limits, parent_pid=None):
    """Guardian-only supervision; loss of telemetry or evaluator stops the child."""
    footprint = _footprint_reader()
    if footprint is None:
        raise IsolationUnavailable("Process memory telemetry is unavailable.")
    if parent_pid is not None and os.getppid() != parent_pid:
        raise IsolationUnavailable("Evaluator exited before candidate launch.")
    started = time.monotonic()
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                               stderr=stderr, start_new_session=True)
    killed, peak = None, 0
    try:
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                break
            if parent_pid is not None and os.getppid() != parent_pid:
                killed = "cancelled"
            elif time.monotonic() - started > limits.wall_seconds:
                killed = "timeout"
            else:
                used = footprint(process.pid)
                if used is None:
                    # The child may have exited between wait4 and the query.
                    pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        break
                    killed = "monitor"
                else:
                    peak = max(peak, used)
                    if used > limits.memory_bytes:
                        killed = "memory"
            if killed:
                _kill_group(process.pid)
                pid, status, usage = os.wait4(process.pid, 0)
                break
            time.sleep(POLL_SECONDS)
    except BaseException:
        _kill_group(process.pid)
        try:
            os.waitpid(process.pid, 0)
        except ChildProcessError:
            pass
        process.returncode = -signal.SIGKILL
        raise
    wall = time.monotonic() - started
    returncode, sig = None, None
    if os.WIFEXITED(status):
        returncode = os.WEXITSTATUS(status)
    elif os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
    process.returncode = returncode if sig is None else -sig  # already reaped by wait4
    cpu = usage.ru_utime + usage.ru_stime
    if killed:
        state = killed
    elif sig == signal.SIGXCPU or (sig == signal.SIGKILL and cpu >= limits.cpu_seconds):
        state = "cpu"
    elif sig is not None:
        state = "signal"
    else:
        state = "ok" if returncode == 0 else "exit"
    return {"status": state, "returncode": returncode, "signal": sig, "wall_seconds": round(wall, 3),
            "cpu_seconds": round(cpu, 3), "max_rss_bytes": usage.ru_maxrss,
            "peak_footprint_bytes": peak}


def _cleanup_tree(path):
    """Remove our scratch tree even if the candidate restricted its permissions.

    Restore directory traversal without following candidate-created symlinks.
    Cleanup errors propagate instead of silently leaving candidate data behind.
    """
    path = Path(path)
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return

    def restore(entry, directory):
        if entry.is_symlink():
            return
        if hasattr(os, "chflags"):
            os.chflags(str(entry), 0, follow_symlinks=False)
        if directory:
            entry.chmod(0o700)

    restore(path, True)
    for current, directories, files in os.walk(str(path), followlinks=False):
        for name in directories:
            restore(Path(current) / name, True)
        for name in files:
            restore(Path(current) / name, False)
    shutil.rmtree(str(path))


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class _RUsageInfoV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
        "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
        "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten")]


def _footprint_reader():
    """Function pid -> physical footprint in bytes (None if unreadable), or None if unsupported."""
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        proc_pid_rusage = libproc.proc_pid_rusage
    except (OSError, AttributeError):
        return None
    proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    proc_pid_rusage.restype = ctypes.c_int

    def read(pid):
        info = _RUsageInfoV2()
        if proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:  # 2 = RUSAGE_INFO_V2
            return None
        return info.ri_phys_footprint
    return read


def _read_output(path, limit):
    try:
        if not path.is_file() or path.is_symlink():
            return None
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError:
        return None
    return data if len(data) <= limit else None


def _tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4 * TAIL_CHARS))
            return stream.read().decode("utf-8", "replace")[-TAIL_CHARS:]
    except OSError:
        return ""


def _sbpl_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
