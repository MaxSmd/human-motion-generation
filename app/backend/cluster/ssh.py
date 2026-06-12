"""SSH / rsync wrappers, multiplexed over one ControlMaster for the app session.

Every `ssh`/`rsync` invocation carries `ControlMaster=auto ControlPath=<sock>
ControlPersist=…`, so the first call opens the shared master connection and the
rest reuse it — fast, and a single temporary footprint on the head node (the same
as keeping one interactive SSH open). Closing is best-effort on shutdown.

All remote commands run through the user's `~/.ssh/config` alias (default `head`),
so auth / ProxyJump / keys are whatever the user already uses by hand. Remote
command strings are assembled from `shlex.quote`-d parts — never raw f-strings of
user input.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass

from .. import config as cfgmod

CONTROL_PERSIST = "300"  # seconds the master lingers after last use

# The shared ControlMaster is opened exactly once, serialized by this lock, so
# concurrent requests (status polling + parallel eval fetches) never race to
# create it (which yields "ControlSocket already exists, disabling multiplexing"
# and stray direct connections that hang). All other calls only *attach*
# (ControlMaster=no), so they multiplex over the one master.
_master_lock = threading.Lock()
_master_ok_until = 0.0  # monotonic-ish skip window to avoid an `-O check` per call
_MASTER_TTL = 30.0

# Serialize sessions over the single master. The head node refuses concurrent
# sessions on a connection AND rate-limits bursts of new connections
# (MaxStartups / fail2ban), so the only safe pattern is one-command-at-a-time
# over one reused connection. Each command is fast over the warm master, so
# serializing (default 1) costs little and keeps us gentle on the login node.
# (Override via RMG_SSH_MAX_SESSIONS only if you know the node allows more.)
_session_sem = threading.BoundedSemaphore(int(os.environ.get("MGEN_SSH_MAX_SESSIONS", os.environ.get("RMG_SSH_MAX_SESSIONS", "1"))))
_MUX_ERR = ("disabling multiplexing", "control socket", "session request failed",
            "multiplexing", "session open refused")


class SSHError(RuntimeError):
    def __init__(self, returncode: int, cmd: str, stderr: str) -> None:
        super().__init__(f"ssh exit {returncode}: {stderr.strip()[:400]}")
        self.returncode = returncode
        self.cmd = cmd
        self.stderr = stderr


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _mux_opts() -> list[str]:
    """`-o` options for attach-only ssh/rsync. ControlMaster=no → never try to
    *create* the master (only `ensure_master` does that, under the lock); just
    attach to the existing socket if present, else connect directly."""
    return [
        "-o", "BatchMode=yes",
        "-o", "ControlMaster=no",
        "-o", f"ControlPath={cfgmod.ssh_control_path()}",
        *cfgmod.ssh_extra_opts(),
    ]


def _control_check() -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", f"ControlPath={cfgmod.ssh_control_path()}", "-O", "check",
             cfgmod.cluster_host()],
            capture_output=True, text=True, timeout=6,
        )
        return r.returncode == 0
    except Exception:
        return False


def ensure_master(force: bool = False, connect_timeout: int = 8) -> None:
    """Make sure the shared ControlMaster is up. Serialized so concurrent callers
    don't race to create it. Cheap: skips re-checking within a short TTL."""
    global _master_ok_until
    with _master_lock:
        now = time.time()
        if not force and now < _master_ok_until:
            return
        if not force and _control_check():
            _master_ok_until = now + _MASTER_TTL
            return
        cp = cfgmod.ssh_control_path()
        # A dead master can leave the socket file behind → new clients see it and
        # "disable multiplexing". Remove it before (re)opening.
        try:
            if os.path.exists(cp):
                os.unlink(cp)
        except OSError:
            pass
        argv = [
            "ssh", "-M", "-N", "-f",
            "-o", "BatchMode=yes",
            "-o", "ControlMaster=yes",
            "-o", f"ControlPath={cp}",
            "-o", f"ControlPersist={CONTROL_PERSIST}",
            "-o", f"ConnectTimeout={connect_timeout}",
            *cfgmod.ssh_extra_opts(),
            cfgmod.cluster_host(),
        ]
        try:
            subprocess.run(argv, capture_output=True, text=True, timeout=connect_timeout + 12)
        except subprocess.TimeoutExpired:
            pass
        _master_ok_until = (time.time() + _MASTER_TTL) if _control_check() else 0.0


def run(
    remote_cmd: str,
    *,
    timeout: float = 30.0,
    connect_timeout: int = 8,
    check: bool = True,
    login_shell: bool = True,
) -> Result:
    """Run a command on the cluster head node and capture its output.

    `remote_cmd` is executed by the remote login shell; build it from
    `shlex.quote`-d pieces (see `submit.py`). `login_shell` wraps it in
    `bash -lc` so module/env init runs (matches the sbatch scripts' `bash -lc`).
    """
    host = cfgmod.cluster_host()
    payload = f"bash -lc {shlex.quote(remote_cmd)}" if login_shell else remote_cmd

    def _once() -> subprocess.CompletedProcess:
        argv = ["ssh", *_mux_opts(), "-o", f"ConnectTimeout={connect_timeout}", host, payload]
        with _session_sem:
            return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    ensure_master(connect_timeout=connect_timeout)
    try:
        proc = _once()
        if proc.returncode == 255 and any(e in proc.stderr.lower() for e in _MUX_ERR):
            ensure_master(force=True, connect_timeout=connect_timeout)  # stale master → rebuild
            proc = _once()
    except subprocess.TimeoutExpired as e:
        raise SSHError(124, payload, f"timed out after {timeout}s") from e
    res = Result(proc.returncode, proc.stdout, proc.stderr)
    if check and not res.ok:
        raise SSHError(res.returncode, remote_cmd, res.stderr)
    return res


def rsync_pull(
    remote_path: str,
    local_dir: str,
    *,
    timeout: float = 120.0,
    flags: tuple[str, ...] = ("-az", "--delete"),
) -> Result:
    """Pull `head:<remote_path>` into `local_dir` over the multiplexed transport.

    `remote_path` is taken verbatim (it's backend-constructed, not user text); it
    may contain a trailing `/` to copy directory contents.
    """
    host = cfgmod.cluster_host()
    ensure_master()
    transport = "ssh " + " ".join(shlex.quote(o) for o in _mux_opts())
    argv = [
        "rsync",
        *flags,
        "-e", transport,
        f"{host}:{remote_path}",
        local_dir,
    ]
    try:
        with _session_sem:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise SSHError(124, " ".join(argv), f"rsync timed out after {timeout}s") from e
    res = Result(proc.returncode, proc.stdout, proc.stderr)
    if not res.ok:
        raise SSHError(res.returncode, " ".join(argv), res.stderr)
    return res


_remote_home: str | None = None


def remote_home() -> str:
    """Cached remote `$HOME`. Lets us turn `~`/`$HOME` paths into absolute paths so
    they can be safely `shlex.quote`-d (a quoted `~` would NOT expand remotely)."""
    global _remote_home
    if _remote_home is None:
        _remote_home = run("printf %s \"$HOME\"", timeout=10).stdout.strip()
    return _remote_home


def abs_remote(path: str) -> str:
    """Expand a leading `~`/`$HOME` against the cached remote home. Falls back to
    the literal path if the cluster is unreachable (e.g. offline dry-run preview)."""
    try:
        home = remote_home()
    except SSHError:
        return path
    if path == "~" or path == "$HOME":
        return home
    if path.startswith("~/"):
        return f"{home}/{path[2:]}"
    if path.startswith("$HOME/"):
        return f"{home}/{path[6:]}"
    return path


def close_master() -> None:
    """Tear down the shared ControlMaster (best-effort; called on shutdown)."""
    global _master_ok_until
    _master_ok_until = 0.0
    cp = cfgmod.ssh_control_path()
    try:
        subprocess.run(
            ["ssh", "-o", f"ControlPath={cp}", "-O", "exit", cfgmod.cluster_host()],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        pass
    try:
        if os.path.exists(cp):
            os.unlink(cp)
    except OSError:
        pass
