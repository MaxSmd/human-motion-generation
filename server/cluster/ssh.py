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

import shlex
import subprocess
from dataclasses import dataclass

from .. import config as cfgmod

CONTROL_PERSIST = "300"  # seconds the master lingers after last use


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


def _base_opts() -> list[str]:
    """Shared `-o` options for ssh and rsync's `-e` transport."""
    return [
        "-o", "BatchMode=yes",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={cfgmod.ssh_control_path()}",
        "-o", f"ControlPersist={CONTROL_PERSIST}",
        *cfgmod.ssh_extra_opts(),
    ]


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
    argv = [
        "ssh",
        *_base_opts(),
        "-o", f"ConnectTimeout={connect_timeout}",
        host,
        payload,
    ]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise SSHError(124, " ".join(argv), f"timed out after {timeout}s") from e
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
    transport = "ssh " + " ".join(shlex.quote(o) for o in _base_opts())
    argv = [
        "rsync",
        *flags,
        "-e", transport,
        f"{host}:{remote_path}",
        local_dir,
    ]
    try:
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
    host = cfgmod.cluster_host()
    try:
        subprocess.run(
            ["ssh", "-o", f"ControlPath={cfgmod.ssh_control_path()}", "-O", "exit", host],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        pass
