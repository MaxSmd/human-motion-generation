"""Cluster reachability probe for the connection gate.

Classifies failure so the UI can be specific: VPN down (no route / timeout) vs
SSH/auth error vs online. Cached with a short TTL so the frontend can poll every
few seconds without hammering the head node.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .. import config as cfgmod
from . import ssh

_TTL = 5.0  # seconds
_cache: tuple[float, "ClusterStatus"] | None = None


@dataclass
class ClusterStatus:
    state: str          # "online" | "vpn_down" | "ssh_error" | "disabled"
    host: str
    hostname: str | None = None
    partitions: list[str] | None = None
    latency_ms: int | None = None
    detail: str | None = None
    checked_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _classify(stderr: str) -> tuple[str, str]:
    s = stderr.lower()
    if "timed out" in s or "timeout" in s or "no route to host" in s or "operation timed out" in s:
        return "vpn_down", "No route to the head node — is the VPN connected?"
    if "could not resolve" in s or "name or service not known" in s:
        return "vpn_down", "Cannot resolve the head node — VPN likely down."
    if "permission denied" in s or "publickey" in s or "host key" in s or "authenticity" in s:
        return "ssh_error", "Reached the host but SSH failed — check your key / known_hosts."
    if "connection refused" in s:
        return "ssh_error", "Connection refused on the head node."
    return "ssh_error", stderr.strip()[:200] or "SSH failed."


def probe(force: bool = False) -> ClusterStatus:
    global _cache
    now = time.time()
    if not force and _cache and (now - _cache[0]) < _TTL:
        return _cache[1]

    host = cfgmod.cluster_host()
    if not cfgmod.cluster_mode():
        st = ClusterStatus(state="disabled", host=host, checked_at=now,
                           detail="RMG_CLUSTER_MODE is off")
        _cache = (now, st)
        return st

    t0 = time.time()
    try:
        # hostname + a compact partition summary in one round-trip.
        res = ssh.run(
            "hostname; echo '---'; sinfo -h -o '%P' 2>/dev/null | sort -u | tr '\\n' ',' ",
            timeout=10, connect_timeout=6, check=True,
        )
        latency = int((time.time() - t0) * 1000)
        out = res.stdout.strip()
        hostname, _, parts = out.partition("---")
        partitions = [p for p in parts.replace("\n", "").split(",") if p.strip()]
        st = ClusterStatus(
            state="online", host=host, hostname=hostname.strip() or None,
            partitions=partitions or None, latency_ms=latency, checked_at=now,
        )
    except ssh.SSHError as e:
        state, detail = _classify(e.stderr or str(e))
        st = ClusterStatus(state=state, host=host, detail=detail, checked_at=now)

    _cache = (now, st)
    return st


def invalidate() -> None:
    global _cache
    _cache = None
