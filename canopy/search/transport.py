"""Every byte the paper finder pulls off the internet, and the gate each one passes first.

Canopy used to promise that it only ever read files a person had put on their own disk. The paper
finder breaks that promise deliberately — an index names a URL and *the server* fetches it — so
this module is where the promise is re-made in a form that can be checked. It is the only code in
`canopy.search` that opens a socket. Everything else talks to `SearchTransport`, which is two
methods wide precisely so a test can replace the internet outright (`RecordedTransport`).

WHY EACH RULE IS HERE
---------------------
An attacker who can influence a URL we fetch — a poisoned index record, a compromised publisher, a
302 from a host that was fine yesterday — is asking our process to make a request from *inside* the
user's network. That is SSRF, and on a laptop it reaches the router's admin page; on a cloud box it
reaches `169.254.169.254` and returns credentials. So:

* **`is_safe_public_host` resolves the name and refuses unless EVERY answer is globally routable.**
  Not the first answer — every one, because a hostile name resolves to `[8.8.8.8, 127.0.0.1]` and
  the connection would pick whichever it liked.
* **`is_global` is the primary test and the deny list is the backstop, because neither is enough
  alone.** Measured here on CPython 3.13.11, `.venv/bin/python`, `ipaddress` only:

  | literal                         | `is_private` | `is_global` | `is_reserved` | verdict |
  |---------------------------------|--------------|-------------|---------------|---------|
  | `127.0.0.1`                     | True         | False       | False         | refuse  |
  | `169.254.169.254` (metadata)    | True         | False       | False         | refuse  |
  | `100.64.0.1` (CGNAT)            | **False**    | False       | False         | refuse — `is_private` MISSES it |
  | `64:ff9b::7f00:1` (NAT64→lo)    | **False**    | **True**    | True          | refuse — `is_global` MISSES it  |
  | `::ffff:127.0.0.1` (v4-mapped)  | True         | False       | False         | refuse, after unwrapping |
  | `2002:7f00:1::` (6to4→lo)       | True         | False       | False         | refuse  |
  | `224.0.0.1`, `0.0.0.0`, `240.…` | —            | —           | —             | refuse  |
  | `8.8.8.8`, `2001:4860:4860::8888`, `100.63.255.255` | False | True | False | ALLOW |

* **The host string is resolved AS GIVEN, never parsed as an address first.** `http://2130706433/`,
  `http://017700000001/` and `http://0x7f.1/` all reach 127.0.0.1 through `getaddrinfo`, and the two
  routes genuinely disagree: on this machine `getaddrinfo("0177.0.0.1")` answers `177.0.0.1` (it
  reads the leading zero as decimal) while `ipaddress` refuses the string entirely. Whatever the C
  library will connect to is the only thing worth judging, so the C library is what we ask.
* **Every redirect hop is re-vetted.** `follow_redirects=False` is an invariant, not a default: if
  httpx followed a redirect itself it would resolve the next hop's name on its own and the address
  pinning below would be worth nothing. The loop is ours, it re-runs `check_public` on each hop, it
  resolves a relative `Location` against the name-form URL of the hop that sent it, and it stops at
  `MAX_REDIRECTS`.
* **The socket goes to the address we vetted.** Checking a name and then letting httpx resolve it
  again is a TOCTOU — DNS rebinding is exactly that race. So the request URL carries the *address*,
  the name rides in the `Host` header and in `extensions={"sni_hostname": …}`, and httpcore passes
  that to the TLS handshake as `server_hostname` (verified: httpx 0.28.1 / httpcore 1.0.9 in
  `.venv`, `httpcore/_sync/connection.py`), so the certificate is still checked against the real
  name. `CANOPY_SEARCH_PIN_ADDRESS=0` turns it off for the day a SNI-routing CDN needs it; that
  accepts the rebinding window and the caller should record which mode applied.
* **`trust_env=False`.** An `HTTP_PROXY` in the environment would route our carefully-pinned request
  through a proxy that resolves the name all over again.
* **No connection reuse**, which is a consequence of the pinning: httpcore keys its pool on the
  request URL's (scheme, host, port), the SNI name is not part of that key, and our host *is* an
  IP — so two vetted names on one CDN address would otherwise share a TLS session authenticated
  for only one of them. See the `limits=` comment in `HttpxTransport.__init__`.
* **No cookies, no auth, no credentials, port 443 only, https only.** `cookies=None` is not the
  cookie half of that: httpx builds a live jar anyway and it persists for the life of the client.
  Worse, the jar keys on the REQUEST URL's host — which, because we pin, is the shared CDN
  address — so publisher A's `Set-Cookie` rode out on the next request to publisher B. Same root
  cause as the TLS-session hazard two bullets up, and the same answer: the jar is emptied before
  every hop of every request (`_open`), so nothing a publisher sets can outlive its own response.

WHAT IT COSTS TO GET THIS WRONG IN THE OTHER DIRECTION
------------------------------------------------------
`rate_limited` is its own outcome and not a flavour of `http_error`. Europe PMC answers 429 to
back-to-back `?pdf=render` requests and sends no `Retry-After` with it; a search that folds that
into "http error" tells the user *the publisher refused* when the truth is *Canopy hammered the
index and should try again*. Those are different sentences and only one of them is honest, so
`HostClock` starts at 2 s for `europepmc.org` and 1 s for any host we have never met.

WHAT IS NOT HERE
----------------
Retry/back-off policy, the OA source order, and `probe_pdf` belong to `canopy/search/fetch.py`:
this module reports what happened and never decides what to do about it. The one exception is the
optional `probe` hook on `get_bytes`, which exists so a fetched PDF is never moved to its final
name until it has been proved readable.

TWO DELIBERATE DEVIATIONS from the backend design §2.1, both noted so nobody thinks they are slips:
the two methods are `get_json`/`get_bytes` rather than `get`/`download` (they say what they return),
and `get_bytes` returns a `Download` record rather than a 3-tuple.
"""
from __future__ import annotations

import base64
import gzip
import ipaddress
import json
import os
import socket
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol
from urllib.parse import urlencode

import httpx

from .. import __version__

__all__ = [
    "UrlRejected", "MissingSearchFixture", "VettedUrl", "HttpResponse", "Download",
    "SearchTransport", "HttpxTransport", "RecordedTransport", "RecordingTransport", "HostClock",
    "ByteBudget", "Reservation", "is_public_address", "is_safe_public_host", "check_public",
    "resolve_host", "pinned_url", "fixture_key", "html_key", "redact_params", "IDENTITY_PARAMS",
    "MAX_HTML_BYTES",
    "default_user_agent", "DENY_NETS", "HOST_INTERVALS", "DEFAULT_HOST_INTERVAL",
    "MAX_REDIRECTS", "REDIRECT_STATUSES", "ALLOWED_PORT", "PDF_CONTENT_TYPES", "OUTCOMES",
    "MAX_JSON_BYTES", "DEFAULT_TIMEOUT", "DEFAULT_FETCH_TIMEOUT",
]

# --------------------------------------------------------------------------------- the policy
#: the only scheme and the only port. Some legitimate repository OA URLs live on :8080/:8443 and
#: this rule costs us those papers — the refusal therefore NAMES the port (review S2) so the
#: candidate is recorded as "we would not fetch this" rather than "the publisher refused".
ALLOWED_SCHEME = "https"
ALLOWED_PORT = 443

#: refused whatever the resolver says — belt to `is_global`'s braces, because CPython's
#: classification of a few of these ranges (100.64.0.0/10, 64:ff9b::/96) has moved between 3.11
#: and 3.13, and a security predicate that changes meaning on a Python upgrade is not a predicate.
DENY_NETS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
    ipaddress.ip_network(n) for n in (
        "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
        "172.16.0.0/12", "192.0.0.0/24", "192.168.0.0/16", "198.18.0.0/15", "224.0.0.0/4",
        "240.0.0.0/4", "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8",
        "64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32"))

#: statuses whose `Location` we follow ourselves, one re-vetted hop at a time
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECTS = 5

#: what a fetched body may claim to be. `application/octet-stream` is on the list because half of
#: the repository servers that hold OA PDFs mislabel them, and a MISSING content-type is treated as
#: octet-stream for the same reason (review S2) — the `%PDF-` magic check inside `stream_upload` is
#: what actually decides, and it cannot be talked out of it by a header.
PDF_CONTENT_TYPES = frozenset({
    "application/pdf", "application/x-pdf", "application/octet-stream", "binary/octet-stream"})

#: one word per outcome, and `rate_limited` is not `http_error` (see the module docstring)
OUTCOMES: tuple[str, ...] = (
    "ok",             # 2xx, and whatever gate applied to this call was satisfied
    "refused",        # this server would not send the request: scheme, port, credentials, address
    "dns_error",      # the name does not resolve — nothing was sent
    "rate_limited",   # HTTP 429: WE were too fast. Never conflated with the publisher saying no
    "http_error",     # any other >= 400, and a redirect chain that never ended
    "timeout",        # the clock ran out mid-request
    "network_error",  # connection reset, TLS failure, and other things that are nobody's fault
    "not_a_pdf",      # a body arrived and it is a login wall, a bot-block page, or simply not a PDF
    "too_large",      # the per-file cap or the search's total byte budget stopped it
    "unreadable",     # a real PDF arrived and the ingester could not read it (optional probe hook)
)

#: per-host courtesy, in seconds between the STARTS of two requests to the same host.
#: `europepmc.org` is 2.0 s because two back-to-back `?pdf=render` fetches were measured returning
#: 200 then 429 (no Retry-After, no rate-limit headers), and it is the primary OA route — every
#: other host we have never met gets the polite default rather than a guess.
HOST_INTERVALS: dict[str, float] = {
    "europepmc.org": 2.0,
    "www.ebi.ac.uk": 0.2,          # Europe PMC's search API: no documented limit, so be polite
    "api.openalex.org": 0.05,      # 100 req/s is the cap; the daily budget is the real constraint
    "api.crossref.org": 0.35,      # the live header says 3/s, whatever the documentation says
    "api.unpaywall.org": 0.1,
    "web.archive.org": 2.0,        # the last-resort copy; the Wayback Machine throttles bursts
    "eutils.ncbi.nlm.nih.gov": 0.4,   # 3 requests/s without a key (NCBI's own rule)
}
DEFAULT_HOST_INTERVAL = 1.0
#: how much of a landing page is worth reading for a PDF link. `citation_pdf_url` sits in `<head>`.
MAX_HTML_BYTES = 1_000_000

DEFAULT_TIMEOUT = 20.0             # one index call
#: how long one transport reuses a name's vetted addresses before asking DNS again
RESOLVE_TTL_S = 300.0
DEFAULT_FETCH_TIMEOUT = 60.0       # one PDF download
#: a JSON body is held in memory, so it needs a cap of its own: an index that answers with 10 GB of
#: `{` should cost us one recorded error, not the machine.
MAX_JSON_BYTES = 32 * 1024 * 1024
#: the per-file default. `canopy.server.uploads` owns the real number; this only stands in when a
#: caller passes nothing.
DEFAULT_MAX_PDF_BYTES = float(os.environ.get("CANOPY_SEARCH_MAX_PDF_MB", "")
                              or os.environ.get("CANOPY_MAX_UPLOAD_MB", "") or 50) * 1e6
#: bytes pulled off the wire at a time — mirrors `uploads.CHUNK` so the two agree about how far
#: past a cap a stream can get before it is cut (one chunk, bounded)
CHUNK = 1 << 20


class UrlRejected(ValueError):
    """A URL this server will not fetch. The message is shown to the user verbatim.

    It carries an `outcome` because "we refused" and "it does not resolve" are different facts and
    a record that flattens them cannot tell a typo from an attack.
    """

    def __init__(self, message: str, outcome: str = "refused",
                 hops: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.outcome = outcome
        #: the hops that were tried before this one was refused, so a record can show the chain
        self.hops = hops


class MissingSearchFixture(LookupError):
    """A test asked `RecordedTransport` for a request nobody recorded.

    Loud on purpose, exactly like `MissingFixture` in `canopy/llm/client.py`: a fake that invents a
    plausible empty answer turns "the index returned nothing" — the precise failure this whole
    feature exists to prevent — into a passing test.
    """


# ------------------------------------------------------------------------------- the address gate
def is_public_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """One address, judged. `is_global` is the primary test; the deny list is the backstop.

    A v4-mapped v6 address is unwrapped and judged as the v4 address it really is, because
    `::ffff:127.0.0.1` is a loopback wearing a hat.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return is_public_address(mapped)
    # the explicit checks are not redundant with `is_global`: they say out loud which properties
    # this server cares about, so a future Python that reclassifies one of them still refuses.
    if (ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified):
        return False
    if not ip.is_global:
        return False
    return not any(ip in net for net in DENY_NETS if net.version == ip.version)


def resolve_host(host: str, port: int = ALLOWED_PORT) -> tuple[str, ...]:
    """Every address `host` resolves to, in the C library's own words.

    The host string goes in AS GIVEN. Parsing it with `ipaddress` first and falling back to
    `getaddrinfo` would be a second, differently-behaved parser: on this machine
    `getaddrinfo("0177.0.0.1")` answers `177.0.0.1` and `ipaddress.ip_address("0177.0.0.1")`
    refuses the string. Only what the C library will actually connect to is worth judging.
    """
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


def _judge_addresses(host: str, addresses: Iterable[str]) -> str:
    """`""` when every address is a public internet address, else the refusal, naming the address.

    EVERY address, not the first: a hostile name answers `[8.8.8.8, 127.0.0.1]` and the connection
    would take whichever it fancied.
    """
    seen = list(addresses)
    if not seen:
        return f"{host!r} does not resolve to any address"
    for raw in seen:
        # an IPv6 answer can carry a scope id (`fe80::1%en0`); `ipaddress` cannot parse that, and
        # anything scoped is link-local by construction, so strip it and let the predicate refuse.
        text = raw.split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(text)
        except ValueError:
            return f"{host!r} resolved to {raw!r}, which is not an address this server can judge"
        if not is_public_address(ip):
            return (f"{host!r} resolves to {ip.compressed}, which is not a public internet "
                    f"address — this server only fetches from the public internet")
    return ""


def _allowed_private_hosts() -> frozenset[str]:
    """Hosts allowed to resolve privately. TESTS ONLY, and unset by default.

    Read on every call rather than at import, so a test can set it with `monkeypatch` and cannot
    leave it set for the next one. `serve()` never sets it; setting it in production hands an
    index the run of your local network.
    """
    raw = os.environ.get("CANOPY_SEARCH_ALLOW_PRIVATE_HOSTS", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def is_safe_public_host(host: str,
                        *, resolve: Callable[[str, int], Iterable[str]] = resolve_host,
                        port: int = ALLOWED_PORT) -> tuple[bool, str]:
    """`(True, "")` when every address `host` resolves to is publicly routable, else `(False, why)`.

    `why` is a sentence for a person, and it names the offending address, because "we refused
    example.org" without saying it landed on 10.0.0.5 is unactionable.
    """
    name = (host or "").strip().lower()
    if not name:
        return False, "the URL has no host"
    if name in _allowed_private_hosts():
        return True, ""
    try:
        addresses = tuple(resolve(name, port))
    except OSError as exc:
        return False, f"{name!r} does not resolve ({exc.__class__.__name__})"
    reason = _judge_addresses(name, addresses)
    return (not reason), reason


@dataclass(frozen=True)
class VettedUrl:
    """A URL that passed the gate, with the addresses it passed on."""

    url: str                      # the name-form URL, normalised. Relative redirects join to THIS
    host: str                     # ascii host: the `Host` header and the TLS SNI name
    address: str                  # the address the socket is pinned to
    addresses: tuple[str, ...]    # every answer, all of them vetted


def check_public(url: str,
                 *, resolve: Callable[[str, int], Iterable[str]] = resolve_host) -> VettedUrl:
    """The whole gate for one URL. Raises `UrlRejected`; does no I/O beyond resolving the name.

    Refusals in this order, each naming what it objected to: a scheme that is not https, a
    userinfo component, a port that is not 443, a missing host, a name that does not resolve, and
    any host ANY of whose addresses is not public.
    """
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, ValueError, TypeError):
        raise UrlRejected(f"{str(url)[:200]!r} is not a URL this server can parse") from None

    scheme = (parsed.scheme or "").lower()
    if scheme != ALLOWED_SCHEME:
        # `http` reaches here only on a redirect hop: the first URL may be rewritten to https once
        # (§5.2), but a 302 to `http://` is the hop an attacker controls and is never followed.
        raise UrlRejected(f"only https:// URLs are fetched — {scheme or 'this URL'}:// is not")
    if parsed.userinfo:
        raise UrlRejected("a URL with a username or password in it is never fetched")
    if parsed.port is not None and parsed.port != ALLOWED_PORT:
        raise UrlRejected(f"only port {ALLOWED_PORT} is fetched — this URL asks for port "
                          f"{parsed.port}, so it was not tried")
    host = (parsed.raw_host or b"").decode("ascii", "replace")   # punycode: what will be sent
    if not host:
        raise UrlRejected("this URL has no host")

    name = host.lower()
    if name in _allowed_private_hosts():
        return VettedUrl(url=str(parsed), host=name, address=name, addresses=(name,))
    try:
        addresses = tuple(resolve(name, ALLOWED_PORT))
    except OSError as exc:
        raise UrlRejected(f"{name!r} does not resolve ({exc.__class__.__name__})",
                          outcome="dns_error") from None
    reason = _judge_addresses(name, addresses)
    if reason:
        raise UrlRejected(reason)
    return VettedUrl(url=str(parsed), host=name, address=addresses[0], addresses=addresses)


def pinned_url(vetted: VettedUrl) -> str:
    """The same URL with the vetted ADDRESS in place of the name, so DNS cannot answer twice."""
    return str(httpx.URL(vetted.url).copy_with(host=vetted.address))


# ------------------------------------------------------------------------------ per-host courtesy
class HostClock:
    """One request per host per interval, across every thread in this search.

    The naive version of this (read `last`, sleep the difference, write `last`) does not survive
    threads: two workers read the same `last`, compute the same short wait and fire together, which
    is the burst the interval existed to prevent. So the slot is CLAIMED under a per-host lock
    before the sleep — a second thread queues behind the first and comes out one interval later.
    The lock is per host, so two different hosts never wait for each other.

    `sleep`/`now` are injectable so a test can pin the arithmetic without spending real seconds.
    """

    def __init__(self, intervals: Mapping[str, float] | None = None,
                 default: float = DEFAULT_HOST_INTERVAL,
                 sleep: Callable[[float], Any] = time.sleep,
                 now: Callable[[], float] = time.monotonic) -> None:
        self._intervals = {k.lower(): float(v) for k, v in HOST_INTERVALS.items()}
        self._intervals.update({k.lower(): float(v) for k, v in (intervals or {}).items()})
        self._default = float(default)
        self._sleep = sleep
        self._now = now
        self._table_lock = threading.Lock()
        self._host_locks: dict[str, threading.Lock] = {}
        self._next_slot: dict[str, float] = {}

    def interval_for(self, host: str) -> float:
        """This host's interval: the exact entry, else a parent-domain entry, else the default.

        The parent match matters because Europe PMC serves PDFs from `europepmc.org` and redirects
        within it (`europepmc.org/api/getPdf`), and both must count as the one host that asked us
        to slow down.
        """
        name = (host or "").lower()
        if name in self._intervals:
            return self._intervals[name]
        parts = name.split(".")
        for cut in range(1, len(parts) - 1):
            parent = ".".join(parts[cut:])
            if parent in self._intervals:
                return self._intervals[parent]
        return self._default

    def set_interval(self, host: str, seconds: float) -> None:
        """Correct a host's interval from what it just told us.

        This is the seam for believing a `x-rate-limit-*` header over the documentation — Crossref
        documents 10 req/s and its live response says 3. Believing the document is how you earn a
        429; the transport reads the headers and the caller calls this.
        """
        with self._table_lock:
            self._intervals[(host or "").lower()] = max(0.0, float(seconds))

    def _lock_for(self, host: str) -> threading.Lock:
        with self._table_lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def wait(self, host: str, interval: float | None = None) -> float:
        """Block until this host may be asked again. Returns how long that took."""
        name = (host or "").lower()
        gap = self.interval_for(name) if interval is None else float(interval)
        if gap <= 0:
            return 0.0
        with self._lock_for(name):
            now = self._now()
            earliest = self._next_slot.get(name, now)
            delay = max(0.0, earliest - now)
            self._next_slot[name] = now + delay + gap   # claimed BEFORE the sleep, so a second
            if delay > 0:                               # thread queues behind this one
                self._sleep(delay)
        return delay


# ------------------------------------------------------------------------------- the byte budget
@dataclass
class Reservation:
    """A slice of the search's byte allowance, held while one download runs.

    A context manager because the failure mode it exists to prevent is a worker that raises between
    reserving and settling: on the way out, whatever was not settled goes back to the pool.
    """

    budget: ByteBudget
    granted: float
    _closed: bool = field(default=False, repr=False)
    #: the close is a check-then-set, and `settle` racing `refund` on two threads would release the
    #: allowance twice and inflate the budget above its own total. Its own lock, not the budget's:
    #: `_release` takes that one, and `threading.Lock` does not nest.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _close(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            return True

    def settle(self, used: float) -> None:
        """Keep `used` bytes and hand the rest back."""
        if self._close():
            self.budget._release(max(0.0, self.granted - max(0.0, float(used))))

    def refund(self) -> None:
        """Nothing was written: give it all back."""
        if self._close():
            self.budget._release(self.granted)

    def __enter__(self) -> Reservation:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.refund()


class ByteBudget:
    """The search's total download allowance, shared by every fetch worker. Thread-safe.

    Review amendment S1: the sequential code this was lifted from treats `remaining` as a local, so
    three concurrent workers each pass their own `total > remaining` check against the same
    snapshot and jointly write up to three times the allowance. Making the arithmetic atomic is the
    fix, and holding a RESERVATION rather than a number is what makes it impossible for a worker to
    forget: the allowance is gone the moment it is handed out, and comes back on the way out.

    It bounds the bytes KEPT, not the bytes fetched: a download that is refused part-way (too big,
    not a PDF, unreadable) hands its whole reservation back, because nothing of it survives on
    disk. A host that answers "too big" forever is therefore bounded by the per-file cap and the
    job deadline, not by this — which is the right division of labour, but is worth saying out
    loud so nobody reads `total` as a transfer quota.

    `ByteBudget(None)` is unlimited, so callers have one code path rather than two.
    """

    def __init__(self, total: float | None) -> None:
        self.total = None if total is None else float(total)
        self._remaining = self.total
        self._lock = threading.Lock()

    @property
    def remaining(self) -> float | None:
        with self._lock:
            return self._remaining

    @property
    def spent(self) -> float:
        with self._lock:
            return 0.0 if self.total is None or self._remaining is None else (
                self.total - self._remaining)

    def reserve(self, want: float) -> Reservation:
        """Take up to `want` bytes out of the pool now. `granted` may be less, or zero."""
        want = max(0.0, float(want))
        with self._lock:
            if self._remaining is None:
                granted = want
            else:
                granted = max(0.0, min(want, self._remaining))
                self._remaining -= granted
        return Reservation(self, granted)

    def _release(self, amount: float) -> None:
        if amount <= 0:
            return
        with self._lock:
            if self._remaining is not None:
                self._remaining = min(self.total or 0.0, self._remaining + amount)


# ------------------------------------------------------------------------------------ the results
@dataclass(frozen=True)
class HttpResponse:
    """One completed — or refused — HTTP GET. Never an exception: a dead index is data.

    A 503, a timeout, a DNS failure and an SSRF refusal all arrive here with an `outcome` and a
    sentence, so the orchestrator records them in `search.json:api_calls` and the search carries on
    with the other index. That is the failure-honesty rule expressed as a method contract.
    """

    url: str = ""                                   # the FINAL url, after redirects
    status: int = 0                                 # 0 when nothing was sent
    outcome: str = "ok"                             # one of OUTCOMES
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""                               # empty for downloads (they stream to disk)
    error: str = ""                                 # "" on success; a sentence otherwise
    seconds: float = 0.0
    hops: tuple[str, ...] = ()                      # every URL tried, in order, name-form
    rewritten_from: str = ""                        # set when http:// was upgraded once (§5.2)
    retry_after: float | None = None                # None when the 429 carried no such header

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"

    def json(self) -> Any | None:
        """The parsed body, or None when it is not JSON. Never raises: a bot-block page is data."""
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8", "replace"))
        except ValueError:
            return None


@dataclass(frozen=True)
class Download:
    """What one PDF fetch produced: the response, the file (if any), and its size."""

    response: HttpResponse
    path: Path | None = None
    n_bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.path is not None and self.response.ok


class SearchTransport(Protocol):
    """The one seam. Two methods, so a test can replace the internet without mocking httpx.

    Neither method raises for anything that happened on the wire; both return the outcome as data.

    Every keyword either implementation accepts is named HERE, and both of them are checked
    against this class by `test_the_two_transports_accept_the_same_keywords`. The fake used to end
    both signatures in `**_: Any`, which meant it swallowed anything: the two had already drifted
    (`accept`, `content_types` and `allow_http_rewrite` existed only on the real one), so a
    renamed parameter at any call site would have passed every offline test and `TypeError`d the
    first time a real search ran.
    """

    def get_json(self, url: str, *, params: Mapping[str, Any] | None = ...,
                 headers: Mapping[str, str] | None = ..., timeout: float = ...,
                 accept: str = ..., max_bytes: float = ...,
                 context: Mapping[str, Any] | None = ...,
                 form_data: Mapping[str, Any] | None = ...) -> HttpResponse:
        """`context` is what the CALLER knows about this request and the wire does not: the
        `{index, form, query_id, page, query_text}` tuple `search_pages` passes. The real
        transport ignores it; the recorder writes it into the fixture; the replayer falls back
        to it when the exact request was never recorded (`RecordedTransport.query_drift`).

        `form_data` turns the call into a POST with a form-encoded body — for the one index
        (PubMed's esearch) whose GET URLs 414 above ~3,000 characters while NCBI documents POST
        for exactly that. The fixture key covers `params` and `form_data` alike."""
        ...

    def get_bytes(self, url: str, dest_dir: str | Path, *, filename: str = ...,
                  headers: Mapping[str, str] | None = ..., timeout: float = ...,
                  max_bytes: float = ..., accept: str = ...,
                  content_types: Iterable[str] = ..., allow_http_rewrite: bool = ...,
                  probe: Callable[[Path], Mapping[str, Any]] | None = ...) -> Download:
        ...

    def get_html(self, url: str, *, headers: Mapping[str, str] | None = ...,
                 timeout: float = ..., max_bytes: float = ...) -> HttpResponse:
        """A landing page, so the fetch stage can read the PDF link out of it. Through the same
        hop-vetting loop as everything else — the page is somebody else's HTML and the link in
        it is re-vetted by `get_bytes` before a byte of it is fetched."""
        ...


# ---------------------------------------------------------------------------------- the internals
class _ChunkReader:
    """`.read(n)` over an httpx byte iterator, so a download goes through `stream_upload`.

    This adapter is the whole reason a fetched PDF and a hand-uploaded one are the same kind of
    object: the `%PDF-` magic check, both size caps, the sha256 name and the always-removed partial
    file are `canopy.server.uploads`' code, not a second, subtly different copy of it.
    """

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = iter(chunks)
        self._buffer = bytearray()
        self.n_read = 0            # bytes actually pulled off the wire, for the size-cap test

    def read(self, n: int) -> bytes:
        while len(self._buffer) < n:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                break
            if not chunk:
                continue
            self.n_read += len(chunk)
            self._buffer.extend(chunk)
        out = bytes(self._buffer[:n])
        del self._buffer[:n]
        return out


class _BodyTooLarge(Exception):
    """Raised while reading, so the connection is dropped AT the cap and not after it."""


class _Rejected(Exception):
    """The bytes arrived and did not survive the gate. Carries the outcome word, so that
    `stream_upload`'s refusals and the probe's refusal reach the record in this module's own
    vocabulary rather than as three different exception types the caller must know about."""

    def __init__(self, message: str, outcome: str) -> None:
        super().__init__(message)
        self.outcome = outcome


def _content_type(headers: Mapping[str, str]) -> str:
    """The bare media type, lowercased. A missing header is `application/octet-stream`.

    Missing rather than refused, because half the repository servers that hold OA PDFs send no
    content-type at all and the magic-byte check is what actually decides (review S2).
    """
    raw = (headers.get("content-type") or "").split(";")[0].strip().lower()
    return raw or "application/octet-stream"


def _declared_length(headers: Mapping[str, str]) -> int | None:
    try:
        return int(headers["content-length"])
    except (KeyError, TypeError, ValueError):
        return None


def _size(n: float) -> str:
    """A size a person can read. `"0.0 MB"` is not a refusal anyone can act on."""
    return f"{int(n)} bytes" if n < 1e6 else f"{n / 1e6:.1f} MB"


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds, when the server said. Europe PMC's 429 says nothing, and that is worth recording."""
    try:
        return max(0.0, float(headers["retry-after"]))
    except (KeyError, TypeError, ValueError):
        return None


def default_user_agent(contact_email: str = "") -> str:
    """A descriptive UA with a contact address when there is one — the etiquette every one of these
    APIs asks for. No project URL is invented here: this repository has no published one, and a UA
    pointing at a page that does not exist is worse manners than none."""
    override = os.environ.get("CANOPY_SEARCH_USER_AGENT", "").strip()
    if override:
        return override
    email = (contact_email or os.environ.get("CANOPY_CONTACT_EMAIL", "")).strip()
    return f"canopy-meta/{__version__}" + (f" (mailto:{email})" if email else "")


#: request parameters that say WHO is asking rather than WHAT is asked: the polite-pool address
#: the indexes ask for, Unpaywall's mandatory `email`, OpenAlex's `api_key`, eutils' `tool`. They
#: are left out of `fixture_key` so a recording made under one person's address replays for
#: everyone, and they are stripped from every recorded fixture (`redact_params`) so a key never
#: lands in a file that is committed.
IDENTITY_PARAMS: frozenset[str] = frozenset({"mailto", "email", "api_key", "tool"})


def redact_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """The request parameters minus the ones that identify (or authenticate) the caller."""
    return {k: v for k, v in dict(params or {}).items() if k not in IDENTITY_PARAMS}


def fixture_key(url: str, params: Mapping[str, Any] | None = None) -> str:
    """A stable, path-safe key for one request: `<host>-<sha1>`.

    Readable half so a person can find the fixture; hashed half so query strings that differ only
    in ordering are one recording. `IDENTITY_PARAMS` are not part of the key: they change with the
    operator, not with the question, and a fixture that only replayed for the address it was
    recorded under would be a fixture for one laptop.
    """
    import hashlib

    try:
        parsed = httpx.URL(url, params=redact_params(params))
    except (httpx.InvalidURL, ValueError, TypeError):
        parsed = None
    if parsed is None:
        host, canonical = "invalid", str(url)
    else:
        host = (parsed.host or "invalid").replace(".", "-")
        query = "&".join(f"{k}={v}" for k, v in sorted(parsed.params.multi_items())
                         if k not in IDENTITY_PARAMS)
        canonical = f"{parsed.scheme}://{parsed.host}{parsed.path}?{query}"
    return f"{host}-{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:16]}"


def html_key(url: str) -> str:
    """The fixture key of a landing page: the URL's key with `-html`, so a page and a download
    of the same address are two recordings."""
    return fixture_key(url) + "-html"


def _store(reader: _ChunkReader, dest_dir: str | Path, filename: str, *,
           max_bytes: float, remaining_bytes: float | None,
           probe: Callable[[Path], Mapping[str, Any]] | None) -> tuple[Path, int]:
    """Write one stream to `<dest_dir>/<sha256>.pdf` through `stream_upload`, via a staging dir.

    Two things this does that a bare `stream_upload` call would not:

    * it stages inside `dest_dir` and only `os.replace`s into place once the limits AND the probe
      have passed, so bytes never sit under their final name while still unproven — the run
      directory is served, and a half-checked file with a real name is a file something else will
      pick up;
    * it runs the probe against the staged path, so an unreadable "PDF" leaves nothing behind.

    `stream_upload` still does all the deciding: the name rule, the `%PDF-` magic on the first
    chunk, both size caps checked BEFORE each write, the sha256 name and the removal of the partial
    file. Those rules must be identical to a human upload's, so they are not restated here.
    """
    # Imported here, not at module import: `canopy.server.__init__` pulls in FastAPI and the whole
    # pipeline (~1 s), and the SSRF gate must stay importable and testable without any of it.
    from ..server.uploads import UploadRejected, stream_upload

    target_dir = Path(dest_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target_dir, prefix=".fetch-") as staging:
        try:
            staged, n_bytes = stream_upload(reader, staging, filename,
                                            max_bytes=max_bytes, remaining_bytes=remaining_bytes)
        except UploadRejected as exc:
            # a 413 is a size problem; everything else it raises (not a `.pdf` name, no `%PDF-`
            # magic, an empty body) means the bytes were not a paper
            raise _Rejected(str(exc),
                            "too_large" if exc.status_code == 413 else "not_a_pdf") from None
        if probe is not None:
            # the probe parses bytes an attacker chose, so it RAISING is an ordinary outcome, not a
            # bug: a decompression bomb or a broken xref table takes the parser with it. Letting
            # that escape would end the whole search over one bad paper.
            try:
                result = probe(staged)
            except Exception as exc:
                raise _Rejected(f"the PDF could not be read: {exc}"[:300], "unreadable") from exc
            if not (result or {}).get("ok"):
                raise _Rejected(str((result or {}).get("error") or "the PDF could not be read"),
                                "unreadable")
        final = target_dir / staged.name
        os.replace(staged, final)
    return final, n_bytes


# ------------------------------------------------------------------------------- the real thing
class HttpxTransport:
    """The real transport: https-only, address-pinned, redirect-vetting, courteous.

    One instance per search job, because the host clock and the byte budget are per-search facts
    and a second instance would quietly double both.
    """

    def __init__(self, *, user_agent: str = "", clock: HostClock | None = None,
                 budget: ByteBudget | None = None, client: httpx.Client | None = None,
                 resolve: Callable[[str, int], Iterable[str]] = resolve_host,
                 pin_address: bool | None = None, max_redirects: int = MAX_REDIRECTS) -> None:
        self.user_agent = user_agent or default_user_agent()
        self.clock = clock if clock is not None else HostClock()
        self.budget = budget if budget is not None else ByteBudget(None)
        self.resolve = resolve
        # a name resolved once is reused for `RESOLVE_TTL_S`: the first 500-paper fetch made a
        # lookup per request and the local resolver started answering "does not resolve"
        # (96 Europe PMC fetches lost to gaierror in one run). The address a name gave is the
        # address the socket is pinned to either way; asking DNS again every time was only
        # ever a chance for it to say something else.
        self._resolved: dict[tuple[str, int], tuple[float, tuple[str, ...]]] = {}
        self._resolve_lock = threading.Lock()
        self.max_redirects = max(0, int(max_redirects))
        # A real flag rather than an undesigned escape hatch: pinning is untested against
        # SNI-routing CDNs and IPv6-only hosts, and the day it breaks one, the operator needs a way
        # out that they can also WRITE DOWN in the record ("this search ran unpinned").
        self.pin_address = (pin_address if pin_address is not None
                            else os.environ.get("CANOPY_SEARCH_PIN_ADDRESS", "1")
                            not in ("0", "false", "False", "no"))
        self.client = client if client is not None else httpx.Client(
            follow_redirects=False,   # INVARIANT, not a default — see the module docstring
            trust_env=False,          # no proxy gets to resolve the name we already vetted
            cookies=None, auth=None,
            # No connection reuse, and this one is a consequence of the pinning rather than
            # ordinary caution. httpcore matches a pooled connection with
            # `origin == self._origin`, and `origin` is (scheme, host, port) taken from the
            # REQUEST URL — which, because we pin, is the IP address. The SNI name lives in
            # `extensions` and is not part of that key. So two vetted names sharing one CDN
            # address would share one TLS session, and the second request would ride on a
            # certificate that was only ever verified for the first name. A fresh handshake per
            # request costs us nothing worth having: this transport makes a handful of requests
            # spaced a second or more apart by `HostClock`.
            limits=httpx.Limits(max_keepalive_connections=0),
            timeout=httpx.Timeout(connect=5.0, read=DEFAULT_TIMEOUT, write=5.0, pool=5.0),
            headers={"User-Agent": self.user_agent})
        if self.client.follow_redirects:
            raise ValueError("HttpxTransport needs follow_redirects=False: httpx following a "
                             "redirect itself would resolve the next hop's name and defeat the "
                             "address pinning")

    def _cached_resolve(self, host: str, port: int = ALLOWED_PORT) -> tuple[str, ...]:
        """`self.resolve`, remembered per (host, port) for `RESOLVE_TTL_S`. A failure is never
        cached: the next request asks again, which is the retry a transient lookup error needs."""
        key = (host.lower(), int(port))
        now = time.monotonic()
        with self._resolve_lock:
            hit = self._resolved.get(key)
            if hit is not None and now - hit[0] < RESOLVE_TTL_S:
                return hit[1]
        addresses = tuple(self.resolve(host, port))
        with self._resolve_lock:
            self._resolved[key] = (now, addresses)
        return addresses

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> HttpxTransport:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ the re-vetting hop loop
    @contextmanager
    def _open(self, url: str, *, headers: Mapping[str, str] | None, timeout: float,
              accept: str, allow_http_rewrite: bool,
              form_data: Mapping[str, Any] | None = None,
              ) -> Iterator[tuple[httpx.Response, VettedUrl, list[str], str]]:
        """Yield the final response, with every hop on the way re-vetted through `check_public`.

        `form_data` is POSTed, form-encoded, on the FIRST hop only; a redirect is followed with
        a GET, as a browser follows a 303. The one caller that POSTs (PubMed) is never redirected.
        """
        target = str(url)
        rewritten_from = ""
        if allow_http_rewrite and target[:7].lower() == "http://":
            # ~12 % of index-supplied OA PDF URLs are plain http, so a flat refusal would silently
            # cost an eighth of the corpus. Upgraded ONCE, before the first request, and recorded —
            # nothing is upgraded silently, and no REDIRECT is ever upgraded (that is the hop an
            # attacker controls).
            rewritten_from, target = target, "https://" + target[7:]

        hops: list[str] = []
        seen: set[str] = set()
        request_headers = {**{"User-Agent": self.user_agent, "Accept": accept}, **(headers or {})}
        with ExitStack() as stack:
            for _hop in range(self.max_redirects + 1):
                try:
                    vetted = check_public(target, resolve=self._cached_resolve)
                except UrlRejected as exc:
                    # name the hop. "we refused 127.0.0.1" is true but useless in a record; the
                    # user needs to see that a URL they trusted sent us somewhere they did not.
                    if not hops:
                        raise
                    raise UrlRejected(f"a redirect from {hops[-1]} to {target} was refused: {exc}",
                                      outcome=exc.outcome, hops=tuple(hops)) from None
                hops.append(vetted.url)
                seen.add(vetted.url)
                # Empty the jar before EVERY hop, not once per request. `Client(cookies=None)`
                # still builds a live `Cookies()` that outlives the response, and it keys on the
                # request URL's host — the pinned IP — so one publisher's session cookie was sent
                # to the next publisher that happened to share a CDN address, and to the next hop
                # of a redirect chain that had left the host that set it. Assigning a fresh jar
                # does NOT work (the setter re-wraps whatever it is given); clearing does.
                self.client.cookies.clear()
                self.clock.wait(vetted.host)
                wire_url = pinned_url(vetted) if self.pin_address else vetted.url
                posting = form_data is not None and not hops[:-1]
                try:
                    response = stack.enter_context(self.client.stream(
                        "POST" if posting else "GET", wire_url,
                        headers={**request_headers, "Host": vetted.host,
                                 **({"Content-Type": "application/x-www-form-urlencoded"}
                                    if posting else {})},
                        content=(urlencode({k: str(v) for k, v in dict(form_data or {}).items()})
                                 .encode("ascii") if posting else None),
                        extensions={"sni_hostname": vetted.host},
                        timeout=timeout))
                except httpx.InvalidURL as exc:
                    # httpx builds the redirect request EAGERLY, to fill in
                    # `response.next_request`, even though `follow_redirects=False` means it will
                    # never send it — so a `Location: data:…` / `javascript:…` / `about:blank`
                    # raises `InvalidURL` out of `stream()` itself, before our own join ever sees
                    # it. `InvalidURL` is not an `httpx.HTTPError` (its bases are `Exception`,
                    # `BaseException`), so neither caller's except tuple matched and it escaped a
                    # method documented as never raising; `run.py`'s broad catch then voided the
                    # fetch stage for every remaining paper, under a note naming none of them.
                    raise UrlRejected(f"{vetted.url} answered with a Location this server cannot "
                                      f"follow ({exc})", outcome="refused",
                                      hops=tuple(hops)) from None
                if response.status_code not in REDIRECT_STATUSES:
                    yield response, vetted, hops, rewritten_from
                    return
                location = response.headers.get("location", "")
                response.close()          # a redirect's body is never read
                if not location:
                    raise UrlRejected(f"{vetted.url} answered {response.status_code} with no "
                                      f"Location to follow", outcome="http_error",
                                      hops=tuple(hops))
                # relative Location resolves against the NAME-form URL of the hop that sent it,
                # never against the address-pinned one we actually put on the wire
                try:
                    target = str(httpx.URL(vetted.url).join(location))
                except (httpx.InvalidURL, ValueError, TypeError) as exc:
                    # Belt to the braces above: the same exception type from the other place it
                    # can come from. httpx's eager redirect build catches most of these first, so
                    # this arm is for a `Location` that parses on its own and not against this
                    # base — and the answer is the same either way, a refusal about THIS
                    # candidate rather than an exception out of a method that promises none.
                    raise UrlRejected(f"{vetted.url} redirected to {str(location)[:120]!r}, "
                                      f"which is not a URL this server can follow ({exc})",
                                      outcome="refused", hops=tuple(hops)) from None
                if target in seen:
                    raise UrlRejected(f"redirect loop at {target}", outcome="http_error",
                                      hops=tuple(hops))
            raise UrlRejected(f"more than {self.max_redirects} redirects, starting at {hops[0]}",
                              outcome="http_error", hops=tuple(hops))

    @staticmethod
    def _outcome_for(status: int) -> str:
        if status == 429:
            return "rate_limited"      # WE were too fast. Never folded into http_error
        if status >= 400:
            return "http_error"
        return "ok"

    def _read_capped(self, response: httpx.Response, max_bytes: float) -> bytes:
        """Read the body, dropping the connection AT the cap rather than after it."""
        declared = _declared_length(response.headers)
        if declared is not None and declared > max_bytes:
            raise _BodyTooLarge(f"the response declares {_size(declared)}, over the "
                                f"{_size(max_bytes)} limit")
        buffer = bytearray()
        for chunk in response.iter_bytes(CHUNK):
            buffer.extend(chunk)
            if len(buffer) > max_bytes:
                raise _BodyTooLarge(f"the response is over the {_size(max_bytes)} limit")
        return bytes(buffer)

    # ------------------------------------------------------------------------------- get_json
    def get_json(self, url: str, *, params: Mapping[str, Any] | None = None,
                 headers: Mapping[str, str] | None = None, timeout: float = DEFAULT_TIMEOUT,
                 accept: str = "application/json",
                 max_bytes: float = MAX_JSON_BYTES,
                 context: Mapping[str, Any] | None = None,
                 form_data: Mapping[str, Any] | None = None) -> HttpResponse:
        """One index call. Never raises: every failure comes back as an outcome and a sentence.

        `context` is accepted for the contract's sake and not used: the wire does not care what
        page of which query this is. The recorder wrapping this transport does. `form_data`
        makes it a POST (see `SearchTransport`)."""
        del context
        started = time.monotonic()
        try:
            target = str(httpx.URL(url, params=dict(params)) if params else httpx.URL(url))
        except (httpx.InvalidURL, ValueError, TypeError):
            return HttpResponse(url=str(url), outcome="refused", seconds=0.0,
                                error=f"{str(url)[:200]!r} is not a URL this server can parse")
        # what we know so far, so that EVERY failure below still records where it got to. A record
        # that says "refused" without saying which hop is a record a reviewer cannot audit.
        known: dict[str, Any] = {"url": target}
        try:
            with self._open(target, headers=headers, timeout=timeout, accept=accept,
                            allow_http_rewrite=False,
                            form_data=form_data) as (response, vetted, hops, rewritten):
                known = {"url": vetted.url, "status": response.status_code,
                         "headers": dict(response.headers), "hops": tuple(hops),
                         "rewritten_from": rewritten,
                         "retry_after": _retry_after(response.headers)}
                body = self._read_capped(response, max_bytes)
                outcome = self._outcome_for(response.status_code)
                return HttpResponse(
                    **known, outcome=outcome, body=body,
                    error="" if outcome == "ok" else self._status_sentence(
                        response.status_code, vetted.host),
                    seconds=time.monotonic() - started)
        except UrlRejected as exc:
            return HttpResponse(**{**known, "hops": exc.hops or known.get("hops", ())},
                                outcome=exc.outcome, error=str(exc),
                                seconds=time.monotonic() - started)
        except _BodyTooLarge as exc:
            return HttpResponse(**known, outcome="too_large", error=str(exc),
                                seconds=time.monotonic() - started)
        except httpx.TimeoutException:
            return HttpResponse(**known, outcome="timeout",
                                error=f"no answer within {timeout:g}s",
                                seconds=time.monotonic() - started)
        except httpx.HTTPError as exc:
            return HttpResponse(**known, outcome="network_error",
                                error=f"{exc.__class__.__name__}: {exc}"[:300],
                                seconds=time.monotonic() - started)

    @staticmethod
    def _status_sentence(status: int, host: str) -> str:
        if status == 429:
            # the wording matters: this is Canopy's fault, not the publisher's
            return (f"{host} asked us to slow down (HTTP 429) — this is our request rate, not a "
                    f"paywall; the same URL usually works a minute later")
        return f"{host} answered HTTP {status}"

    # ------------------------------------------------------------------------------- get_html
    def get_html(self, url: str, *, headers: Mapping[str, str] | None = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 max_bytes: float = MAX_HTML_BYTES) -> HttpResponse:
        """One landing page, capped, through `_open` — every hop vetted and pinned like an index
        call. The `http://` rewrite is allowed here exactly as it is for a PDF URL, because the
        landing page an index hands us is as often plain http as the PDF is."""
        started = time.monotonic()
        known: dict[str, Any] = {"url": str(url)}
        try:
            with self._open(str(url), headers=headers, timeout=timeout,
                            accept="text/html,application/xhtml+xml",
                            allow_http_rewrite=True) as (response, vetted, hops, rewritten):
                known = {"url": vetted.url, "status": response.status_code,
                         "headers": dict(response.headers), "hops": tuple(hops),
                         "rewritten_from": rewritten,
                         "retry_after": _retry_after(response.headers)}
                body = self._read_capped(response, max_bytes)
                outcome = self._outcome_for(response.status_code)
                return HttpResponse(
                    **known, outcome=outcome, body=body,
                    error="" if outcome == "ok" else self._status_sentence(
                        response.status_code, vetted.host),
                    seconds=time.monotonic() - started)
        except UrlRejected as exc:
            return HttpResponse(**{**known, "hops": exc.hops or known.get("hops", ())},
                                outcome=exc.outcome, error=str(exc),
                                seconds=time.monotonic() - started)
        except _BodyTooLarge as exc:
            return HttpResponse(**known, outcome="too_large", error=str(exc),
                                seconds=time.monotonic() - started)
        except httpx.TimeoutException:
            return HttpResponse(**known, outcome="timeout",
                                error=f"no answer within {timeout:g}s",
                                seconds=time.monotonic() - started)
        except httpx.HTTPError as exc:
            return HttpResponse(**known, outcome="network_error",
                                error=f"{exc.__class__.__name__}: {exc}"[:300],
                                seconds=time.monotonic() - started)

    # ------------------------------------------------------------------------------ get_bytes
    def get_bytes(self, url: str, dest_dir: str | Path, *, filename: str = "download.pdf",
                  headers: Mapping[str, str] | None = None,
                  timeout: float = DEFAULT_FETCH_TIMEOUT,
                  max_bytes: float = DEFAULT_MAX_PDF_BYTES,
                  accept: str = "application/pdf",
                  content_types: Iterable[str] = PDF_CONTENT_TYPES,
                  allow_http_rewrite: bool = True,
                  probe: Callable[[Path], Mapping[str, Any]] | None = None) -> Download:
        """Fetch one PDF to `<dest_dir>/<sha256>.pdf`. Never raises.

        The gate, in order, each failure a recorded outcome rather than an exception:
        the URL (`check_public`, on every hop) → the status (429 is `rate_limited`, on its own) →
        the content type → the declared length → the bytes themselves through `stream_upload`
        (magic, both caps, sha name) → the optional probe. Only then does the file get its name.
        """
        started = time.monotonic()
        allowed = {t.lower() for t in content_types}
        # S1: the allowance is taken from the shared pool ATOMICALLY, before a byte is read, and
        # handed back on the way out — three workers cannot each spend the same last 10 MB.
        with self.budget.reserve(max_bytes) as grant:
            if grant.granted <= 0:
                return Download(HttpResponse(
                    url=str(url), outcome="too_large", seconds=time.monotonic() - started,
                    error="this search has used its whole download allowance "
                          "(raise it with CANOPY_MAX_UPLOAD_TOTAL_MB)"))
            base: dict[str, Any] = {"url": str(url)}
            try:
                with self._open(str(url), headers=headers, timeout=timeout, accept=accept,
                                allow_http_rewrite=allow_http_rewrite) as (
                                    response, vetted, hops, rewritten):
                    base = dict(url=vetted.url, status=response.status_code,
                                headers=dict(response.headers), hops=tuple(hops),
                                rewritten_from=rewritten,
                                retry_after=_retry_after(response.headers))
                    outcome = self._outcome_for(response.status_code)
                    if outcome != "ok":
                        # a few bytes of the body often say WHY (Europe PMC's 429 body is the
                        # string "Rate limit exceeded"), and it costs nothing to quote it back
                        snippet = self._peek(response)
                        sentence = self._status_sentence(response.status_code, vetted.host)
                        return Download(HttpResponse(
                            **base, outcome=outcome, seconds=time.monotonic() - started,
                            error=f"{sentence}{f': {snippet}' if snippet else ''}"))

                    kind = _content_type(response.headers)
                    if kind not in allowed:
                        # this is how a login wall, a Cloudflare interstitial or a bot-block page
                        # shows up, and it is a fact about the publisher the user should see
                        return Download(HttpResponse(
                            **base, outcome="not_a_pdf", seconds=time.monotonic() - started,
                            error=f"{vetted.host} served {kind}, not a PDF"))

                    declared = _declared_length(response.headers)
                    if declared is not None and declared > grant.granted:
                        return Download(HttpResponse(
                            **base, outcome="too_large", seconds=time.monotonic() - started,
                            error=f"the file declares {_size(declared)}, over the "
                                  f"{_size(grant.granted)} left for this search"))

                    reader = _ChunkReader(response.iter_bytes(CHUNK))
                    path, n_bytes = _store(
                        reader, dest_dir, filename,
                        # both caps stay distinct so `stream_upload`'s two different 413 messages
                        # keep meaning what they say: this file is too big vs the search is full
                        max_bytes=max_bytes, remaining_bytes=grant.granted, probe=probe)
                    grant.settle(n_bytes)
                    return Download(HttpResponse(**base, outcome="ok",
                                                 seconds=time.monotonic() - started),
                                    path=path, n_bytes=n_bytes)
            except UrlRejected as exc:
                return Download(HttpResponse(
                    **{**base, "hops": exc.hops or base.get("hops", ())},
                    outcome=exc.outcome, error=str(exc), seconds=time.monotonic() - started))
            except _Rejected as exc:       # stream_upload said no, or the probe could not read it
                return Download(HttpResponse(**base, outcome=exc.outcome, error=str(exc)[:300],
                                             seconds=time.monotonic() - started))
            except httpx.TimeoutException:
                return Download(HttpResponse(**base, outcome="timeout",
                                             error=f"the download stalled for {timeout:g}s",
                                             seconds=time.monotonic() - started))
            except httpx.HTTPError as exc:
                return Download(HttpResponse(**base, outcome="network_error",
                                             error=f"{exc.__class__.__name__}: {exc}"[:300],
                                             seconds=time.monotonic() - started))

    @staticmethod
    def _peek(response: httpx.Response, limit: int = 200) -> str:
        """A short, printable quote of an error body. Never more than `limit` bytes off the wire."""
        try:
            for chunk in response.iter_bytes(limit):
                return chunk[:limit].decode("utf-8", "replace").strip().replace("\n", " ")
        except httpx.HTTPError:
            return ""
        return ""


# ------------------------------------------------------------------------------ the test double
class RecordedTransport:
    """A `SearchTransport` that only ever answers from recordings, and shouts when it cannot.

    The discipline is `canopy/llm/client.py`'s, for the same reason: a fake that improvises a
    plausible empty answer turns "the index silently returned nothing" — the exact failure this
    feature exists to prevent — into a green test. So an unrecorded request is
    `MissingSearchFixture`, loudly, with the key that would have to be recorded.
    """

    def __init__(self, responses: Mapping[str, HttpResponse] | None = None,
                 payloads: Mapping[str, str | Path] | None = None,
                 budget: ByteBudget | None = None, *, strict: bool = True) -> None:
        self.responses = dict(responses or {})
        #: fixture key → a local PDF on disk, served through the same `stream_upload` path the real
        #: transport uses, so a replayed download lands under the identical rules
        self.payloads = {k: Path(v) for k, v in (payloads or {}).items()}
        self.budget = budget if budget is not None else ByteBudget(None)
        self.calls: list[dict[str, Any]] = []
        #: a key that was asked MORE THAN ONCE in the recording (a 429 and then the retry that
        #: worked) replays in the order it happened; the last answer repeats thereafter
        self.sequences: dict[str, list[HttpResponse]] = {}
        self._cursor: dict[str, int] = {}
        #: `(index, form, query_id, page)` → `(fixture key, query text, recorded_at)`, the
        #: fallback for a request whose exact text was never recorded. Filled by `from_dir`.
        self.tuples: dict[tuple[str, str, str, int], tuple[str, str, str]] = {}
        #: how many requests were answered through that fallback, and which — the number the
        #: bench prints beside every measurement taken offline against a changed query string
        self.query_drift = 0
        self.drift_log: list[dict[str, Any]] = []
        #: `strict=False` turns an unrecorded request into a `refused` outcome instead of an
        #: exception — for the bench, where one missing recording must cost one row of the
        #: record and not the whole fetch stage. Tests keep the default and the loud failure.
        self.strict = bool(strict)
        self.unrecorded: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, url: str, response: HttpResponse,
               params: Mapping[str, Any] | None = None) -> None:
        self.responses[fixture_key(url, params)] = response

    @classmethod
    def from_dir(cls, directory: str | Path, *, budget: ByteBudget | None = None,
                 strict: bool = True) -> RecordedTransport:
        """Every fixture `RecordingTransport` wrote under `directory`, ready to replay.

        `<key>.json.gz` files become recorded responses (a key recorded several times becomes a
        sequence, oldest first); `pdfs.manifest.json` becomes recorded downloads, served from
        `pdfs/<key>.pdf` when the file is there and as a `refused` outcome naming the missing
        file when it is not (the PDFs are gitignored; the manifest is not).
        """
        root = Path(directory)
        transport = cls(budget=budget, strict=strict)
        for path in sorted(root.glob("*.json.gz")):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    fixture = json.load(handle)
            except (OSError, ValueError):
                continue
            key = str(fixture.get("key") or path.name[: -len(".json.gz")])
            answers = [_response_from_fixture(row) for row in _fixture_rows(fixture)]
            if not answers:
                continue
            transport.responses[key] = answers[-1]
            if len(answers) > 1:
                transport.sequences[key] = answers
            last = _fixture_rows(fixture)[-1]
            tuple_key = _tuple_of(last)
            if tuple_key is not None:
                # the newest recording for a tuple wins: a query iterated on three times has
                # three fixtures for page 1 of Q1, and the one to fall back to is the latest
                stamp = str(last.get("recorded_at") or fixture.get("recorded_at") or "")
                kept = transport.tuples.get(tuple_key)
                if kept is None or stamp >= kept[2]:
                    transport.tuples[tuple_key] = (key, str(last.get("query_text") or ""), stamp)
        manifest = root / "pdfs.manifest.json"
        if manifest.is_file():
            try:
                rows = json.loads(manifest.read_text(encoding="utf-8"))
            except ValueError:
                rows = {}
            for key, entry in (rows or {}).items():
                attempts = list((entry or {}).get("attempts") or [])
                if not attempts:
                    continue
                answers = []
                for attempt in attempts:
                    response = _response_from_fixture(attempt)
                    if response.outcome == "ok":
                        pdf = root / "pdfs" / f"{key}.pdf"
                        if pdf.is_file():
                            transport.payloads[key] = pdf
                        else:
                            response = HttpResponse(
                                url=response.url, status=response.status, outcome="refused",
                                error=f"the recorded PDF {pdf.name} is not on disk (pdfs/ is "
                                      f"not committed) — replay from the machine that "
                                      f"recorded it, or record again")
                    answers.append(response)
                transport.responses[key] = answers[-1]
                if len(answers) > 1:
                    transport.sequences[key] = answers
        return transport

    def _next(self, key: str) -> HttpResponse:
        """The recorded answer — the next one of a sequence when the key was asked repeatedly."""
        sequence = self.sequences.get(key)
        if not sequence:
            return self.responses[key]
        with self._lock:
            position = self._cursor.get(key, 0)
            self._cursor[key] = position + 1
        return sequence[min(position, len(sequence) - 1)]

    def _refusal(self, url: str, key: str, method: str) -> HttpResponse:
        """What an unrecorded request gets under `strict=False`: a refusal that names itself."""
        row = {"method": method, "url": url, "key": key}
        with self._lock:
            self.unrecorded.append(row)
        return HttpResponse(url=url, outcome="refused",
                            error=f"no recording for this request ({key}) — this replay is "
                                  f"offline, and nothing was sent")

    def get_json(self, url: str, *, params: Mapping[str, Any] | None = None,
                 headers: Mapping[str, str] | None = None,
                 timeout: float = DEFAULT_TIMEOUT, accept: str = "application/json",
                 max_bytes: float = MAX_JSON_BYTES,
                 context: Mapping[str, Any] | None = None,
                 form_data: Mapping[str, Any] | None = None) -> HttpResponse:
        """The recorded answer for this URL. Signature-identical to the real one on purpose —
        see `SearchTransport`. There is no `**kwargs`: a keyword the real transport does not have
        must fail HERE, in a test, and not in the one place there are no tests.

        A miss falls back to the `(index, form, query_id, page)` tuple in `context` when one was
        recorded — and COUNTS it (`query_drift`), because an answer to a slightly different query
        is a measurement with an asterisk, and the bench prints the asterisk.
        """
        key = fixture_key(url, {**dict(params or {}), **dict(form_data or {})})
        self.calls.append({"method": "get_json", "url": url,
                           "params": {**dict(params or {}), **dict(form_data or {})},
                           "key": key, "context": dict(context or {})})
        if key in self.responses:
            return self._next(key)
        tuple_key = _tuple_of(context) if context and not context.get("exact") else None
        if tuple_key is not None and tuple_key in self.tuples:
            recorded_key, recorded_text = self.tuples[tuple_key][:2]
            if recorded_key in self.responses:
                with self._lock:
                    self.query_drift += 1
                    self.drift_log.append({"tuple": list(tuple_key), "recorded_key": recorded_key,
                                           "recorded_query": recorded_text,
                                           "asked_query": str(context.get("query_text") or "")})
                return self._next(recorded_key)
        if not self.strict:
            return self._refusal(url, key, "get_json")
        raise MissingSearchFixture(
            f"no recorded response for {key} ({url}) — record one with "
            f"CANOPY_SEARCH_RECORD=1, or hand the test a RecordedTransport that has it")

    def get_html(self, url: str, *, headers: Mapping[str, str] | None = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 max_bytes: float = MAX_HTML_BYTES) -> HttpResponse:
        """The recorded landing page. Keyed apart from a download of the same URL (`-html`), because
        the fetch stage asks for the bytes first and the page second, and both are recorded."""
        key = html_key(url)
        self.calls.append({"method": "get_html", "url": url, "key": key})
        if key in self.responses:
            return self._next(key)
        if not self.strict:
            return self._refusal(url, key, "get_html")
        raise MissingSearchFixture(
            f"no recorded landing page for {key} ({url}) — record one with "
            f"CANOPY_SEARCH_RECORD=1, or hand the test a RecordedTransport that has it")

    def get_bytes(self, url: str, dest_dir: str | Path, *, filename: str = "download.pdf",
                  headers: Mapping[str, str] | None = None,
                  timeout: float = DEFAULT_FETCH_TIMEOUT,
                  max_bytes: float = DEFAULT_MAX_PDF_BYTES,
                  accept: str = "application/pdf",
                  content_types: Iterable[str] = PDF_CONTENT_TYPES,
                  allow_http_rewrite: bool = True,
                  probe: Callable[[Path], Mapping[str, Any]] | None = None) -> Download:
        """The recorded PDF for this URL, stored through the real `_store`.

        `accept`, `content_types` and `allow_http_rewrite` are accepted and RECORDED rather than
        honoured: a replayed payload is already known to be a PDF, and there is no wire to
        rewrite. They are here because a fake that quietly swallowed them let the two signatures
        drift apart — see `SearchTransport`.
        """
        key = fixture_key(url)
        self.calls.append({"method": "get_bytes", "url": url, "key": key, "accept": accept,
                           "content_types": sorted(str(t) for t in content_types),
                           "allow_http_rewrite": bool(allow_http_rewrite)})
        recorded = self._next(key) if key in self.responses else None
        if recorded is not None and recorded.outcome != "ok":
            return Download(recorded)          # a recorded 429/403 replays as itself
        if key not in self.payloads:
            if not self.strict:
                return Download(self._refusal(url, key, "get_bytes"))
            raise MissingSearchFixture(
                f"no recorded PDF for {key} ({url}) — add one to `payloads`, or record the "
                f"failure it should replay instead")
        with self.budget.reserve(max_bytes) as grant:
            if grant.granted <= 0:
                return Download(HttpResponse(url=url, outcome="too_large",
                                             error="this search has used its whole allowance"))
            with self.payloads[key].open("rb") as handle:
                reader = _ChunkReader(iter(lambda: handle.read(CHUNK), b""))
                try:
                    path, n_bytes = _store(reader, dest_dir, filename, max_bytes=max_bytes,
                                           remaining_bytes=grant.granted, probe=probe)
                except _Rejected as exc:
                    return Download(HttpResponse(url=url, outcome=exc.outcome,
                                                 error=str(exc)[:300]))
            grant.settle(n_bytes)
        return Download(recorded or HttpResponse(url=url, status=200, outcome="ok"),
                        path=path, n_bytes=n_bytes)


# ------------------------------------------------------------------------------------ the recorder
#: the fields of one recorded answer. `body_b64` because a JSON body is bytes off the wire and a
#: fixture must replay exactly what arrived, not what `json.loads` made of it.
_FIXTURE_FIELDS = ("status", "outcome", "error", "retry_after", "seconds", "url", "hops")


def _tuple_of(context: Mapping[str, Any] | None) -> tuple[str, str, str, int] | None:
    """`(index, form, query_id, page)` out of a context, or None when it does not name one."""
    if not context or not context.get("index") or not context.get("query_id"):
        return None
    try:
        page = int(context.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    return (str(context["index"]), str(context.get("form") or ""), str(context["query_id"]),
            page)


def _fixture_rows(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The recorded answers in a fixture file, oldest first. A file holds one (`{...}`) or, when
    the same request was made again in one recording, several (`{"responses": [...]}`)."""
    rows = fixture.get("responses")
    if isinstance(rows, list) and rows:
        return [dict(r) for r in rows if isinstance(r, Mapping)]
    return [dict(fixture)]


def _response_from_fixture(row: Mapping[str, Any]) -> HttpResponse:
    body = base64.b64decode(str(row.get("body_b64") or "")) if row.get("body_b64") else b""
    retry_after = row.get("retry_after")
    return HttpResponse(url=str(row.get("url") or ""), status=int(row.get("status") or 0),
                        outcome=str(row.get("outcome") or "ok"), body=body,
                        error=str(row.get("error") or ""),
                        seconds=float(row.get("seconds") or 0.0),
                        hops=tuple(str(h) for h in (row.get("hops") or ())),
                        retry_after=float(retry_after) if retry_after is not None else None)


def _fixture_row(response: HttpResponse, *, context: Mapping[str, Any] | None,
                 url: str, params: Mapping[str, Any] | None, body: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": str((context or {}).get("index") or ""),
        "form": str((context or {}).get("form") or ""),
        "query_id": str((context or {}).get("query_id") or ""),
        "page": int((context or {}).get("page") or 0),
        "query_text": str((context or {}).get("query_text") or ""),
        "request_url": str(url), "params": redact_params(params),
        "url": response.url, "status": int(response.status), "outcome": response.outcome,
        "error": response.error, "retry_after": response.retry_after,
        "seconds": round(float(response.seconds), 3), "hops": list(response.hops),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if body:
        row["body_b64"] = base64.b64encode(response.body).decode("ascii")
    return row


class RecordingTransport:
    """The real transport, with every answer written down so the next run needs no network.

    Wraps any `SearchTransport` (`inner`) and persists what it returns under `directory`:

    * `get_json` → `<key>.json.gz`, the whole response plus the caller's `context` tuple and the
      request (with `IDENTITY_PARAMS` removed — a committed fixture must never carry a key or an
      address). The same key asked twice in one recording (a 429, then the retry) becomes a
      `responses` list, replayed in order.
    * `get_bytes` → the outcome into `pdfs.manifest.json` and, when a PDF landed, a copy of it at
      `pdfs/<key>.pdf` with its sha256 in the manifest. The PDFs are not committed; the manifest
      is, so a reader without the bytes still knows what was fetched and from where.

    `RecordedTransport.from_dir` reads all of it back. This class does nothing else: no caching,
    no dedupe, no judgement — the inner transport already made every decision that matters, and
    this only remembers what it said.
    """

    def __init__(self, inner: SearchTransport, directory: str | Path) -> None:
        self.inner = inner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "pdfs").mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self.n_recorded = 0
        self.calls: list[dict[str, Any]] = []

    # the budget is the inner transport's: `searches.py` reads it off the object it built
    @property
    def budget(self) -> ByteBudget | None:
        return getattr(self.inner, "budget", None)

    def _write_json(self, key: str, row: dict[str, Any]) -> None:
        path = self.directory / f"{key}.json.gz"
        with self._lock:
            rows: list[dict[str, Any]] = []
            if path.is_file():
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        rows = _fixture_rows(json.load(handle))
                except (OSError, ValueError):
                    rows = []
            rows.append(row)
            payload: dict[str, Any] = {"key": key, "recorded_at": row["recorded_at"]}
            if len(rows) == 1:
                payload.update(rows[0])
            else:
                payload["responses"] = rows
            tmp = path.with_suffix(".gz.part")
            with gzip.open(tmp, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            tmp.replace(path)
            self.n_recorded += 1

    def get_json(self, url: str, *, params: Mapping[str, Any] | None = None,
                 headers: Mapping[str, str] | None = None, timeout: float = DEFAULT_TIMEOUT,
                 accept: str = "application/json", max_bytes: float = MAX_JSON_BYTES,
                 context: Mapping[str, Any] | None = None,
                 form_data: Mapping[str, Any] | None = None) -> HttpResponse:
        response = self.inner.get_json(url, params=params, headers=headers, timeout=timeout,
                                       accept=accept, max_bytes=max_bytes, context=context,
                                       form_data=form_data)
        every = {**dict(params or {}), **dict(form_data or {})}
        key = fixture_key(url, every)
        self.calls.append({"method": "get_json", "url": url, "key": key,
                           "context": dict(context or {}), "outcome": response.outcome})
        row = _fixture_row(response, context=context, url=url, params=every, body=True)
        if form_data is not None:
            row["method"] = "POST"
        self._write_json(key, row)
        return response

    def get_html(self, url: str, *, headers: Mapping[str, str] | None = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 max_bytes: float = MAX_HTML_BYTES) -> HttpResponse:
        response = self.inner.get_html(url, headers=headers, timeout=timeout, max_bytes=max_bytes)
        key = html_key(url)
        self.calls.append({"method": "get_html", "url": url, "key": key,
                           "outcome": response.outcome})
        self._write_json(key, _fixture_row(response, context={"form": "html"}, url=url,
                                           params=None, body=True))
        return response

    def get_bytes(self, url: str, dest_dir: str | Path, *, filename: str = "download.pdf",
                  headers: Mapping[str, str] | None = None,
                  timeout: float = DEFAULT_FETCH_TIMEOUT,
                  max_bytes: float = DEFAULT_MAX_PDF_BYTES,
                  accept: str = "application/pdf",
                  content_types: Iterable[str] = PDF_CONTENT_TYPES,
                  allow_http_rewrite: bool = True,
                  probe: Callable[[Path], Mapping[str, Any]] | None = None) -> Download:
        download = self.inner.get_bytes(url, dest_dir, filename=filename, headers=headers,
                                        timeout=timeout, max_bytes=max_bytes, accept=accept,
                                        content_types=content_types,
                                        allow_http_rewrite=allow_http_rewrite, probe=probe)
        key = fixture_key(url)
        self.calls.append({"method": "get_bytes", "url": url, "key": key,
                           "outcome": download.response.outcome})
        attempt = _fixture_row(download.response, context=None, url=url, params=None, body=False)
        attempt["n_bytes"] = int(download.n_bytes)
        if download.ok and download.path is not None:
            import hashlib
            import shutil

            target = self.directory / "pdfs" / f"{key}.pdf"
            with self._lock:
                shutil.copyfile(download.path, target)
            attempt["sha256"] = hashlib.sha256(Path(download.path).read_bytes()).hexdigest()
            attempt["file"] = f"pdfs/{key}.pdf"
        manifest = self.directory / "pdfs.manifest.json"
        with self._lock:
            rows: dict[str, Any] = {}
            if manifest.is_file():
                try:
                    rows = json.loads(manifest.read_text(encoding="utf-8")) or {}
                except ValueError:
                    rows = {}
            entry = rows.setdefault(key, {"url": str(url), "attempts": []})
            entry["attempts"].append(attempt)
            tmp = manifest.with_suffix(".json.part")
            tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(manifest)
            self.n_recorded += 1
        return download
