"""The SSRF gate and the fetch envelope — pinned, with no network of any kind.

Not one test here opens a socket or asks a DNS server anything. Names are answered by a table the
test hands to `check_public`; the HTTP side is an `httpx.MockTransport` handler. A test that needs
DNS is a test that fails on a plane, and a security predicate whose tests only run when the wifi is
up is a predicate that stops being run.

The one place real system code is exercised is the numeric-spelling test: `getaddrinfo` with
`AI_NUMERICHOST` never consults a resolver, so `http://2130706433/` can be proved to reach
127.0.0.1 by the actual C library, offline.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest

from canopy.search.transport import (CHUNK, DEFAULT_HOST_INTERVAL, HOST_INTERVALS, ByteBudget,
                                     HostClock, HttpResponse, HttpxTransport, MissingSearchFixture,
                                     RecordedTransport, UrlRejected, check_public, fixture_key,
                                     is_public_address, is_safe_public_host, pinned_url)

PUBLIC_IP = "93.184.216.34"
PDF_BYTES = b"%PDF-1.4\n" + b"paper" * 200


@pytest.fixture(autouse=True)
def _no_dangerous_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both escape hatches off, whatever the shell says. `CANOPY_SEARCH_ALLOW_PRIVATE_HOSTS` in
    particular is a switch that turns the whole module into a no-op, and a test suite that
    accidentally inherits it would go green while proving nothing."""
    monkeypatch.delenv("CANOPY_SEARCH_ALLOW_PRIVATE_HOSTS", raising=False)
    monkeypatch.setenv("CANOPY_SEARCH_PIN_ADDRESS", "1")


# ---------------------------------------------------------------------------- fakes, not mocks
#: the name table every test resolves against. No DNS server is involved in any of it.
NAMES: dict[str, tuple[str, ...]] = {
    "good.example": (PUBLIC_IP,),
    "also-good.example": ("8.8.8.8",),
    "evil.example": ("10.0.0.5",),
    "mixed.example": ("8.8.8.8", "127.0.0.1"),
    "metadata.example": ("169.254.169.254",),
    "v6.example": ("2001:4860:4860::8888",),
}


def table_resolver(host: str, port: int = 443) -> tuple[str, ...]:
    """Names from the table; an address literal resolves to itself, exactly as libc does.

    The literal branch is not decoration: a redirect whose `Location` is `https://127.0.0.1/` is
    the attack this module exists for, and a fake that answered "no such host" to it would have
    turned that test green for the wrong reason.
    """
    try:
        return (ipaddress.ip_address(host.strip("[]")).compressed,)
    except ValueError:
        pass
    if host not in NAMES:
        raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided")
    return NAMES[host]


class Wire:
    """A recorded conversation: what the transport asked for, and what it was told.

    `requests` is the assertion surface for the rules that are about what is NOT sent — a refused
    redirect must leave exactly one entry here.
    """

    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request, len(self.requests))

    def transport(self, **kwargs) -> HttpxTransport:
        client = httpx.Client(transport=httpx.MockTransport(self), follow_redirects=False,
                              trust_env=False)
        kwargs.setdefault("clock", HostClock(default=0.0))     # courtesy is tested on its own
        return HttpxTransport(client=client, resolve=table_resolver, **kwargs)


def wire_of(handler) -> Wire:
    return Wire(handler)


def ok_pdf(_request: httpx.Request, _n: int) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES)


# =============================================================================== the address gate
#: every address the brief names, plus the two the review proved a single predicate misses
ADVERSARIAL = [
    "127.0.0.1",                 # loopback
    "10.0.0.5",                  # private
    "172.16.0.1",                # private
    "192.168.1.1",               # private
    "169.254.169.254",           # cloud metadata — the reason this module exists
    "169.254.0.1",               # link-local
    "100.64.0.1",                # CGNAT: is_private says False, so is_private alone is not enough
    "0.0.0.0",                   # unspecified
    "224.0.0.1",                 # multicast: is_global says True
    "240.0.0.1",                 # reserved
    "255.255.255.255",           # broadcast
    "192.0.0.1",                 # IETF protocol assignments
    "198.18.0.1",                # benchmarking
    "::1",                       # v6 loopback
    "::",                        # v6 unspecified
    "::ffff:127.0.0.1",          # v4-mapped loopback in disguise
    "::ffff:169.254.169.254",    # v4-mapped metadata
    "64:ff9b::7f00:1",           # NAT64 → 127.0.0.1: is_global says True, so it alone is not enough
    "64:ff9b:1::7f00:1",         # local-use NAT64
    "2002:7f00:1::",             # 6to4 wrapping 127.0.0.1
    "2001::1",                   # Teredo
    "2001:db8::1",               # documentation
    "fd00::1",                   # unique local
    "fe80::1",                   # v6 link-local
    "ff02::1",                   # v6 multicast
]

PUBLIC = ["8.8.8.8", PUBLIC_IP, "1.1.1.1", "100.63.255.255", "2001:4860:4860::8888"]


@pytest.mark.parametrize("literal", ADVERSARIAL)
def test_every_adversarial_address_is_refused(literal: str) -> None:
    assert is_public_address(ipaddress.ip_address(literal)) is False


@pytest.mark.parametrize("literal", PUBLIC)
def test_a_genuinely_public_address_is_allowed(literal: str) -> None:
    assert is_public_address(ipaddress.ip_address(literal)) is True


@pytest.mark.parametrize("literal", ADVERSARIAL)
def test_a_name_resolving_to_an_adversarial_address_is_refused(literal: str) -> None:
    """The predicate is not enough on its own: the gate must apply it to whatever DNS answers."""
    ok, why = is_safe_public_host("attacker.example",
                                  resolve=lambda host, port: (literal,))
    assert ok is False
    # the refusal must name the address, or "we would not fetch attacker.example" is unactionable
    assert ipaddress.ip_address(literal).compressed in why


def test_a_url_naming_an_adversarial_address_directly_is_refused() -> None:
    with pytest.raises(UrlRejected) as caught:
        check_public("https://169.254.169.254/latest/meta-data/", resolve=table_resolver)
    assert "169.254.169.254" in str(caught.value)


def test_a_name_with_one_private_answer_among_public_ones_is_refused() -> None:
    """EVERY answer must be public: the connection would have taken whichever it liked."""
    ok, why = is_safe_public_host("mixed.example", resolve=table_resolver)
    assert ok is False and "127.0.0.1" in why


def test_a_public_name_is_allowed_and_carries_its_addresses() -> None:
    vetted = check_public("https://good.example/a?x=1", resolve=table_resolver)
    assert (vetted.host, vetted.address, vetted.addresses) == ("good.example", PUBLIC_IP,
                                                               (PUBLIC_IP,))
    assert pinned_url(vetted) == f"https://{PUBLIC_IP}/a?x=1"


def test_a_name_that_does_not_resolve_is_a_dns_error_not_a_refusal() -> None:
    """Different facts: a typo and an attack should not read the same in the record."""
    with pytest.raises(UrlRejected) as caught:
        check_public("https://nowhere.example/x", resolve=table_resolver)
    assert caught.value.outcome == "dns_error"
    ok, why = is_safe_public_host("nowhere.example", resolve=table_resolver)
    assert ok is False and "does not resolve" in why


def test_an_ipv6_answer_with_a_scope_id_is_refused_not_crashed() -> None:
    """`fe80::1%en0` cannot be parsed by `ipaddress`; it must refuse, never raise."""
    ok, why = is_safe_public_host("scoped.example", resolve=lambda h, p: ("fe80::1%en0",))
    assert ok is False and "fe80::1" in why


# ------------------------------------------------------------------ decimal / octal / hex hosts
#: measured on this machine (CPython 3.13.11, macOS) with `socket.getaddrinfo(host, 443,
#: flags=AI_NUMERICHOST)` — which parses locally and never consults a resolver
NUMERIC_LOOPBACK = ["2130706433", "017700000001", "0x7f.1", "127.1", "0x7f000001"]


def numeric_resolver(host: str, port: int = 443) -> tuple[str, ...]:
    """Real libc parsing with zero network: AI_NUMERICHOST refuses to look anything up."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_NUMERICHOST)
    return tuple(dict.fromkeys(str(info[4][0]) for info in infos))


@pytest.mark.parametrize("spelling", NUMERIC_LOOPBACK)
def test_a_numeric_spelling_of_loopback_is_refused(spelling: str) -> None:
    """`http://2130706433/` is 127.0.0.1 and the gate must see that."""
    assert numeric_resolver(spelling) == ("127.0.0.1",)          # the C library, offline
    ok, why = is_safe_public_host(spelling, resolve=numeric_resolver)
    assert ok is False and "127.0.0.1" in why
    with pytest.raises(UrlRejected):
        check_public(f"https://{spelling}/x", resolve=numeric_resolver)


def test_the_host_is_resolved_as_given_and_never_parsed_as_an_address_first() -> None:
    """The two parsers genuinely disagree, so only the one that will connect gets a vote.

    `ipaddress` refuses `0177.0.0.1` outright while `getaddrinfo` answers with an address — so a
    gate that tried `ipaddress.ip_address(host)` first and fell back to DNS would be judging a
    different string from the one the socket uses.
    """
    with pytest.raises(ValueError):
        ipaddress.ip_address("0177.0.0.1")
    assert numeric_resolver("0177.0.0.1")            # the C library has an opinion regardless

    seen: list[str] = []

    def spy(host: str, port: int = 443) -> tuple[str, ...]:
        seen.append(host)
        return ("127.0.0.1",)

    with pytest.raises(UrlRejected):
        check_public("https://2130706433/x", resolve=spy)
    assert seen == ["2130706433"]                    # verbatim, unparsed, un-normalised


def test_the_private_host_allow_list_is_off_unless_it_is_set(monkeypatch: pytest.MonkeyPatch
                                                             ) -> None:
    """It exists for tests that need a loopback fixture server, and it is dangerous, so pin both
    that it is off by default and that it never loosens the scheme or port rules."""
    with pytest.raises(UrlRejected):
        check_public("https://localhost.test/x", resolve=lambda h, p: ("127.0.0.1",))
    monkeypatch.setenv("CANOPY_SEARCH_ALLOW_PRIVATE_HOSTS", "localhost.test")
    assert check_public("https://localhost.test/x",
                        resolve=lambda h, p: ("127.0.0.1",)).host == "localhost.test"
    with pytest.raises(UrlRejected):                 # still https-only
        check_public("http://localhost.test/x", resolve=lambda h, p: ("127.0.0.1",))
    with pytest.raises(UrlRejected):                 # still port 443 only
        check_public("https://localhost.test:8080/x", resolve=lambda h, p: ("127.0.0.1",))


# ====================================================================== scheme / port / userinfo
@pytest.mark.parametrize("url", [
    "http://good.example/paper.pdf",
    "file:///etc/passwd",
    "ftp://good.example/paper.pdf",
    "data:application/pdf;base64,JVBERi0=",
    "gopher://good.example/1",
])
def test_only_https_is_fetched(url: str) -> None:
    with pytest.raises(UrlRejected) as caught:
        check_public(url, resolve=table_resolver)
    assert "https" in str(caught.value)


def test_credentials_in_a_url_are_refused() -> None:
    with pytest.raises(UrlRejected) as caught:
        check_public("https://user:secret@good.example/x", resolve=table_resolver)
    message = str(caught.value)
    assert "password" in message and "secret" not in message   # never echo the credential


@pytest.mark.parametrize("port", [8080, 8443, 80, 22])
def test_a_port_other_than_443_is_refused_and_the_refusal_names_it(port: int) -> None:
    """Policy, not an accident — and it costs real recall, so the record must say WHICH port."""
    with pytest.raises(UrlRejected) as caught:
        check_public(f"https://good.example:{port}/x", resolve=table_resolver)
    assert str(port) in str(caught.value)


def test_an_explicit_port_443_is_the_default_and_is_allowed() -> None:
    assert check_public("https://good.example:443/x", resolve=table_resolver).host == "good.example"


# ============================================================================ the request itself
def test_the_request_goes_to_the_vetted_address_with_the_name_in_host_and_sni() -> None:
    """DNS must not get a second vote between the check and the connection."""
    wire = wire_of(lambda request, n: httpx.Response(200, json={"ok": True}))
    result = wire.transport().get_json("https://good.example/search", params={"q": "tremor"})

    assert result.outcome == "ok" and result.json() == {"ok": True}
    sent = wire.requests[0]
    assert sent.url.host == PUBLIC_IP                       # the socket goes to the vetted address
    assert sent.headers["host"] == "good.example"           # …the name rides in the header…
    assert sent.extensions["sni_hostname"] == "good.example"  # …and in the TLS handshake
    assert sent.url.params["q"] == "tremor"


def test_an_ipv6_host_is_pinned_too() -> None:
    """The bracketing is httpx's job, but an IPv6-only publisher must not silently go unpinned."""
    wire = wire_of(lambda request, n: httpx.Response(200, json={}))
    assert wire.transport().get_json("https://v6.example/x").outcome == "ok"
    sent = wire.requests[0]
    assert sent.url.host == "2001:4860:4860::8888"
    assert sent.headers["host"] == "v6.example"
    assert sent.extensions["sni_hostname"] == "v6.example"


def test_pinning_can_be_turned_off_deliberately() -> None:
    """A real flag, not an undesigned escape hatch — the day a SNI-routing CDN needs it."""
    wire = wire_of(lambda request, n: httpx.Response(200, json={}))
    wire.transport(pin_address=False).get_json("https://good.example/search")
    assert wire.requests[0].url.host == "good.example"


def test_the_default_client_never_reuses_a_connection() -> None:
    """A consequence of the pinning, not ordinary caution.

    httpcore matches a pooled connection with `origin == self._origin`, and `origin` is
    (scheme, host, port) read off the REQUEST url — which, because we pin, is the IP address. The
    SNI name is in `extensions` and is NOT part of that key. So two vetted names sharing one CDN
    address would share one TLS session and the second request would ride on a certificate
    verified only for the first name. This pokes at httpx internals on purpose: it is the only way
    to pin the invariant without opening a socket, and it will fail loudly if an upgrade moves it.
    """
    pool = HttpxTransport(resolve=table_resolver).client._transport._pool
    assert pool._max_keepalive_connections == 0

    origin = httpx.Client().build_request("GET", "https://93.184.216.34/x").url
    assert origin.host == "93.184.216.34"      # what httpcore keys on: the address, not the name


def test_a_client_that_follows_redirects_is_refused_at_construction() -> None:
    """The invariant that makes every other redirect rule work."""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)),
                          follow_redirects=True)
    with pytest.raises(ValueError, match="follow_redirects"):
        HttpxTransport(client=client, resolve=table_resolver)


# ==================================================================================== redirects
def redirect_to(location: str):
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if n == 1:
            return httpx.Response(302, headers={"location": location})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=PDF_BYTES)
    return handler


def test_a_second_hop_pointing_at_loopback_is_refused_and_never_requested(tmp_path: Path) -> None:
    """The mitigation that actually matters: hop 1 is a real public host, hop 2 is not."""
    wire = wire_of(redirect_to("https://127.0.0.1/latest/meta-data/"))
    result = wire.transport().get_bytes("https://good.example/paper.pdf", tmp_path)

    assert result.response.outcome == "refused"
    assert "127.0.0.1" in result.response.error
    # the record must name the hop: "we refused 127.0.0.1" alone never tells the user that a URL
    # they trusted is the thing that sent us there
    assert "redirect from https://good.example/paper.pdf" in result.response.error
    assert len(wire.requests) == 1                 # the second hop was never put on the wire
    assert result.path is None and list(tmp_path.iterdir()) == []


def test_a_second_hop_pointing_at_the_metadata_service_is_refused() -> None:
    wire = wire_of(redirect_to("https://169.254.169.254/latest/meta-data/iam/"))
    result = wire.transport().get_json("https://good.example/x")
    assert result.outcome == "refused" and "169.254.169.254" in result.error
    assert len(wire.requests) == 1


def test_a_redirect_to_http_is_refused_even_though_the_first_url_may_be_upgraded() -> None:
    """The one-shot http→https rewrite applies to the FIRST url only: a 302 to http:// is the hop
    an attacker controls."""
    wire = wire_of(redirect_to("http://good.example/paper.pdf"))
    result = wire.transport().get_json("https://good.example/x")
    assert result.outcome == "refused" and "https" in result.error
    assert len(wire.requests) == 1


def test_a_relative_redirect_resolves_against_the_hop_that_sent_it() -> None:
    wire = wire_of(redirect_to("../c/paper.pdf"))
    result = wire.transport().get_json("https://good.example/a/b/x")

    assert result.outcome == "ok"
    second = wire.requests[1]
    assert second.url.path == "/a/c/paper.pdf"
    assert second.url.host == PUBLIC_IP and second.headers["host"] == "good.example"
    assert result.hops == ("https://good.example/a/b/x", "https://good.example/a/c/paper.pdf")


def test_a_redirect_that_leaves_the_first_host_is_re_vetted_not_trusted() -> None:
    wire = wire_of(redirect_to("https://also-good.example/paper.pdf"))
    result = wire.transport().get_json("https://good.example/x")
    assert result.outcome == "ok"
    assert wire.requests[1].url.host == "8.8.8.8"      # vetted afresh, pinned afresh


def test_a_redirect_loop_is_stopped() -> None:
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        target = "https://good.example/b" if n == 1 else "https://good.example/a"
        return httpx.Response(302, headers={"location": target})

    result = wire_of(handler).transport().get_json("https://good.example/a")
    assert result.outcome == "http_error" and "loop" in result.error


def test_a_redirect_chain_stops_at_the_hop_cap() -> None:
    wire = wire_of(lambda request, n: httpx.Response(302, headers={"location": f"/hop{n}"}))
    result = wire.transport(max_redirects=5).get_json("https://good.example/a")
    assert result.outcome == "http_error" and "more than 5 redirects" in result.error
    assert len(wire.requests) == 6                # the first request plus five followed hops


def test_a_redirect_without_a_location_is_an_error_not_a_hang() -> None:
    wire = wire_of(lambda request, n: httpx.Response(302))
    result = wire.transport().get_json("https://good.example/a")
    assert result.outcome == "http_error" and "Location" in result.error


# =================================================================================== statuses
def test_a_429_is_rate_limited_and_never_http_error() -> None:
    """The whole point: a 429 is Canopy's fault, and telling the user "the publisher refused"
    when we hammered the index is a lie the record would carry forever."""
    wire = wire_of(lambda request, n: httpx.Response(429, content=b"Rate limit exceeded"))
    result = wire.transport().get_json("https://good.example/x")

    assert result.outcome == "rate_limited"
    assert result.status == 429
    assert result.retry_after is None              # Europe PMC's 429 carries no Retry-After
    assert "slow down" in result.error and "paywall" in result.error


def test_a_429_on_a_download_is_rate_limited_and_quotes_the_server(tmp_path: Path) -> None:
    wire = wire_of(lambda request, n: httpx.Response(
        429, headers={"retry-after": "30"}, content=b"Rate limit exceeded"))
    result = wire.transport().get_bytes("https://good.example/paper.pdf", tmp_path)

    assert result.response.outcome == "rate_limited"
    assert result.response.retry_after == 30.0
    assert "Rate limit exceeded" in result.response.error
    assert result.path is None and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", [403, 404, 500, 503])
def test_an_index_that_is_down_or_says_no_is_data_not_an_exception(status: int) -> None:
    wire = wire_of(lambda request, n: httpx.Response(status))
    result = wire.transport().get_json("https://good.example/x")
    assert result.outcome == "http_error" and result.status == status
    assert str(status) in result.error


def test_a_timeout_is_data_not_an_exception() -> None:
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    result = wire_of(handler).transport().get_json("https://good.example/x", timeout=1.0)
    assert result.outcome == "timeout" and result.status == 0


def test_a_connection_failure_is_data_not_an_exception() -> None:
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    result = wire_of(handler).transport().get_json("https://good.example/x")
    assert result.outcome == "network_error" and "ConnectError" in result.error


def test_a_json_body_over_its_cap_is_refused_mid_read() -> None:
    """An index that answers with an endless `{` costs one recorded error, not the machine."""
    wire = wire_of(lambda request, n: httpx.Response(
        200, headers={"content-type": "application/json"},
        content=(b"x" * CHUNK for _ in range(100))))
    result = wire.transport().get_json("https://good.example/x", max_bytes=2 * CHUNK)
    assert result.outcome == "too_large"


# ============================================================== the download gate, in the order
def test_a_pdf_is_written_under_its_sha256_by_stream_upload(tmp_path: Path) -> None:
    """Identical rules to a hand upload, because it is literally the same function."""
    import hashlib

    result = wire_of(ok_pdf).transport().get_bytes("https://good.example/paper.pdf", tmp_path)

    assert result.ok and result.n_bytes == len(PDF_BYTES)
    assert result.path is not None
    assert result.path.name == f"{hashlib.sha256(PDF_BYTES).hexdigest()}.pdf"
    assert result.path.read_bytes() == PDF_BYTES
    assert [p.name for p in tmp_path.iterdir()] == [result.path.name]   # no staging left behind


@pytest.mark.parametrize("content_type", ["text/html", "text/plain", "application/json"])
def test_a_bot_block_page_is_not_accepted_as_a_pdf(content_type: str, tmp_path: Path) -> None:
    """About half of publisher PDF links serve an interstitial; that is a fact about the
    publisher the user should see, not a mystery."""
    wire = wire_of(lambda request, n: httpx.Response(
        200, headers={"content-type": content_type}, content=b"<html>Please log in</html>"))
    result = wire.transport().get_bytes("https://good.example/paper.pdf", tmp_path)

    assert result.response.outcome == "not_a_pdf" and content_type in result.response.error
    assert list(tmp_path.iterdir()) == []


def test_a_body_that_lies_about_being_a_pdf_is_caught_by_the_magic_bytes(tmp_path: Path) -> None:
    wire = wire_of(lambda request, n: httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"<html>Please log in</html>"))
    result = wire.transport().get_bytes("https://good.example/paper.pdf", tmp_path)

    assert result.response.outcome == "not_a_pdf" and "%PDF-" in result.response.error
    assert list(tmp_path.iterdir()) == []


def test_a_missing_content_type_is_treated_as_octet_stream_not_refused(tmp_path: Path) -> None:
    """Half the repository servers holding OA PDFs send no content-type; the magic check decides."""
    wire = wire_of(lambda request, n: httpx.Response(200, content=PDF_BYTES))
    assert wire.transport().get_bytes("https://good.example/paper.pdf", tmp_path).ok


def test_a_declared_length_over_the_cap_is_refused_before_a_byte_is_read(tmp_path: Path) -> None:
    body: list[int] = []

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        def stream():
            body.append(1)
            yield b"%PDF-" + b"x" * CHUNK
        return httpx.Response(200, headers={"content-type": "application/pdf",
                                            "content-length": str(80 * 1_000_000)},
                              content=stream())

    result = wire_of(handler).transport().get_bytes("https://good.example/p.pdf", tmp_path,
                                                    max_bytes=5e6)
    assert result.response.outcome == "too_large" and body == []


def test_a_download_over_the_size_cap_stops_at_the_limit_mid_stream(tmp_path: Path) -> None:
    """A hostile endless 200 must be cut while it arrives, not after the disk is full."""
    served = {"bytes": 0}

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        def endless():
            served["bytes"] += len(b"%PDF-")
            yield b"%PDF-"
            while True:                                  # no content-length, no end
                served["bytes"] += CHUNK
                yield b"x" * CHUNK
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=endless())

    cap = 5 * CHUNK
    result = wire_of(handler).transport().get_bytes("https://good.example/p.pdf", tmp_path,
                                                    max_bytes=cap)

    assert result.response.outcome == "too_large"
    assert result.path is None and list(tmp_path.iterdir()) == []      # not even a partial file
    # bounded at one chunk past the cap — the connection is dropped AT the limit
    assert served["bytes"] <= cap + 2 * CHUNK


def test_the_search_wide_budget_stops_a_download_the_per_file_cap_would_allow(tmp_path: Path
                                                                              ) -> None:
    """The two caps stay distinct, so `stream_upload`'s two different refusals keep meaning what
    they say: *this file* is too big, versus *this search* is full."""
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        # streamed, so there is no content-length to short-circuit on: the budget must be enforced
        # by the same code that enforces a human upload's total cap
        return httpx.Response(200, headers={"content-type": "application/pdf"},
                              content=iter([PDF_BYTES]))

    budget = ByteBudget(len(PDF_BYTES) - 1)
    result = wire_of(handler).transport(budget=budget).get_bytes(
        "https://good.example/p.pdf", tmp_path, max_bytes=50e6)

    assert result.response.outcome == "too_large" and "total size limit" in result.response.error
    assert list(tmp_path.iterdir()) == []
    assert budget.remaining == len(PDF_BYTES) - 1        # the failed reservation was handed back


def test_an_exhausted_budget_refuses_before_any_request_is_made(tmp_path: Path) -> None:
    wire = wire_of(ok_pdf)
    result = wire.transport(budget=ByteBudget(0)).get_bytes("https://good.example/p.pdf", tmp_path)
    assert result.response.outcome == "too_large" and wire.requests == []


def test_a_successful_download_spends_exactly_what_it_wrote(tmp_path: Path) -> None:
    budget = ByteBudget(10e6)
    result = wire_of(ok_pdf).transport(budget=budget).get_bytes("https://good.example/p.pdf",
                                                                tmp_path, max_bytes=5e6)
    assert result.ok
    assert budget.spent == len(PDF_BYTES)                # not the 5 MB it reserved


def test_a_pdf_that_cannot_be_read_leaves_nothing_behind(tmp_path: Path) -> None:
    """The probe runs against the STAGED file, so bytes never sit under their final name while
    still unproven."""
    seen: list[Path] = []

    def probe(path: Path) -> dict[str, object]:
        seen.append(path)
        assert path.parent != tmp_path               # staged, not yet at its final home
        assert list(tmp_path.glob("*.pdf")) == []
        return {"ok": False, "error": "no readable page"}

    result = wire_of(ok_pdf).transport().get_bytes("https://good.example/p.pdf", tmp_path,
                                                   probe=probe)
    assert result.response.outcome == "unreadable" and "no readable page" in result.response.error
    assert len(seen) == 1 and list(tmp_path.iterdir()) == []


def test_a_probe_that_raises_is_recorded_not_propagated(tmp_path: Path) -> None:
    """The probe parses bytes an attacker chose, so it raising is an ordinary outcome. Letting a
    decompression bomb out of `get_bytes` would end the whole search over one bad paper."""
    def exploding_probe(path: Path) -> dict[str, object]:
        raise RecursionError("maximum recursion depth exceeded parsing the xref table")

    result = wire_of(ok_pdf).transport().get_bytes("https://good.example/p.pdf", tmp_path,
                                                   probe=exploding_probe)
    assert result.response.outcome == "unreadable"
    assert "xref table" in result.response.error
    assert list(tmp_path.iterdir()) == []


def test_a_pdf_that_passes_the_probe_is_moved_into_place(tmp_path: Path) -> None:
    result = wire_of(ok_pdf).transport().get_bytes(
        "https://good.example/p.pdf", tmp_path,
        probe=lambda path: {"ok": True, "n_pages": 3})
    assert result.ok and result.path is not None and result.path.parent == tmp_path


def test_an_http_url_is_upgraded_to_https_once_and_the_rewrite_is_recorded(tmp_path: Path) -> None:
    """~12 % of index-supplied OA PDF URLs are plain http, so a flat refusal would silently cost an
    eighth of the corpus. Nothing is upgraded silently: the rewrite is in the record."""
    wire = wire_of(ok_pdf)
    result = wire.transport().get_bytes("http://good.example/paper.pdf", tmp_path)

    assert result.ok
    assert result.response.rewritten_from == "http://good.example/paper.pdf"
    assert result.response.url == "https://good.example/paper.pdf"
    assert wire.requests[0].url.scheme == "https"


def test_an_index_call_is_never_upgraded_from_http() -> None:
    """The rewrite is a fetch-path concession for publisher URLs, not a general loosening."""
    result = wire_of(ok_pdf).transport().get_json("http://good.example/search")
    assert result.outcome == "refused" and "https" in result.error


# ================================================================================== HostClock
def test_the_courtesy_table_is_polite_where_it_was_measured_to_need_to_be() -> None:
    clock = HostClock()
    assert HOST_INTERVALS["europepmc.org"] >= 2.0        # measured: 200 then 429, no Retry-After
    assert clock.interval_for("europepmc.org") >= 2.0
    assert DEFAULT_HOST_INTERVAL >= 1.0
    assert clock.interval_for("some-publisher-we-have-never-met.org") >= 1.0


def test_a_subdomain_inherits_its_parents_interval() -> None:
    """Europe PMC redirects `?pdf=render` within its own domain; both count as the one host that
    asked us to slow down."""
    assert HostClock().interval_for("www.europepmc.org") >= 2.0


def test_the_clock_spaces_two_calls_to_one_host() -> None:
    ticks = {"now": 100.0}
    clock = HostClock(intervals={"a.example": 4.0}, now=lambda: ticks["now"],
                      sleep=lambda d: ticks.__setitem__("now", ticks["now"] + d))
    assert clock.wait("a.example") == 0.0            # nothing owed on the first request
    assert clock.wait("a.example") == 4.0
    assert clock.wait("b.example") == 0.0            # a different host owes nothing


def test_the_clock_serialises_one_host_across_threads_and_does_not_serialise_two() -> None:
    """The naive version fails here: three threads read the same `last`, compute the same wait and
    fire together, which is the burst the interval existed to prevent."""
    interval = 0.05
    clock = HostClock(default=interval)

    def hammer(hosts: list[str]) -> float:
        start = time.monotonic()
        threads = [threading.Thread(target=clock.wait, args=(h,)) for h in hosts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return time.monotonic() - start

    one_host = hammer(["a.example"] * 3)
    many_hosts = hammer(["p.example", "q.example", "r.example"])

    assert one_host >= 2 * interval * 0.9            # three requests, two intervals of waiting
    assert many_hosts < interval                     # different hosts never wait for each other


# ================================================================================= ByteBudget
def test_a_reservation_that_is_never_settled_is_refunded() -> None:
    budget = ByteBudget(100.0)
    with budget.reserve(40.0) as grant:
        assert grant.granted == 40.0
        assert budget.remaining == 60.0
    assert budget.remaining == 100.0


def test_a_reservation_settles_only_what_was_used() -> None:
    budget = ByteBudget(100.0)
    with budget.reserve(40.0) as grant:
        grant.settle(10.0)
    assert budget.remaining == 90.0


def test_a_reservation_is_capped_by_what_is_left() -> None:
    budget = ByteBudget(30.0)
    with budget.reserve(50.0) as grant:
        assert grant.granted == 30.0
        with budget.reserve(50.0) as second:
            assert second.granted == 0.0             # nothing left to hand out
        grant.settle(30.0)
    assert budget.remaining == 0.0


def test_a_reservation_can_never_be_released_twice() -> None:
    """`settle` and `refund` are a check-then-set; racing them would hand the same allowance back
    twice and inflate the budget above its own total."""
    budget = ByteBudget(100.0)
    grant = budget.reserve(40.0)
    grant.settle(10.0)
    grant.refund()
    grant.settle(0.0)
    assert budget.remaining == 90.0


def test_an_unlimited_budget_grants_what_is_asked() -> None:
    budget = ByteBudget(None)
    with budget.reserve(1e9) as grant:
        assert grant.granted == 1e9
    assert budget.remaining is None


def test_concurrent_workers_can_never_jointly_exceed_the_budget() -> None:
    """Review S1: with a plain `remaining` integer, three workers each pass their own check against
    the same snapshot and jointly write up to three times the allowance. The lock is the fix, and
    this is the test that would have caught it."""
    cap = 25 * CHUNK
    per_file = 10 * CHUNK
    budget = ByteBudget(cap)
    granted: list[float] = []
    lock = threading.Lock()
    ready = threading.Barrier(8)

    def worker() -> None:
        ready.wait()                                  # every thread reserves at the same instant
        with budget.reserve(per_file) as grant:
            with lock:
                granted.append(grant.granted)
            grant.settle(grant.granted)               # each worker writes every byte it was given

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(granted) <= cap                        # the whole point
    assert budget.remaining == 0.0
    assert sum(1 for g in granted if g > 0) <= 3      # 25 MB cannot feed four 10 MB downloads


# ============================================================================ RecordedTransport
def test_an_unrecorded_request_is_a_loud_failure_not_an_empty_answer() -> None:
    """A fake that improvises an empty result turns "the index silently returned nothing" — the
    exact failure this feature exists to prevent — into a green test."""
    transport = RecordedTransport()
    with pytest.raises(MissingSearchFixture) as caught:
        transport.get_json("https://api.example.org/works", params={"q": "tremor"})
    message = str(caught.value)
    assert fixture_key("https://api.example.org/works", {"q": "tremor"}) in message
    assert "https://api.example.org/works" in message
    assert transport.calls[0]["url"] == "https://api.example.org/works"


def test_an_unrecorded_download_is_a_loud_failure_too(tmp_path: Path) -> None:
    with pytest.raises(MissingSearchFixture):
        RecordedTransport().get_bytes("https://good.example/p.pdf", tmp_path)


def test_a_recorded_response_replays_without_any_network() -> None:
    transport = RecordedTransport()
    transport.record("https://api.example.org/works",
                     HttpResponse(url="https://api.example.org/works", status=200,
                                  body=b'{"results": [1, 2]}'),
                     params={"q": "tremor"})
    replayed = transport.get_json("https://api.example.org/works", params={"q": "tremor"})
    assert replayed.json() == {"results": [1, 2]}


def test_a_recorded_key_ignores_parameter_ORDER_but_not_parameter_VALUES() -> None:
    a = fixture_key("https://x.example/w", {"q": "tremor", "rows": 10})
    b = fixture_key("https://x.example/w", {"rows": 10, "q": "tremor"})
    c = fixture_key("https://x.example/w", {"rows": 20, "q": "tremor"})
    assert a == b and a != c


def test_a_replayed_download_goes_through_the_same_stream_upload_rules(tmp_path: Path) -> None:
    import hashlib

    source = tmp_path / "fixture.pdf"
    source.write_bytes(PDF_BYTES)
    dest = tmp_path / "papers"
    transport = RecordedTransport(payloads={fixture_key("https://good.example/p.pdf"): source})

    result = transport.get_bytes("https://good.example/p.pdf", dest)
    assert result.ok and result.n_bytes == len(PDF_BYTES)
    assert result.path == dest / f"{hashlib.sha256(PDF_BYTES).hexdigest()}.pdf"


# ================================================================== the record after a failure
def test_a_refused_redirect_still_records_the_hops_it_got_through(tmp_path: Path) -> None:
    """Failure honesty: every recorded outcome must say how far it got, or a reviewer asking
    "what did you not show me?" cannot answer it from the record alone."""
    wire = wire_of(redirect_to("https://127.0.0.1/x"))
    result = wire.transport().get_bytes("https://good.example/paper.pdf", tmp_path)
    assert result.response.hops == ("https://good.example/paper.pdf",)


def test_a_body_refused_after_two_hops_records_the_status_and_both_hops(tmp_path: Path) -> None:
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if n == 1:
            return httpx.Response(302, headers={"location": "https://also-good.example/real.pdf"})
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")

    result = wire_of(handler).transport().get_bytes("https://good.example/p.pdf", tmp_path)
    assert result.response.outcome == "not_a_pdf"
    assert result.response.status == 200
    assert result.response.url == "https://also-good.example/real.pdf"
    assert result.response.hops == ("https://good.example/p.pdf",
                                    "https://also-good.example/real.pdf")


def test_a_recorded_failure_replays_as_that_failure(tmp_path: Path) -> None:
    transport = RecordedTransport()
    transport.record("https://good.example/p.pdf",
                     HttpResponse(status=429, outcome="rate_limited", error="slow down"))
    result = transport.get_bytes("https://good.example/p.pdf", tmp_path)
    assert result.response.outcome == "rate_limited" and result.path is None
