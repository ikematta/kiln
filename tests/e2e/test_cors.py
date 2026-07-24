"""CORS allowlist e2e (SPEC §8.3, `server.cors_origins`).

CORS is enforced by the BROWSER, not the server: the server only emits
headers, and the browser decides whether page JS gets the response (or,
for preflighted requests, whether the real request is sent at all). So
header inspection via httpx can only prove the default-off case; the
allow/deny semantics are driven through a real browser (playwright, same
fixture as the admin-UI suite) loading a real minimal HTML page that
calls fetch() against the gateway.

The preflight case is exercised explicitly and verified END TO END: a
chat completion (JSON body + Content-Type + Authorization) is never a
CORS "simple request", so the browser first sends OPTIONS and only on a
correct answer proceeds to the POST. Both legs are asserted server-side
via kiln_http_requests_total — an OPTIONS sample AND a POST sample for
the allowed origin; an OPTIONS sample and NO new POST for the
unconfigured origin. That distinction is exactly what curl-level checks
miss: a plausible-looking OPTIONS response proves nothing unless the
browser demonstrably followed through.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import pathlib
import tempfile
import threading

import httpx
import pytest
from conftest import API_KEY, MODEL_ID, model_dir, running_stack
from test_admin_ui import browser_page  # noqa: F401 (fixture)

# The probe page. Served from a throwaway local origin; reads the gateway
# URL, API key, and model id from the query string, runs one simple
# request (GET /healthz — no custom headers, so no preflight) and one
# preflighted request (the chat completion), and reports each outcome in
# a DOM node for playwright to read. "blocked" = the fetch promise
# rejected, which is how the browser surfaces a CORS denial to page JS.
PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Kiln CORS probe</title>
<pre id="simple">pending</pre>
<pre id="chat">pending</pre>
<pre id="reqid">pending</pre>
<pre id="done">no</pre>
<script>
const q = new URLSearchParams(location.search);
const gw = q.get("gw");
const put = (id, text) =>
  (document.getElementById(id).textContent = text);
(async () => {
  try {
    const r = await fetch(gw + "/healthz");
    put("simple", r.ok ? "ok" : "status-" + r.status);
  } catch {
    put("simple", "blocked");
  }
  try {
    const r = await fetch(gw + "/v1/chat/completions", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "authorization": "Bearer " + q.get("key"),
      },
      body: JSON.stringify({
        model: q.get("model"),
        messages: [{ role: "user", content: "Say hello." }],
        max_tokens: 8,
      }),
    });
    const body = await r.json();
    put("chat", r.ok && body.choices ? "ok" : "status-" + r.status);
    put("reqid", r.headers.get("x-request-id") || "unreadable");
  } catch {
    put("chat", "blocked");
  }
  put("done", "yes");
})();
</script>
"""


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextlib.contextmanager
def page_origin():
    """Serves the probe page from a fresh 127.0.0.1 port; yields the
    origin URL. Two instances = two distinct origins (ports differ)."""
    with tempfile.TemporaryDirectory(prefix="kiln-cors-") as root:
        (pathlib.Path(root) / "index.html").write_text(PAGE)
        handler = functools.partial(QuietHandler, directory=root)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


def http_count(stack, method: str, path: str, status: str) -> int:
    """Sum of kiln_http_requests_total samples matching all three labels
    (matched by substring so label ordering can't bite)."""
    total = 0
    for line in stack.metrics_text().splitlines():
        if not line.startswith("kiln_http_requests_total{"):
            continue
        if (
            f'method="{method}"' in line
            and f'path="{path}"' in line
            and f'status="{status}"' in line
        ):
            total += int(float(line.rsplit(" ", 1)[1]))
    return total


def test_cors_configured_and_unconfigured_origins(browser_page):  # noqa: F811
    """One stack with cors_origins=[origin A]; the same probe page loaded
    from origin A (everything works, preflight included) and from origin
    B (browser blocks everything, and the preflighted POST is never
    sent)."""
    from playwright.sync_api import expect

    if model_dir() is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    page = browser_page
    with page_origin() as allowed, page_origin() as other:
        extra = f'cors_origins = ["{allowed}"]\n'
        with running_stack([(MODEL_ID, "rust")], extra_toml=extra) as stack:
            stack.wait_ready()
            query = f"?gw={stack.base_url}&key={API_KEY}&model={MODEL_ID}"

            # --- allowed origin: simple AND preflighted requests succeed.
            page.goto(f"{allowed}/index.html{query}")
            expect(page.locator("#done")).to_have_text("yes", timeout=120_000)
            expect(page.locator("#simple")).to_have_text("ok")
            expect(page.locator("#chat")).to_have_text("ok")
            # expose_headers made x-request-id readable by page JS (a
            # non-safelisted response header is invisible otherwise).
            reqid = page.locator("#reqid").text_content()
            assert reqid and reqid not in ("unreadable", "pending"), reqid

            # Server-side proof of the full preflight dance: the browser
            # sent OPTIONS, got a correct answer, and then REALLY sent
            # the POST (the "ok" above came from a genuine completion).
            chat = "/v1/chat/completions"
            assert http_count(stack, "OPTIONS", chat, "200") >= 1
            posts_after_allowed = http_count(stack, "POST", chat, "200")
            assert posts_after_allowed >= 1

            # --- unconfigured origin: same page, same gateway, blocked.
            healthz_before = http_count(stack, "GET", "/healthz", "200")
            options_before = http_count(stack, "OPTIONS", chat, "200")
            page.goto(f"{other}/index.html{query}")
            expect(page.locator("#done")).to_have_text("yes", timeout=60_000)
            expect(page.locator("#simple")).to_have_text("blocked")
            expect(page.locator("#chat")).to_have_text("blocked")

            # Enforcement is client-side and the ledger shows it: the
            # simple GET did reach the server (its response was withheld
            # from JS), the preflight did reach the server (answered
            # without allow-origin), and the real POST was NEVER sent —
            # the browser refused after the failed preflight.
            assert http_count(stack, "GET", "/healthz", "200") > healthz_before
            assert http_count(stack, "OPTIONS", chat, "200") > options_before
            assert http_count(stack, "POST", chat, "200") == posts_after_allowed


def test_no_cors_headers_by_default():
    """Default config ships no CORS machinery at all: no allow-origin on
    any response, and a preflight OPTIONS falls through to the API
    surface's auth route_layer, which 401s it (browsers send no
    credentials on preflights) — byte-for-byte the pre-CORS behavior,
    and precisely why the opt-in layer must answer preflights before
    auth."""
    with running_stack([]) as stack:
        stack.wait_ready()
        origin = {"Origin": "http://localhost:5173"}
        plain = httpx.get(f"{stack.base_url}/healthz", headers=origin, timeout=10)
        assert plain.status_code == 200
        assert "access-control-allow-origin" not in plain.headers

        preflight = httpx.request(
            "OPTIONS",
            f"{stack.base_url}/v1/chat/completions",
            headers={
                **origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
            timeout=10,
        )
        assert preflight.status_code == 401
        assert "access-control-allow-origin" not in preflight.headers
