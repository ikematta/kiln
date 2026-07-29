"""Admin UI operator-flow e2e (SPEC §12 Phase 10: "admin UI performs a
full load/serve/unload cycle").

Drives the ACTUAL embedded SPA with a real browser (playwright), not the
admin API: connect with the admin token, watch the model's status and
token counters update live over the /admin/stats SSE stream, unload and
reload the model from the table, toggle pinning, and launch a download
job against the local stub hub — files land on disk.

Browser: the machine's installed Chrome (present on dev machines and
GitHub macOS runners), falling back to playwright's own chromium. Skips
with an actionable message when neither exists, unless
KILN_E2E_REQUIRE_BROWSER=1 (set in CI) makes that a failure.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import threading

import httpx
import pytest
from conftest import (
    API_KEY,
    MODEL_ID,
    QWEN_MODEL_ID,
    build_binaries,
    model_dir,
    pinned_model_dir,
    running_stack,
)
from test_add_model import add_stack, assert_only_inserted, dir_hub
from test_admin_jobs import STUB_FILES, STUB_SHA, stub_hub  # noqa: F401 (fixture)
from test_priority import prompt_for

ADMIN_TOKEN = "kiln-e2e-ui-admin-token"

# The exact fail-closed message (crates/kiln-gateway/src/error.rs
# admin_disabled) — the UI must surface it verbatim, because it names the
# fix.
ADMIN_DISABLED_MESSAGE = (
    "The admin API is disabled: set auth.admin_token_hash in kiln.toml "
    "(hash a token with `kiln-gateway hash-key`)."
)


@pytest.fixture(scope="module")
def browser_page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = None
        errors = []
        # Installed Chrome first (no download), then playwright's chromium.
        for kwargs in ({"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(headless=True, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001 (launch error shape varies)
                errors.append(str(exc).splitlines()[0])
        if browser is None:
            message = (
                "no automatable browser: install Google Chrome or run "
                "`uv run --project tests/e2e playwright install chromium` "
                f"(tried: {errors})"
            )
            if os.environ.get("KILN_E2E_REQUIRE_BROWSER"):
                pytest.fail(message)
            pytest.skip(message)
        page = browser.new_page()
        yield page
        browser.close()


@contextlib.contextmanager
def ui_stack(stub_hub_url: str):
    """Rust-worker stack with the admin surface enabled and kiln-jobs
    pointed at the local stub hub."""
    gateway = build_binaries()
    token_hash = subprocess.run(
        [gateway, "hash-key", ADMIN_TOKEN], capture_output=True, text=True, check=True
    ).stdout.strip()
    previous = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = stub_hub_url
    try:
        with running_stack(
            [(MODEL_ID, "rust")],
            extra_toml=f'[auth]\nadmin_token_hash = "{token_hash}"\n',
        ) as stack:
            stack.wait_ready()
            yield stack
    finally:
        if previous is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous


def connect(page, base_url: str, token: str) -> None:
    page.goto(f"{base_url}/ui/")
    page.get_by_test_id("token-input").fill(token)
    page.get_by_test_id("connect").click()


# --- live queue depth + gateway counters -----------------------------------

RL_KEY = "kiln-e2e-ui-ratelimited-key"  # rpm = 1, dedicated to this test
# TTFT budget, and the burst that must overrun it. kiln-engine admits at
# most one prefilling request at a time, so a burst of distinct
# ~1150-token prompts is drained one prefill per step: the tail of a
# 16-deep burst waits for fifteen prefills before its own, which is well
# past 5s on any hardware this suite runs on (measured on an M4: members
# 9+ of a 12-deep burst already overran it). That is what makes the
# timeout counter fire from real queueing rather than from a sleep.
UI_TTFT_S = 5
UI_FLOOD_STREAMS = 16
UI_FLOOD_MAX_TOKENS = 768
# Waiting >= 1 with any running count: "2 / 10". Anchored so a leading zero
# (an empty queue) cannot match.
QUEUE_NONEMPTY = re.compile(r"^[1-9]\d* / \d+$")


@contextlib.contextmanager
def counters_stack():
    """Rust-worker stack with the admin surface enabled, a TTFT budget, and
    a dedicated rpm=1 key — everything needed to drive the three gateway
    counters and the engine queue for real."""
    gateway = build_binaries()

    def hashed(raw: str) -> str:
        return subprocess.run(
            [gateway, "hash-key", raw], capture_output=True, text=True, check=True
        ).stdout.strip()

    # ttft_timeout_secs belongs to [server], so it must precede the [auth]
    # table this appends (running_stack splices this in after [server]'s
    # own keys).
    extra = (
        f"ttft_timeout_secs = {UI_TTFT_S}\n"
        f'[auth]\nadmin_token_hash = "{hashed(ADMIN_TOKEN)}"\n'
        f'[[auth.api_keys]]\nname = "ui-rl"\nkey_hash = "{hashed(RL_KEY)}"\nrpm = 1\n'
    )
    with running_stack([(MODEL_ID, "rust")], extra_toml=extra) as stack:
        stack.wait_ready()
        yield stack


def metric_total(text: str, name: str) -> float:
    """Sum of every series of `name` in a /metrics scrape — the same number
    the dashboard shows as a counter's total."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        if head == name or head.startswith(f"{name}{{"):
            total += float(value)
    return total


def hold_stream(stack, tag: str, stop: threading.Event) -> None:
    """One flood member: a real streaming completion held open until `stop`,
    then abandoned mid-generation (the gateway cancels on disconnect, the
    same as any client going away)."""
    try:
        with httpx.stream(
            "POST",
            f"{stack.base_url}/v1/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": stack.model_id,
                "prompt": prompt_for(tag),
                "max_tokens": UI_FLOOD_MAX_TOKENS,
                "stream": True,
                # No priority: every member and the victim below are one
                # class, so the surplus genuinely queues instead of
                # preempting its way in.
                "temperature": 0,
            },
            timeout=httpx.Timeout(10.0, read=600.0),
        ) as response:
            if response.status_code != 200:
                response.read()
                return
            for _ in response.iter_lines():
                if stop.is_set():
                    return
    except httpx.HTTPError:
        return  # a timed-out or cancelled member is fine; the load is real


def test_admin_ui_live_queue_depth_and_gateway_counters(browser_page):
    """Queue depth and gateway counters in the live view, under REAL load.

    The wiring is not the bar — the numbers moving is. A dozen concurrent
    streams are pushed through the actual API while the dashboard is open,
    and the UI must show the engine's own WAITING/RUNNING counts climbing
    off zero and falling back to it, with no reload. The three gateway
    counters are driven the same way: genuine 429s from an rpm=1 key and a
    genuine 504 from a request that queues past its TTFT budget.

    Cross-checked against /metrics: the SSE payload and the scrape are two
    exposures of the same counters and must agree.
    """
    from playwright.sync_api import expect

    if model_dir() is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    page = browser_page
    with counters_stack() as stack:
        connect(page, stack.base_url, ADMIN_TOKEN)
        expect(page.get_by_test_id("connected")).to_be_visible()

        # Idle baseline. "0 / 0" (not the "–" placeholder) proves the field
        # is genuinely arriving from worker Stats, not merely absent.
        queue = page.get_by_test_id(f"queue-{MODEL_ID}")
        expect(queue).to_have_text("0 / 0", timeout=15_000)

        # The counters section is present and honestly zero before anything
        # has fired, and says out loud that these are cumulative.
        expect(page.get_by_test_id("counters-note")).to_contain_text("cumulative")
        rate_limited = page.get_by_test_id("counter-total-kiln_rate_limited_total")
        timed_out = page.get_by_test_id("counter-total-kiln_timeout_total")
        uncovered = page.get_by_test_id(
            "counter-total-kiln_admission_uncovered_bytes_total"
        )
        expect(rate_limited).to_have_text("0")
        expect(timed_out).to_have_text("0")
        expect(uncovered).to_have_text("0")

        stop = threading.Event()
        threads = [
            threading.Thread(
                target=hold_stream, args=(stack, f"ui-flood-{i}", stop), daemon=True
            )
            for i in range(UI_FLOOD_STREAMS)
        ]
        # Submitted behind the whole burst, so it is the last request the
        # engine can admit — the deterministic end of the TTFT overrun.
        # Runs on its own thread: it blocks until its 504, and the queue
        # observation below must not wait for that.
        victim: list[int] = []

        def submit_victim() -> None:
            try:
                response = httpx.post(
                    f"{stack.base_url}/v1/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": stack.model_id,
                        "prompt": prompt_for("ui-victim"),
                        "max_tokens": 32,
                        "temperature": 0,
                    },
                    timeout=180,
                )
                victim.append(response.status_code)
            except httpx.HTTPError:
                pass

        victim_thread = threading.Thread(target=submit_victim, daemon=True)
        try:
            for thread in threads:
                thread.start()
            victim_thread.start()

            # THE POINT: the live view shows a non-empty engine queue while
            # the flood is in flight — the burst arrives together and the
            # engine drains it one prefill at a time.
            expect(queue).to_have_text(QUEUE_NONEMPTY, timeout=90_000)
            observed = queue.inner_text().strip()
            print(
                f"admin UI queue depth under a {UI_FLOOD_STREAMS}-deep "
                f"flood: {observed}"
            )
            waiting, running = (int(part) for part in observed.split(" / "))
            assert waiting >= 1 and running >= 1, observed
            # Sanity bound, not a tight one: the flood plus the victim is
            # everything this test has in flight.
            assert waiting + running <= UI_FLOOD_STREAMS + 2, (
                f"queue depth {observed} exceeds the requests in flight"
            )

            victim_thread.join(timeout=180)
            print(f"victim submitted behind the burst returned {victim}")

            # Rate limiting: an rpm=1 key spends its single request, then
            # every further call is a 429 counted against the key.
            statuses = []
            for _ in range(3):
                response = httpx.post(
                    f"{stack.base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {RL_KEY}"},
                    json={
                        "model": stack.model_id,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                    },
                    timeout=120,
                )
                statuses.append(response.status_code)
            assert 429 in statuses, f"rpm=1 key was never limited: {statuses}"

            # Both counters climb in the dashboard itself, with no reload.
            # (Timeouts here are the victim's 504 and the burst tail's —
            # every one of them a request that really did wait too long.)
            expect(rate_limited).not_to_have_text("0", timeout=15_000)
            expect(timed_out).not_to_have_text("0", timeout=15_000)
            # Never-fired counter stays an honest zero, byte-formatted.
            expect(uncovered).to_have_text("0")

            # The two exposures agree: what the dashboard renders is what a
            # scrape of the same counter reports.
            scrape = stack.metrics_text()
            for locator, name in (
                (rate_limited, "kiln_rate_limited_total"),
                (timed_out, "kiln_timeout_total"),
            ):
                shown = float(locator.inner_text().strip())
                assert shown >= 1, f"{name} rendered {shown}"
                assert shown == metric_total(scrape, name), (
                    f"{name}: UI shows {shown}, /metrics totals "
                    f"{metric_total(scrape, name)}"
                )
        finally:
            stop.set()
            for thread in threads + [victim_thread]:
                thread.join(timeout=180)

        # Released: the queue drains back to empty in the live view — the
        # numbers track the engine down as well as up.
        expect(queue).to_have_text("0 / 0", timeout=120_000)


def test_admin_ui_full_operator_flow(browser_page, stub_hub):  # noqa: F811
    from playwright.sync_api import expect

    if model_dir() is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    page = browser_page
    with ui_stack(stub_hub) as stack:
        connect(page, stack.base_url, ADMIN_TOKEN)
        expect(page.get_by_test_id("connected")).to_be_visible()

        # The models table shows the loaded model; live numbers arrive with
        # the first SSE frame (the '–' placeholder becomes a counter).
        status = page.get_by_test_id(f"status-{MODEL_ID}")
        expect(status).to_have_text("ready")
        tokens = page.get_by_test_id(f"tokens-{MODEL_ID}")
        expect(tokens).to_have_text("0", timeout=10_000)

        # Live stats via SSE: a completion served over the API moves the
        # UI's token counter without any reload or click.
        response = httpx.post(
            f"{stack.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Count to five."}],
                "max_tokens": 16,
            },
            timeout=120,
        )
        assert response.status_code == 200, response.text
        generated = response.json()["usage"]["completion_tokens"]
        assert generated > 0
        expect(tokens).to_have_text(str(generated), timeout=15_000)

        # Unload from the table: Drain -> SIGTERM ladder runs behind a 202;
        # the status cell tracks it to "unloaded (admin)" over SSE.
        page.get_by_test_id(f"unload-{MODEL_ID}").click()
        expect(status).to_have_text("unloaded (admin)", timeout=60_000)

        # Load it back and watch it return to ready (worker respawn + model
        # load; generous timeout for CI).
        page.get_by_test_id(f"load-{MODEL_ID}").click()
        expect(status).to_have_text("ready", timeout=300_000)

        # Pin, cross-check against the API (the UI really wrote it), unpin.
        pinned = page.get_by_test_id(f"pinned-{MODEL_ID}")
        expect(pinned).to_have_text("no")
        page.get_by_test_id(f"pin-{MODEL_ID}").click()
        expect(pinned).to_have_text("yes", timeout=10_000)
        listed = httpx.get(
            f"{stack.base_url}/admin/models",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        ).json()
        assert listed["models"][0]["pinned"] is True
        page.get_by_test_id(f"pin-{MODEL_ID}").click()
        expect(pinned).to_have_text("no", timeout=10_000)

        # Job launcher: a real download through the gateway-spawned
        # kiln-jobs against the stub hub, progress watched in the jobs
        # table (first submit also spawns the jobs server; allow for it).
        dest = stack.runtime_dir / "ui-download"
        page.get_by_test_id("dl-repo").fill("stub/tiny")
        page.get_by_test_id("dl-dest").fill(str(dest))
        page.get_by_test_id("dl-submit").click()
        job_state = page.locator("[data-testid^='job-state-']").first
        expect(job_state).to_have_text("succeeded", timeout=60_000)
        for name, data in STUB_FILES.items():
            assert (dest / name).read_bytes() == data
        assert (dest / ".kiln-revision").read_text() == f"stub/tiny@{STUB_SHA}\n"


def test_admin_ui_add_model_full_flow(browser_page):
    """The whole Add Model story through the real UI: HF repo id in, memory
    estimate against the machine's live budget, download progress from the
    job launcher, auto-registration, "load now", and a completion served on
    the new model — all on one gateway process (zero restarts), with
    kiln.toml gaining exactly one [[model]] block and nothing else."""
    from playwright.sync_api import expect
    from test_add_model import ADMIN_TOKEN as ADD_ADMIN_TOKEN

    if model_dir() is None:
        pytest.skip(
            f"pinned test model '{MODEL_ID}' not found; run ./scripts/fetch-test-model.sh"
        )
    qwen_dir = pinned_model_dir(QWEN_MODEL_ID)
    if qwen_dir is None:
        pytest.skip(
            f"pinned test model '{QWEN_MODEL_ID}' not found; run "
            "./scripts/fetch-test-model.sh"
        )
    page = browser_page
    # A stub hub serving the REAL qwen model from disk: the download is a
    # genuine multi-hundred-MB transfer and the result genuinely serves.
    with dir_hub("stub/qwen3", qwen_dir) as hub_url:
        with add_stack([(MODEL_ID, "rust")], hub_url) as (stack, _dest_root):
            config_path = stack.runtime_dir / "kiln.toml"
            before = config_path.read_text()

            connect(page, stack.base_url, ADD_ADMIN_TOKEN)
            expect(page.get_by_test_id("connected")).to_be_visible()

            # Memory estimate before committing to the download: hub-listed
            # weight bytes against the live budget ledger.
            page.get_by_test_id("add-path").fill("stub/qwen3")
            page.get_by_test_id("add-estimate").click()
            estimate = page.get_by_test_id("add-estimate-text")
            expect(estimate).to_be_visible(timeout=15_000)
            expect(estimate).to_contain_text("needs ~")
            expect(estimate).to_contain_text("free of")

            # Add: not downloaded yet, so the download-job flow runs first
            # (progress visible), then auto-registers — one continuous flow.
            page.get_by_test_id("add-id").fill("qwen-added")
            page.get_by_test_id("add-submit").click()
            expect(page.get_by_test_id("add-progress")).to_be_visible(timeout=30_000)
            status_line = page.get_by_test_id("add-status")
            expect(status_line).to_contain_text(
                "registered — persisted to", timeout=180_000
            )

            # The new model appears in the live table, never loaded yet.
            row_status = page.get_by_test_id("status-qwen-added")
            expect(row_status).to_have_text("unloaded (registered)", timeout=15_000)

            # "Load now" → ready via the existing lifecycle machinery.
            page.get_by_test_id("add-load-now").click()
            expect(row_status).to_have_text("ready", timeout=300_000)

            # Servable through the API — same gateway process throughout.
            completion = httpx.post(
                f"{stack.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "qwen-added",
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 16,
                },
                timeout=120,
            )
            assert completion.status_code == 200, completion.text
            assert completion.json()["usage"]["completion_tokens"] > 0
            assert stack.gateway.poll() is None, "gateway must not have restarted"

            # kiln.toml: every pre-existing line intact, exactly one block
            # added, and it points at the downloaded local dir.
            after = config_path.read_text()
            inserted = assert_only_inserted(before, after)
            assert any('id = "qwen-added"' in line for line in inserted), inserted
            assert any("stub--qwen3" in line for line in inserted), inserted


def test_admin_ui_surfaces_disabled_admin_verbatim(browser_page):
    """No admin_token_hash: the UI must show the exact fail-closed 403
    message (which names the fix), not a generic error."""
    from playwright.sync_api import expect

    page = browser_page
    with running_stack([]) as stack:  # model-less, no admin token
        stack.wait_ready()
        connect(page, stack.base_url, "anything")
        expect(page.get_by_test_id("banner")).to_have_text(ADMIN_DISABLED_MESSAGE)
        expect(page.get_by_test_id("connected")).not_to_be_visible()


def test_ui_shell_is_served_embedded():
    """The rust-embed path itself: /ui/ answers the prerendered shell."""
    with running_stack([]) as stack:
        stack.wait_ready()
        response = httpx.get(f"{stack.base_url}/ui/", timeout=10)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Kiln Admin" in response.text


def shell_asset_links(html: str) -> list[str]:
    """Every asset URL the shell references (modulepreload/stylesheet
    hrefs and dynamic-import specifiers — SvelteKit emits them all as
    ./_app/... relative links)."""
    links = re.findall(r'(?:href|src)="([^"]+)"', html)
    links += re.findall(r'import\("([^"]+)"\)', html)
    return [link for link in links if "_app/" in link]


def test_ui_no_trailing_slash_full_chain():
    """Regression (blank-page bug): GET /ui without the trailing slash
    must redirect to /ui/ BEFORE the shell is served. The shell's asset
    links are relative (./_app/...), so a document served at bare /ui
    resolves them to /_app/... — every asset 404s and the page renders
    blank. Verify the whole chain, not just the redirect status: follow
    the redirect, land on /ui/, then fetch every asset the shell links,
    resolved exactly as a browser would against the final document URL —
    each must genuinely 200."""
    with running_stack([]) as stack:
        stack.wait_ready()
        # The redirect itself: permanent, to /ui/ (relative links resolve
        # only under the slashed URL).
        bare = httpx.get(f"{stack.base_url}/ui", timeout=10)
        assert bare.status_code == 308, bare.status_code
        assert bare.headers["location"] == "/ui/"

        # The full chain a browser walks: /ui -> /ui/ -> shell -> assets.
        page = httpx.get(f"{stack.base_url}/ui", follow_redirects=True, timeout=10)
        assert page.status_code == 200
        assert str(page.url).endswith("/ui/")
        assert "text/html" in page.headers["content-type"]
        links = shell_asset_links(page.text)
        assert links, f"shell references no assets? {page.text[:500]}"
        for link in links:
            resolved = page.url.join(link)
            asset = httpx.get(str(resolved), timeout=10)
            assert asset.status_code == 200, (
                f"{link} -> {resolved}: {asset.status_code}"
            )

        # Negative control — proof this test catches the original bug:
        # the same links resolved against un-redirected bare /ui (what a
        # browser does when the shell is served there directly) point at
        # /_app/... and do NOT resolve.
        for link in links:
            broken = httpx.URL(f"{stack.base_url}/ui").join(link)
            assert "/_app/" in broken.path and not broken.path.startswith("/ui/"), (
                broken
            )
            assert httpx.get(str(broken), timeout=10).status_code == 404, broken
