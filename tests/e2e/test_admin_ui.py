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
import subprocess

import httpx
import pytest
from conftest import API_KEY, MODEL_ID, build_binaries, model_dir, running_stack
from test_admin_jobs import STUB_FILES, STUB_SHA, stub_hub  # noqa: F401 (fixture)

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
